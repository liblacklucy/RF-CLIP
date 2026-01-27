import logging
import sys
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.structures import PixelData
from mmseg.models.data_preprocessor import SegDataPreProcessor
from mmseg.models.segmentors import BaseSegmentor
from mmseg.registry import MODELS
from torchvision import transforms
from PIL import Image
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.metrics.pairwise import cosine_distances
import re

import clip
from prompts.imagenet_template import openai_imagenet_template
from pamr import PAMR

sys.path.append("..")


@MODELS.register_module()
class RF_CLIP(BaseSegmentor):
    def __init__(self, 
                clip_path,
                name_path,
                device=torch.device('cuda'),
                pamr_steps=10, 
                pamr_stride=(8, 16), 
                scale_up=False,
                prob_thd=0.0, 
                logit_scale=40, 
                slide_stride=112, 
                slide_crop=224, 
                global_semantics_weight = 0.0, 
                mean_vector_weight = 0.0, 
                h_threshold = 0.0, 
                dataset = './configs/cfg_voc20.py'):

        data_preprocessor = SegDataPreProcessor(mean=[122.771, 116.746, 104.094], std=[68.501, 66.632, 70.323], rgb_to_bgr=True)
        super().__init__(data_preprocessor=data_preprocessor)
        self.net, _ = clip.load(clip_path, device=device, jit=False)
        self.scale_up = scale_up
        if scale_up:
            self.cls_net, _ = clip.load('ViT-L/14', device=device, jit=False)
        else:
            self.cls_net = self.net
        self.query_words, self.query_idx = get_cls_idx(name_path)
        # print(self.query_words)  # ['aeroplane', 'bicycle', 'bird', 'ship', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'table', 'dog', \\
        # 'horse', 'motorbike', 'person', 'person in shirt', 'person in jeans', 'person in dress', 'person in sweater', 'person in skirt', \\
        # 'person in jacket', 'pottedplant', 'sheep', 'sofa', 'train', 'television monitor', 'tv monitor', 'monitor', 'television', 'screen']
        # print(self.query_idx)  # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 14, 14, 14, 14, 14, 14, 15, 16, 17, 18, 19, 19, 19, 19, 19]
        self.num_queries = len(self.query_words)
        self.query_idx = torch.Tensor(self.query_idx).to(torch.int64).to(device)
        self.global_semantics_weight = global_semantics_weight
        self.mean_vector_weight = mean_vector_weight
        self.h_threshold = h_threshold
        self.dataset = re.search(r'cfg_(\w+)\.py', dataset).group(1)
        self.query_features = self._encode_text_features(self.query_words, self.net)  # [num_queries, 512]
        if self.scale_up:
            self.cls_query_features = self._encode_text_features(self.query_words, self.cls_net)
        else:
            self.cls_query_features = self.query_features
        self.cluster_info = self.hierarchical_clustering(self.query_features)
        # print(self.cluster_info)  # [7 6 15 8 11 9 10 14 19 12 21 14 12 6 2 3 5 2 0 4 1 16 13 20 9 17 17 17 17 18]
        self.dtype = self.query_features.dtype
        # self.net.visual.set_params(arch, attn_strategy, gaussian_std)
        self.logit_scale = logit_scale
        self.prob_thd = prob_thd
        self.slide_stride = slide_stride
        self.slide_crop = slide_crop
        self.align_corners = False
        self.pamr = PAMR(pamr_steps, dilations=pamr_stride).to(device) if pamr_steps > 0 else None
        self.dino_model = None
        self.inv_normalize = transforms.Normalize(
                mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],
                std=[1/0.229, 1/0.224, 1/0.255]
            )

    def _encode_text_features(self, words, net):
        features = []
        with torch.no_grad():
            for word in words:
                query = clip.tokenize([temp(word) for temp in openai_imagenet_template]).to('cuda')
                feature = net.encode_text(query)
                feature /= feature.norm(dim=-1, keepdim=True)
                feature = feature.mean(dim=0)
                feature /= feature.norm()
                features.append(feature.unsqueeze(0))
        return torch.cat(features, dim=0)

    def hierarchical_clustering(self, text_embeddings):
        text_embeddings = text_embeddings.detach().cpu().numpy()
        distance_matrix = cosine_distances(text_embeddings)
        condensed_matrix = squareform(distance_matrix)
        Z = linkage(condensed_matrix, method='complete')
        clusters = fcluster(Z, t=self.h_threshold, criterion='distance') - 1  
        return clusters  # 为每个类别分配一个聚类标签

    def _get_cluster_indices(self, cluster_id: int) -> np.ndarray:
        return np.where(self.cluster_info == cluster_id)[0]

    def _get_most_similar_class(self, similarities: np.ndarray, indices: np.ndarray) -> int:
        similarities_in_cluster = similarities[indices]
        idx_max_similarity = np.argmax(similarities_in_cluster)
        return int(indices[idx_max_similarity])

    def _compute_topk_similarities(
        self, 
        image_features: torch.Tensor, 
        text_embedding: torch.Tensor, 
        k: int = 2
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        similarities = torch.matmul(image_features, text_embedding.t()).squeeze(1)
        k = min(k, image_features.size(0))
        return torch.topk(similarities, k)

    def _compute_adjusted_embedding(
        self, 
        text_embedding: torch.Tensor, 
        mean_vector: torch.Tensor, 
        alpha: float
    ) -> torch.Tensor:
        adjusted = (1 - alpha) * text_embedding + alpha * mean_vector
        return F.normalize(adjusted, dim=0)

    def hierarchical_prompt(
        self, 
        image_features: torch.Tensor, 
        image_class_similarities: torch.Tensor
    ) -> torch.Tensor:
        image_class_similarities_np = image_class_similarities.squeeze(0).detach().cpu().numpy()  # [1, cls] -> [cls]
        image_features = image_features.squeeze(0).to(self.query_features.device)  # [1, n, 512] -> [n, 512]
        text_features = self.query_features.clone().detach().to(self.query_features.device)
        adjusted_text_features = text_features.clone()
        num_clusters = len(set(self.cluster_info))
        for cluster_id in range(num_clusters):
            indices = self._get_cluster_indices(cluster_id)
            class_i = self._get_most_similar_class(image_class_similarities_np, indices)  # image_class_similarities_np包含所有类别（按照论文划分，有的类别有很多不同的命名）的相似度，该函数是从子类别池中选择最相似的子类别
            text_embedding_i = F.normalize(text_features[class_i].unsqueeze(0), dim=1)
            image_features_norm = F.normalize(image_features, dim=1)
            _, topk_indices = self._compute_topk_similarities(
                image_features_norm, 
                text_embedding_i
            )
            topk_image_features = image_features[topk_indices]
            mean_vector = F.normalize(topk_image_features.mean(dim=0), dim=0)  # 图片对应类别的区域特征平均值，使用这个视觉特征加权文本特征
            adjusted_embedding = self._compute_adjusted_embedding(
                text_embedding_i.squeeze(0),
                mean_vector,
                self.mean_vector_weight
            )
            adjusted_text_features[class_i] = adjusted_embedding
        return adjusted_text_features

    def forward_feature(self, img, whole_img, idx=None):
        if type(img) == list:
            img = img[0]
        # inv_image = self.inv_normalize(img.squeeze().cpu()).permute(1, 2, 0).numpy() * 255
        # inv_image = Image.fromarray(inv_image.astype('uint8'))
        image_features, x_last = self.net.encode_image(img, return_all=True, return_cls=False)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        x_last = x_last / x_last.norm(dim=-1, keepdim=True)
        net_to_use = self.cls_net if self.scale_up else self.net
        query_features_to_use = self.cls_query_features if self.scale_up else self.query_features
        img_to_use = whole_img if whole_img is not None else img
        if whole_img is None and not self.scale_up:
            global_clip_sim = x_last @ self.query_features.T
        else:
            global_vector = net_to_use.encode_image(img_to_use, return_all=False, return_cls=True)  # scale_up作用于CLS token
            global_vector = global_vector / global_vector.norm(dim=-1, keepdim=True)
            global_clip_sim = global_vector @ query_features_to_use.T
        adjusted_text_features = self.hierarchical_prompt(image_features, global_clip_sim)  # [b, n, 512] [b, 1, cls]
        logits = image_features @ adjusted_text_features.T
        logits = logits * (1-self.global_semantics_weight) + global_clip_sim.reshape(1,1,logits.shape[-1]).repeat(1,logits.shape[1],1) * self.global_semantics_weight
        patch_size = self.net.visual.patch_size
        w, h = img[0].shape[-2] // patch_size, img[0].shape[-1] // patch_size
        out_dim = logits.shape[-1]
        logits = logits.permute(0, 2, 1).reshape(-1, out_dim, w, h)
        logits = nn.functional.interpolate(logits, size=img.shape[-2:], mode='bilinear', align_corners=self.align_corners)
        # if getattr(self, "__VISUALIZATION__", False):
        #     if idx is not None:  # 裁切可视化
        #         existing = getattr(self, "__image__", {})
        #         existing.update({f"img{idx}": inv_image})
        #         setattr(self, "__image__", existing)
        #         if not hasattr(self, "__whole_image__"):
        #             processed_img = (self.inv_normalize(whole_img.squeeze().cpu()).permute(1, 2, 0).numpy() * 255).astype('uint8')
        #             setattr(self, "__whole_image__", Image.fromarray(processed_img))
        #     else:  # 非裁切
        #         setattr(self, "__whole_image__", inv_image)
        return logits

    def forward_slide(self, img, stride=112, crop_size=224):
        if type(img) == list:
            img = img[0].unsqueeze(0)
        if type(stride) == int:
            stride = (stride, stride)
        if type(crop_size) == int:
            crop_size = (crop_size, crop_size)
        h_stride, w_stride = stride
        h_crop, w_crop = crop_size
        batch_size, _, h_img, w_img = img.shape
        out_channels = self.num_queries
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = img.new_zeros((batch_size, out_channels, h_img, w_img))
        count_mat = img.new_zeros((batch_size, 1, h_img, w_img))
        idx = 0
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = img[:, :, y1:y2, x1:x2]
                if self.dataset == 'city_scapes':
                    crop_seg_logit = self.forward_feature(crop_img, None, idx=idx)
                else:
                    crop_seg_logit = self.forward_feature(crop_img, img, idx=idx)
                preds += nn.functional.pad(crop_seg_logit,
                                           (int(x1), int(preds.shape[3] - x2), int(y1), int(preds.shape[2] - y2)))
                count_mat[:, :, y1:y2, x1:x2] += 1
                idx += 1
        assert (count_mat == 0).sum() == 0
        logits = preds / count_mat
        return logits

    def predict(self, inputs, data_samples):
        if data_samples is not None:
            batch_img_metas = [data_sample.metainfo for data_sample in data_samples]
        else:
            batch_img_metas = [dict(
                ori_shape=inputs.shape[2:],
                img_shape=inputs.shape[2:],
                pad_shape=inputs.shape[2:],
                padding_size=[0, 0, 0, 0])
            ] * inputs.shape[0]
        if self.slide_crop > 0:
            seg_logits = self.forward_slide(inputs, self.slide_stride, self.slide_crop)
        else:
            seg_logits = self.forward_feature(inputs, None)
        img_size = batch_img_metas[0]['ori_shape']
        seg_logits = nn.functional.interpolate(seg_logits, size=img_size, mode='bilinear', align_corners=self.align_corners)
        if self.pamr:
            img = nn.functional.interpolate(inputs, size=img_size, mode='bilinear', align_corners=self.align_corners)
            try:
                seg_logits = self.pamr(img, seg_logits.to(img.dtype)).to(self.dtype)
            except RuntimeError as e:
                logging.warning(f"Couldn't apply PAMR for image {batch_img_metas[0]['img_path'].split('/')[-1]} "
                                f"of size {img_size}, probably due to low memory. Error message: \"{str(e)}\"")
        return self.postprocess_result(seg_logits, data_samples)

    def postprocess_result(self, seg_logits, data_samples):
        batch_size = seg_logits.shape[0]
        for i in range(batch_size):
            seg_logits = seg_logits[i] * self.logit_scale
            seg_logits = seg_logits.softmax(0)  # n_queries * w * h
            num_cls, num_queries = max(self.query_idx) + 1, len(self.query_idx)
            if num_cls != num_queries:
                seg_logits = seg_logits.unsqueeze(0)
                cls_index = nn.functional.one_hot(self.query_idx)
                cls_index = cls_index.T.view(num_cls, num_queries, 1, 1)
                seg_logits = (seg_logits * cls_index).max(1)[0]
            seg_pred = seg_logits.argmax(0, keepdim=True)
            seg_pred[seg_logits.max(0, keepdim=True)[0] < self.prob_thd] = 0
            if data_samples is None:
                return seg_pred
            else:
                data_samples[i].set_data({
                    'seg_logits':
                        PixelData(**{'data': seg_logits}),
                    'pred_sem_seg':
                        PixelData(**{'data': seg_pred})
                })
            # if getattr(self, "__VISUALIZATION__", False):
            #     setattr(self, "__seg_logits__", seg_logits)
        return data_samples

    def _forward(data_samples):
        pass

    def inference(self, img, batch_img_metas):
        pass

    def encode_decode(self, inputs, batch_img_metas):
        pass

    def extract_feat(self, inputs):
        pass

    def loss(self, inputs, data_samples):
        pass


def get_cls_idx(path):
    with open(path, 'r') as f:
        name_sets = f.readlines()
    num_cls = len(name_sets)

    class_names, class_indices = list(), list()
    for idx in range(num_cls):
        names_i = name_sets[idx].split(', ')
        class_names += names_i
        class_indices += [idx for _ in range(len(names_i))]
    class_names = [item.replace('\n', '') for item in class_names]
    return class_names, class_indices