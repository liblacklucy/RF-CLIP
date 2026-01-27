### CLIP source code from OpenAI:
# https://github.com/openai/CLIP/blob/main/clip/clip.py

from collections import OrderedDict
from typing import Tuple, Union

import copy
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops import DeformConv2d
from scipy import ndimage
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
import pdb
from skimage import io, segmentation, color, transform
from skimage.segmentation import mark_boundaries, find_boundaries
from collections import defaultdict
from PIL import Image
from torchvision import transforms


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        # return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]
        return self.attn(x, x, x, need_weights=True, average_attn_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


def get_attention_hook(layer_idx, attention_results):
    def hook(module, input, output):
        attention = output[1]
        # print(attention.shape)
        attention_results[layer_idx] = attention
    return hook


class VisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.patch_size = patch_size
        self.output_dim = output_dim
        self.width = width
        self.heads = heads
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

        # 存储每层自注意力矩阵
        self.attention_results = {}
        self.qq_attention_results = {}
        self.kk_attention_results = {}
        self.vv_attention_results = {}
        # self.q_results = {}
        # self.k_results = {}
        # self.v_results = {}
        # self.featmap_results = {}
        # for layer_idx, layer in enumerate(self.transformer.resblocks):
        #     layer.attn.register_forward_hook(get_attention_hook(layer_idx, self.attention_results))
        # self.inv_normalize = transforms.Normalize(
        #         mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],
        #         std=[1/0.229, 1/0.224, 1/0.255]
        #     )

    def attn_refocusing(self, attn_layer, x, **kwargs):
        config = {
            'return_attn': False,
            'with_attn': False,
            'strategy': 'vanilla', # 'refocus' or 'vanilla'
            'threshold': 0.1,  # sink点阈值
            'beta': 0.6,  # 衰减系数，衰减越多，感受野越小
        }
        config.update(kwargs)
        num_heads = attn_layer.num_heads
        num_tokens, bsz, embed_dim = x.size()
        head_dim = embed_dim // num_heads
        scale = head_dim ** -0.5
        q, k, v = F.linear(x, attn_layer.in_proj_weight, attn_layer.in_proj_bias).chunk(3, dim=-1)
        q = q.contiguous().view(num_tokens, bsz * num_heads, head_dim).transpose(0, 1)
        k = k.contiguous().view(num_tokens, bsz * num_heads, head_dim).transpose(0, 1)
        v = v.contiguous().view(num_tokens, bsz * num_heads, head_dim).transpose(0, 1)
        attn_weights = torch.bmm(q * scale, k.transpose(1, 2))
        attn_weights = F.softmax(attn_weights, dim=-1)  # [num_heads*bsz, N+1, N+1]
        cur_layer_idx = config["cur_layer_idx"]
        threshold = config['threshold']  # 界定高注意力token
        beta =config['beta']  # 注意力调整系数
        final_attn = attn_weights.clone()  # [num_heads*bsz, N+1, N+1]
        if config['strategy'] == 'refocus':
            # curr_attn = attn_weights  # [num_heads*bsz, N+1, N+1]
            n_heads, N, _ = final_attn.shape
            for head_idx in range(n_heads):
                head_attn = final_attn[head_idx]  # [N+1, N+1]
                attn_score = head_attn[1:, 1:].sum(dim=0)  # [N, ]
                sink_idx = (attn_score / N > threshold).nonzero().squeeze(dim=-1)  # [N, ] 这里的索引对应原注意力矩阵的索引+1的位置
                if sink_idx.numel() > 0:  # 如果存在sink_idx,则对该head进行注意力调整
                    original_sink_weights = head_attn[1:, sink_idx+1].sum(dim=1)  # 记录原始权重（衰减前）
                    head_attn[1:, sink_idx+1] *= beta  # 应用衰减，不修改cls token
                    available_weights = original_sink_weights * (1 - beta)  # 计算可用权重
                    copied_head_attn = head_attn[1:, 1:].clone()  # 创建副本并屏蔽sink列
                    copied_head_attn[:, sink_idx] = 0
                    denominator = copied_head_attn.sum(dim=1, keepdim=True)
                    ratios = copied_head_attn / (denominator + 1e-8)
                    head_attn[1:,1:] += available_weights.unsqueeze(1) * ratios  # 权重再分配
        self.attention_results[cur_layer_idx] = final_attn.view(bsz, num_heads, num_tokens, num_tokens)  # [bsz, num_heads, N+1, N+1]
        kk_attn_weights = torch.bmm(k * scale, k.transpose(1, 2))
        kk_attn_weights = F.softmax(kk_attn_weights, dim=-1)  # [num_heads*bsz, N+1, N+1]
        # qq_attn_weights = torch.bmm(q * scale, q.transpose(1, 2))
        # qq_attn_weights = F.softmax(qq_attn_weights, dim=-1)  # [num_heads*bsz, N+1, N+1]
        # vv_attn_weights = torch.bmm(v * scale, v.transpose(1, 2))
        # vv_attn_weights = F.softmax(vv_attn_weights, dim=-1)  # [num_heads*bsz, N+1, N+1]
        # self.qq_attention_results[cur_layer_idx] = qq_attn_weights
        self.kk_attention_results[cur_layer_idx] = kk_attn_weights
        # self.vv_attention_results[cur_layer_idx] = vv_attn_weights
        if config['return_attn']:
            return final_attn
        attn_output = torch.bmm(final_attn, v)
        attn_output = attn_output.transpose(0, 1).contiguous().view(num_tokens, bsz, embed_dim)
        attn_output = attn_layer.out_proj(attn_output)
        return attn_output if not config['with_attn'] else (attn_output, final_attn)

    def last_attn(self, attn_layer, x, **kwargs):
        config = {
            'return_attn': False,
            'with_attn': False,
        }
        config.update(kwargs)
        num_heads = attn_layer.num_heads
        num_tokens, bsz, embed_dim = x.size() 
        head_dim = embed_dim // num_heads
        scale = head_dim ** -0.5
        q, k, v = F.linear(x, attn_layer.in_proj_weight, attn_layer.in_proj_bias).chunk(3, dim=-1)
        q = q.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        kk_attn_weights = torch.bmm(k * scale, k.transpose(1, 2))
        kk_attn_weights = F.softmax(kk_attn_weights, dim=-1)  # [num_heads*bsz, N+1, N+1]
        # qq_attn_weights = torch.bmm(q * scale, q.transpose(1, 2))
        # qq_attn_weights = F.softmax(qq_attn_weights, dim=-1)  # [num_heads*bsz, N+1, N+1]
        # vv_attn_weights = torch.bmm(v * scale, v.transpose(1, 2))
        # vv_attn_weights = F.softmax(vv_attn_weights, dim=-1)  # [num_heads*bsz, N+1, N+1]
        # self.qq_attention_results[len(self.transformer.resblocks)-1] = qq_attn_weights
        self.kk_attention_results[len(self.transformer.resblocks)-1] = kk_attn_weights
        # self.vv_attention_results[len(self.transformer.resblocks)-1] = vv_attn_weights
        q, v, k = q[:, 1:, :], v[:, 1:, :], k[:, 1:, :]
        attn_weights_list = [self.kk_attention_results[i][:, 1:, 1:] for i in range(len(self.transformer.resblocks))]  # [num_heads, N, N]
        attn_weights = torch.stack(attn_weights_list, dim=0).mean(dim=0)
        # if getattr(self, "__VISUALIZATION__", False):
        #     setattr(self, "__last_attn__", attn_weights)
        if config['return_attn']:
            return attn_weights
        attn_output = torch.bmm(attn_weights, v)
        attn_output = attn_output.transpose(0, 1).contiguous().view(-1, bsz, embed_dim)
        attn_output = attn_layer.out_proj(attn_output)
        if config['with_attn']:
            return attn_output, attn_weights
        return attn_output

    def feat_denoising(self, x, n_patches, sink_channel=[4, 162, 189, 326, 429, 474, 633, 713], sigma=5.0, alpha=0.4, last=False):  # VIT/B-16
    # def feat_denoising(self, x, n_patches, sink_channel=[250, 261, 437, 650, 720, 779, 936, 1005], sigma=5.0, alpha=0.4, last=False):  # VIT/L-14
        x_clone = x.clone()  # [N+1, B, dim]
        feat = x_clone[1:, :, :] if not last else x_clone  # [N, B, dim]
        feat_min = feat.min(dim=-1, keepdim=True)[0]
        feat_norm = (feat - feat_min) / (feat - feat_min).sum(dim=-1, keepdim=True).clamp(min=1e-8)
        sink_feat = feat_norm[:, :, sink_channel]  # [N, B, d']
        sink_value, _ = sink_feat.max(dim=-1)
        threshold = sigma * (1.0 / feat.shape[-1])
        mask = (sink_value > threshold)  # [N, B]
        B, N, C = feat.shape[1], feat.shape[0], feat.shape[2]
        H, W = n_patches
        feat_2d = feat.view(H, W, B, C).permute(2, 3, 0, 1)  # [B, C, H, W]
        avg_conv = nn.Conv2d(C, C, kernel_size=3, padding=1, bias=False, groups=C)
        nn.init.constant_(avg_conv.weight, 1/9)  # TODO: 不再使用均值，而是直接用周边3*3的最小值
        avg_conv = avg_conv.to(feat.device, feat.dtype)
        neighbor_avg = avg_conv(feat_2d)  # [B, C, H, W]
        mask_2d = mask.view(H, W, B).permute(2, 0, 1)  # [B, H, W]
        feat_denoised = torch.where(
            mask_2d.unsqueeze(1),  # 扩展为 [B, 1, H, W]
            neighbor_avg,
            feat_2d
        )  # [B, C, H, W]
        feat_processed = feat_denoised.permute(2, 3, 0, 1).view(N, B, C)  # [N, B, dim]
        if not last:
            x_clone[1:] = (1-alpha)*x_clone[1:] + alpha*feat_processed
        else:
            x_clone = (1-alpha)*x_clone + alpha*feat_processed
        return x_clone

    def feat_enhancing(self, x, x_guided, beta=0.4, last=False, n_patches=None):
        """
        args:
        x: [N+1,b,d]
        x_guided: [N+1, b, d]
        return:
        x: [N+1,b,d]
        simi_return: [N+1, b, d]
        """
        if not last:
            x_guided = x_guided[1:]
        N = x_guided.shape[0]
        simi = F.cosine_similarity(
            x_guided.unsqueeze(2),  # [N, b, 1, d]
            x_guided.permute(1, 0, 2).unsqueeze(1),  # [b, 1, N, d]
            dim=-1
        ).permute(1, 0, 2)  # [N, b, N] -> [b, N, N]
        simi_return = simi.clone()
        # simi[simi < beta] = -float('inf')
        # simi_normalized = F.softmax(simi, dim=-1)
        simi[simi < beta] = 0
        if last:
            x = torch.matmul(simi, x.permute(1, 0, 2)).permute(1, 0, 2)
        else:
            x[1:] = torch.matmul(simi, x[1:].permute(1, 0, 2)).permute(1, 0, 2)
        return x, simi_return
    
    def forward(
            self, 
            x: torch.Tensor, 
            return_all=True, 
            return_cls=False
            ):
        B, nc, w, h = x.shape
        n_patches = (w // self.patch_size, h // self.patch_size)
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        if x.shape[1] != self.positional_embedding.shape[0]:
            x = x + self.interpolate_pos_encoding(x, w, h).to(x.dtype)
        else:
            x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x_guide = []
        for i, blk in enumerate(self.transformer.resblocks[:-1], start=0):
            x = x + self.attn_refocusing(blk.attn, blk.ln_1(x), cur_layer_idx=i)
            x = x + blk.mlp(blk.ln_2(x))
            x = self.feat_denoising(x, n_patches)
            x_guide.append(x)
        blk = self.transformer.resblocks[-1]
        x_last = blk(x)
        x_last = self.feat_denoising(x_last, n_patches)
        x_guide.append(x_last)
        x_last = x_last.permute(1, 0, 2)  # LND -> NLD [197, 1, 768]
        x_last = self.ln_post(x_last[:,0,:]) # [1, 768]
        x_last = x_last @ self.proj # [1,512]
        if return_cls:
            return x_last
        x_decoded = []
        x_guide_tensor = torch.stack(x_guide, dim=0).mean(dim=0)
        x, _ = self.feat_enhancing(x, x_guide_tensor, n_patches=n_patches)
        x_decoded.append(self.last_attn(blk.attn, blk.ln_1(x)))
        x_guide_tensor, _ = self.feat_enhancing(x_guide_tensor, x_guide_tensor, n_patches=n_patches)
        x_decoded.append(self.last_attn(blk.attn, blk.ln_1(x_guide_tensor)))
        x = torch.stack(x_decoded, dim=0).mean(dim=0)
        x = x.permute(1, 0, 2)  # LND -> NLD
        if return_all:
            return self.ln_post(x) @ self.proj, x_last
        x = self.ln_post(x[:, 0, :])
        if self.proj is not None:
            x = x @ self.proj
        return x

    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - 1
        N = self.positional_embedding.shape[0] - 1
        if npatch == N and w == h:
            return self.positional_embedding
        class_pos_embed = self.positional_embedding[[0]]
        patch_pos_embed = self.positional_embedding[1:]
        dim = x.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2), mode='bicubic',
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)), align_corners=False, recompute_scale_factor=False
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)


class CLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,  # 512
                 # vision
                 image_resolution: int,  # 224
                 vision_layers: Union[Tuple[int, int, int, int], int],  # 12
                 vision_width: int,  # 768
                 vision_patch_size: int,  # 16
                 # text
                 context_length: int,  # 77
                 vocab_size: int,  # 49408
                 transformer_width: int,  # 512
                 transformer_heads: int,  # 8
                 transformer_layers: int  # 12
                 ):
        super().__init__()
        self.context_length = context_length

        vision_heads = vision_width // 64
        self.visual = VisionTransformer(
            input_resolution=image_resolution,
            patch_size=vision_patch_size,
            width=vision_width,
            layers=vision_layers,
            heads=vision_heads,
            output_dim=embed_dim
        )

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image, return_all=False, return_cls=False):
        return self.visual(image.type(self.dtype), return_all=return_all, return_cls=return_cls)

    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        return x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

    def forward(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        # normalized features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text


def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


def build_model(state_dict: dict):
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))

    model = CLIP(
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    convert_weights(model)
    model.load_state_dict(state_dict)
    return model.eval()
