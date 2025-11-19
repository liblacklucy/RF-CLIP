# Target Refocusing via Attention Redistribution for Open-Vocabulary Semantic Segmentation: An Explainability Perspective (AAAI 2026)

Authors: Jiahao Li, Yang Lu, Yachao Zhang, Yong Xie*, Fangyong Wang, Yuan Xie, Yanyun Qu*.     *Corresponding author

[[paper]()] 

> **Open-vocabulary semantic segmentation (OVSS) employs pixel-level vision-language alignment to associate category-related prompts with corresponding pixels. A key challenge is enhancing the multimodal dense prediction capability, specifically this pixel-level multimodal alignment. Although existing methods achieve promising results by leveraging CLIP’s vision-language alignment, they rarely investigate the performance boundaries of CLIP for dense prediction from an interpretability mechanisms perspective. In this work, we systematically investigate CLIP's internal mechanisms and identify a critical phenomenon: analogous to human distraction, CLIP diverts significant attention resources from target regions to irrelevant tokens. Our analysis reveals that these tokens arise from dimension-specific over-activation; filtering them enhances CLIP's dense prediction performance. Consequently, we propose ReFocusing CLIP (RF-CLIP), a training-free approach that emulates human distraction-refocusing behavior to redirect attention from distraction tokens back to target regions, thereby refining CLIP's multimodal alignment granularity. Our method achieves SOTA performance on eight benchmarks while maintaining high inference efficiency.** 
>
> <p align="center">
> <img width="800" src="figs/overview.png">
> </p>

## News
* **2025-11** :loudspeaker: Our work, [RF-CLIP](), has been accepted by AAAI 2026.
* **2025-12** :rocket: We will release the code for RF-CLIP.

## Installation

### Requirements
```shell script
pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 -f https://download.pytorch.org/whl/cu111/torch_stable.html
pip install openmim scikit-learn scikit-image
mim install mmcv==2.0.1 mmengine==0.8.4 mmsegmentation==1.1.1
pip install ftfy regex numpy==1.23.5 yapf==0.40.1
```

### Download Datasets
We include the listed dataset configurations in this repo, following [SCLIP](https://github.com/wangf3014/SCLIP): PASCAL VOC (with and without the background category), PASCAL Context (with and without the background category), Cityscapes, ADE20k, COCO-Stuff164k, and COCO-Object.

Please follow the [MMSeg data preparation document](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/user_guides/2_dataset_prepare.md) to download and pre-process the datasets. The COCO-Object dataset can be converted from COCO-Stuff164k by executing the following command:

```shell script
python ./datasets/cvt_coco_object.py PATH_TO_COCO_STUFF164K -o PATH_TO_COCO_OBJECT
```

## Evaluation
To evaluate CASS on a all 8 benchmarks, run the following command:
```shell script
bash eval_all.sh
```


## :scroll: Acknowledgement
This repository has been developed based on the [NACLIP](https://github.com/sinahmr/NACLIP), [SCLIP](https://github.com/wangf3014/SCLIP), and[CASS](https://github.com/MICV-yonsei/CASS) repository. Thanks for the great work!