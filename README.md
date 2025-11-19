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
To evaluate CASS on a single benchmark, run the following command:
```shell script
python eval.py --config ./configs/cfg_{benchmark_name}.py --pamr off
```
To evaluate CASS on a all 8 benchmarks, run the following command:
```shell script
bash eval_all.sh
```


## :scroll: Acknowledgement
This repository has been developed based on the [NACLIP](https://github.com/sinahmr/NACLIP) and [SCLIP](https://github.com/wangf3014/SCLIP) repository. Thanks for the great work!