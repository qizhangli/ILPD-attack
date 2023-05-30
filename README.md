# ILPD-attack
Code for our paper [Improving Adversarial Transferability via Intermediate-level Perturbation Decay](https://arxiv.org/abs/2304.13410).

## Requirements
* Python 3.8.8
* PyTorch 1.12.0
* Torchvision 0.13.0
* timm 0.6.11
  
## Datasets
Select images from ImageNet validation set, and write ```.csv``` file as following:
```
class_index, class, image_name
0,n01440764,ILSVRC2012_val_00002138.JPEG
2,n01484850,ILSVRC2012_val_00004329.JPEG
...
```

## Attack and Evaluate
### Attack
Perform attack:
```
python3 attack.py --model-name ${SOURCE_MODEL_NAME} --data-dir ${IMAGENET_VAL_DIR} --data-info-dir ${DATASET_CSV_FILE} --save-dir ${ADV_IMG_SAVE_DIR}
```
For ``` --model-name ```, use the model name in [timm](https://github.com/huggingface/pytorch-image-models). For instance, ``` --model-name tv_resnet50 ```.
### Evaluate
Evaluate the success rate of adversarial examples:
```
python3 test.py --dir ${ADV_IMG_SAVE_DIR} --model-name ${VICTIM_MODEL_NAME} --log-dir {RESULTS_LOG_DIR}
```
For ``` --model-name ```, use the model name in [timm](https://github.com/huggingface/pytorch-image-models). Separate different victim models using commas. For instance, ``` --model-name resnet50,vit_base_patch16_224 ```.

## Acknowledgements
The following resources are very helpful for our work:

* [timm](https://github.com/huggingface/pytorch-image-models)

## Citation
Please cite our work in your publications if it helps your research:

```
@article{li2023improving,
  title={Improving Adversarial Transferability via Intermediate-level Perturbation Decay},
  author={Li, Qizhang and Guo, Yiwen and Zuo, Wangmeng and Chen, Hao},
  journal={arXiv preprint arXiv:2304.13410},
  year={2023}
}
```
