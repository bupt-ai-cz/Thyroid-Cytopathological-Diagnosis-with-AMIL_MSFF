# Attention Based Multi-Instance Thyroid Cytopathological Diagnosis with Multi-Scale Feature Fusion

Implementation detail of our paper ["Attention Based Multi-Instance Thyroid Cytopathological Diagnosis with Multi-Scale Feature Fusion"](https://ieeexplore.ieee.org/abstract/document/9413184).

## Citation
Please cite this paper in your publications if it helps your research:
```
@inproceedings{qiu2021attention,
  title={Attention Based Multi-Instance Thyroid Cytopathological Diagnosis with Multi-Scale Feature Fusion},
  author={Qiu, Shuhao and Guo, Yao and Zhu, Chuang and Zhou, Wenli and Chen, Huang},
  booktitle={2020 25th International Conference on Pattern Recognition (ICPR)},
  pages={3536--3541},
  year={2021},
  organization={IEEE}
}
```
## Requirements
* Python == 3.5
* Pytorch == 1.3.0
* torchvision == 0.4.1

## Run
run `main.py` to start training and testing, the options are as follows:
* `--dataset`: dataset to train with [thyroid | breast]
* `--backbone`: select backbone [resnet18 | resnet34]
* `--method`: select method [B | BF | BFA]
* `--mode`: select mode [train | test]
* `--save_weight`: path to save checkpoint
* `--load_weight`: path to load checkpoint
* `--random_seed`: set random seed
* `--gpu`: gpu index

To train with our method:

`python main.py --dataset thyroid --backbone resent18 --method BFA --mode train`

To test with our best model:

`python main.py --dataset thyroid --backbone resent18 --method BFA --mode train --load_weight ./weight/best.pth`


## Contact
* email: qiushuhao@bupt.edu.cn; czhu@bupt.edu.cn
