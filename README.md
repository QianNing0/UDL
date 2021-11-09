# Uncertainty-Driven Loss for Single Image Super-Resolution
  [[paper]]() [[homepage]](https://see.xidian.edu.cn/faculty/wsdong/)


This repository is Pytorch code for our proposed uncertainty-driven loss (UDL).

The code is built on [RCAN](https://github.com/yulunzhang/RCAN)  and tested on Ubuntu 16.04 environment (Python 3.5/3.6/3.7, PyTorch 1.4.0) with 2080Ti/1080Ti GPUs.


If you find our work useful in your research or publications, please consider citing:

```Bibtex
@inproceedings{ning2021uncertainty,
  title={ Uncertainty-Driven Loss for Single Image Super-Resolution },
  author={ Ning Qian and Dong, WeiSheng and Li, Xin and Wu, Jinjian and Shi, Guangming },
  booktitle={Advances in Neural Information Processing Systems},
  year={2021}
}

```

## Contents
1. [Requirements](#Requirements)
2. [Test](#test)
3. [Acknowledgements](#acknowledgements)

## Requirements
- Python 3 
- skimage
- imageio
- Pytorch (Pytorch version 1.0.1 is recommended)
- tqdm 
- cv2 (pip install opencv-python)

## Train

#### Quick start

```
   cd code
   sh train.sh
   ```

## Test

#### Quick start

#### Test on standard SR benchmark

1. If you have cloned this repository, the pre-trained models can be found in experiment fold and test dataset Set5 can be found in data fold.

2. Then, run command:
   ```
   cd code
   sh test.sh
   ```
3. Finally, PSNR values are shown on your screen, you can find the reconstruction images in `../experiment/xx/results/`


## Acknowledgements
- This code is built on [RCAN (PyTorch)](https://github.com/yulunzhang/RCAN). We thank the authors for sharing their codes.
