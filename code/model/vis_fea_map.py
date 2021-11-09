#原文链接：https: // blog.csdn.net / weixin_40500230 / article / details / 93845890
import cv2
import time
import os
import matplotlib.pyplot as plt
import torch
from torch import nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np



def draw_features(x, savename):
    img = x[0, 0, :, :]
    pmin = np.min(img)
    pmax = np.max(img)
    img = ((img - pmin) / (pmax - pmin + 0.000001)) * 255  # float在[0，1]之间，转换成0-255
    img = img.astype(np.uint8)  # 转成unit8
    img = cv2.applyColorMap(img, cv2.COLORMAP_JET)  # 生成heat map
    cv2.imwrite(savename,img)





