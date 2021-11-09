import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)

class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean, rgb_std, sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        self.requires_grad = False

class BasicBlock(nn.Sequential):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, bias=False,
        bn=True, act=nn.ReLU(True)):

        m = [nn.Conv2d(
            in_channels, out_channels, kernel_size,
            padding=(kernel_size//2), stride=stride, bias=bias)
        ]
        if bn: m.append(nn.BatchNorm2d(out_channels))
        if act is not None: m.append(act)
        super(BasicBlock, self).__init__(*m)

class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: m.append(nn.BatchNorm2d(n_feat))
            if i == 0: m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res


class GSM(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(GSM, self).__init__()
        m = []
        for i in range(1):
            m.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: m.append(nn.BatchNorm2d(n_feat))
            if i == 0: m.append(act)
        self.mean_conv = conv(n_feat, n_feat, kernel_size, bias=bias)
        self.var_conv = nn.Sequential(conv(n_feat, n_feat, 1, bias=bias))
        self.sigmoid =nn.Sigmoid()
        self.body = nn.Sequential(*m)
        self.res_scale = res_scale
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        res = self.body(x)

        # v = self.relu(self.mean_conv(res))
        # var = self.relu(self.var_conv(res))
        # a = 1 / (1 + var)

        # v = self.mean_conv(res)
        # var = abs(self.var_conv(res))
        # a = 1/(1+var)

        # v = self.relu(self.mean_conv(res))
        # var = self.relu(self.var_conv(res))
        # a = var

        v = self.relu(self.mean_conv(res))
        var = self.sigmoid(self.var_conv(res))
        a = var

        return x+torch.mul(a,v)
        # return res+torch.mul(a,v)
        # return res+torch.mul(a,v)*self.res_scale

class GSM_HT(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(GSM_HT, self).__init__()
        m = []
        for i in range(1):
            m.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: m.append(nn.BatchNorm2d(n_feat))
            if i == 0: m.append(act)
        self.mean_conv = conv(n_feat, n_feat, kernel_size, bias=bias)
        self.var_conv = nn.Sequential(conv(n_feat, n_feat, 1, bias=bias))
        self.sigmoid =nn.Sigmoid()
        self.body = nn.Sequential(*m)
        self.res_scale = res_scale
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        res = self.body(x)

        # v = self.relu(self.mean_conv(res))
        # var = self.relu(self.var_conv(res))
        # a = 1 / (1 + var)

        # v = self.mean_conv(res)
        # var = abs(self.var_conv(res))
        # a = 1/(1+var)

        # v = self.relu(self.mean_conv(res))
        # var = self.relu(self.var_conv(res))
        # a = var

        v = self.mean_conv(res)
        var = self.sigmoid(self.var_conv(res))
        a = var

        return self.relu(x+torch.mul(a,v))
        # return self.relu(torch.mul(a,x)+torch.mul(1.0-a,v))


class GSM_multiscale(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(GSM_multiscale, self).__init__()
        m = []
        mean_convs = []
        scale = 4
        width = int(n_feat/scale)
        if scale == 1:
            self.nums = 1
        else:
            self.nums = scale - 1
        for i in range(self.nums):
            mean_convs.append(conv(width, width, kernel_size, bias=bias))

        self.mean_convs = nn.ModuleList(mean_convs)
        var_convs = []
        for i in range(self.nums):
            var_convs.append(conv(width, width, 1, bias=bias))
        self.var_convs = nn.ModuleList(var_convs)
        self.sigmoid =nn.Sigmoid()
        self.body = nn.Sequential(*m)
        self.res_scale = res_scale
        self.relu = nn.ReLU(inplace=True)
        self.scale = scale
        self.width = width

    def forward(self, x):

        spx = torch.split(x, self.width, 1)


        for i in range(self.nums):
            if i == 0:
                sp = spx[i]
            else:
                sp = sp + spx[i]
            v = self.relu(self.mean_convs[i](sp))
            var = self.sigmoid(self.var_convs[i](sp))
            sp = torch.mul(var, v)
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)

        if self.scale != 1 :
            out = torch.cat((out, spx[self.nums]), 1)

        return out



class Bottle2neck(nn.Module):


    def __init__(self, inplanes, planes, expansion = 1, stride=1, downsample=None, baseWidth=16, scale=4, stype='normal'):
        """ Constructor
        Args:
            inplanes: input channel dimensionality
            planes: output channel dimensionality
            stride: conv stride. Replaces pooling layer.
            downsample: None when stride = 1
            baseWidth: basic width of conv3x3
            scale: number of scale.
            type: 'normal': normal set. 'stage': first block of a new stage.
        """
        super(Bottle2neck, self).__init__()
        self.expansion = expansion
        width = int(math.floor(planes * (baseWidth / 64.0)))

        self.conv1 = nn.Conv2d(inplanes, width * scale, kernel_size=1, bias=False)


        if scale == 1:
            self.nums = 1
        else:
            self.nums = scale - 1
        if stype == 'stage':
            self.pool = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)
        convs = []

        for i in range(self.nums):
            convs.append(nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=1, bias=False))

        self.convs = nn.ModuleList(convs)


        self.conv3 = nn.Conv2d(width * scale, planes * self.expansion, kernel_size=1, bias=False)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stype = stype
        self.scale = scale
        self.width = width

    def forward(self, x):
        residual = x

        out = self.conv1(x)


        out = self.relu(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if i == 0 or self.stype == 'stage':
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.relu(sp)
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)

        if self.scale != 1 and self.stype == 'normal':
            out = torch.cat((out, spx[self.nums]), 1)
        elif self.scale != 1 and self.stype == 'stage':
            out = torch.cat((out, self.pool(spx[self.nums])), 1)

        out = self.conv3(out)


        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        # print(out)
        return out

class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feat, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feat, 4 * n_feat, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn: m.append(nn.BatchNorm2d(n_feat))
                if act: m.append(act())
        elif scale == 3:
            m.append(conv(n_feat, 9 * n_feat, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn: m.append(nn.BatchNorm2d(n_feat))
            if act: m.append(act())
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)
