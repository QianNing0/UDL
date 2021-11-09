from model import common

import torch.nn as nn
import torch
import random


def make_model(args, parent=False):
    return EDSR(args)


class EDSR(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(EDSR, self).__init__()

        n_resblock = 16
        n_feats = 64
        self.args = args
        kernel_size = 3
        scale = args.scale[0]
        act = nn.ReLU(True)
        self.up_factor = args.scale[0]

        # define head module
        m_head = [conv(args.n_colors, n_feats, kernel_size)]

        # define body module
        m_body = [
            common.ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=args.res_scale
            ) for _ in range(n_resblock)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        m_tail = [
            common.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, args.n_colors, kernel_size)
        ]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

        self.var_conv = nn.Sequential(*[conv(n_feats, n_feats, kernel_size), nn.ELU(),conv(n_feats, n_feats, kernel_size), nn.ELU(),conv(n_feats, args.n_colors, kernel_size), nn.ELU()])

    def forward(self, x):

        x = self.head(x)
        res = self.body(x)
        res += x
        x = self.tail(res)
        var = self.var_conv(nn.functional.interpolate(res, scale_factor=self.up_factor, mode='nearest'))
        return [x, var]


    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))
