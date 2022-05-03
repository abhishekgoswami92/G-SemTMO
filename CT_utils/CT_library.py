"""
This file has the colour transfer computations, RGB 2 LAB colour space
"""

import math as m
import torch

# Matrix definitions
CovRGB2LMS = torch.tensor([[0.3811, 0.5783, 0.0402],
                           [0.1967, 0.7244, 0.0782],
                           [0.0241, 0.1288, 0.8444]]).cuda()

m1 = torch.tensor([[1 / m.sqrt(3), 0, 0],
                   [0, 1 / m.sqrt(6), 0],
                   [0, 0, 1 / m.sqrt(2)]]).cuda()

m2 = torch.tensor([[1, 1, 1],
                   [1, 1, -2],
                   [1, -1, 0]]).float().cuda()

CovLMS2lab = torch.matmul(m1, m2)
Covlab2LMS = torch.inverse(CovLMS2lab)
CovLMS2RGB = torch.tensor([[4.4679, -3.5873, 0.1193],
                           [-1.2186, 2.3809, -0.1624],
                           [0.0497, -0.2439, 1.2045]]).cuda()


def rgb2lab(rgb):
    lms = torch.matmul(rgb, CovRGB2LMS.t())
    loglms = torch.log10(lms)
    lab = torch.matmul(loglms, CovLMS2lab.t())
    lab[torch.isnan(lab)] = 0
    lab[lab == float('-inf')] = 0
    return lab


def transfer_dist(src, srcmean, seg_p):

    shift = seg_p[:3]
    factor = seg_p[-3:]

    src = src - srcmean
    src = src * factor + shift

    return src


def lab2rgb(lab):

    loglms = torch.matmul(lab, Covlab2LMS.t())
    lms = torch.pow(10, loglms)
    rgb = torch.matmul(lms, CovLMS2RGB.t())

    return rgb


