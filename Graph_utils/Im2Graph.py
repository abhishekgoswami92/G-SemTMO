"""
This file can be used to convert the image to a graph needed for GCN with the adjacency information in the COO format as
required by pytorchGeometric.
"""

import tifffile as tif
from matplotlib import pyplot as plt
import numpy as np
import cv2
import torch
from torch_geometric.data import Data
import Graph_utils.adjacency as adj


def imshow(file):
    plt.imshow(file)
    plt.show()


def getobslum(label):
    seg_mask_2d = seg_mask[:, :, 0]
    mask = np.array([seg_mask_2d == label])
    masked_lum = [lum[m] for m in mask]
    print(label, np.median(masked_lum))
    return np.median(masked_lum)


def get_edgeindex(adjdict):
    lbl = list(adjdict.keys())
    ei = []
    for i in lbl:
        for j in adjdict.get(i):
            if not ei:
                ei = [[i, j]]
            else:
                ei.append([i, j])
    return ei


if __name__ == '__main__':
    hdr_add = 'Data/Input/Raw/'
    seg_mask_add = 'Data/Input/Segmap/'
    seg_mask = cv2.imread(seg_mask_add + 'pecklake.png')

    linrgb = tif.imread(hdr_add + 'pecklake.tif')
    linrgb = cv2.resize(linrgb, None, fx=2, fy=2, interpolation=cv2.INTER_AREA)

    adj_dict = adj.get_adj(seg_mask)
    labels = list(adj_dict.keys())
    lum_factor = np.array([.299, .587, .114])
    lum = np.matmul(linrgb, lum_factor)
    edges = []

    edge_idx = get_edgeindex(adj_dict)
    feed_obs_lum = []

    for label in labels:
        obs_lum = getobslum(label)
        feed_obs_lum.append([obs_lum])

    edge_index = torch.tensor(edges, dtype=torch.long)
    x = torch.tensor([feed_obs_lum], dtype=torch.float)
    y = torch.tensor(labels, dtype=torch.float)

    graph = Data(x=x, y=y, edge_index=edge_index)
    print(graph)
