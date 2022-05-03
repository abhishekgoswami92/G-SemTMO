"""
This file creates the dataset for GCN training. The images are loaded and preprocessed - features extracted and saved as npy files
which constitute as the dataset to be loaded while training.
"""

import numpy as np
from matplotlib import pyplot as plt
from glob import glob
from Graph_utils import adjacency as adj
import cv2
import tifffile as tif
import os
import Graph_utils.Im2Graph as I2G
from Graph_utils.mini_histogram import histogram


def imshow(image, title):
    plt.imshow(image)
    plt.title(title)
    plt.show()


def getrgbfeature(seg, rgb):
    unique = np.unique(seg)
    medrgb_per_seg = []
    stdrgb_per_seg = []
    for label in unique:
        boolmask = np.where(seg == label, rgb, np.nan)
        med = np.nanmedian(boolmask, axis=(0, 1)).astype('float16')
        std = np.nanstd(boolmask, axis=(0, 1)).astype('float16')
        medrgb_per_seg.append(list(med))
        stdrgb_per_seg.append(list(std))
    return medrgb_per_seg, stdrgb_per_seg


def getlumfeature(seg, lum):
    unique = np.unique(seg)
    med_per_seg = []
    for label in unique:
        boolmask = np.where(seg[:, :, 0] == label, lum, 0)
        lum_med = np.median(boolmask[np.nonzero(boolmask)])
        med_per_seg.append(lum_med)
    return med_per_seg


if __name__ == "__main__":

    lum_factor = np.array([.299, .587, .114])

    seg_mask = 'Data/Input/Segmap/'                  # directory for segmentation masks 100x100
    raw = 'Data/Input/Raw/'                          # directory for input tiff images
    ds_add = 'Data/Preprocessed Dataset/'            # directory for saving preprocessed dataset
    gt_add = 'Data/Input/Groundtruth/'               # directory for GT images

    for fn in glob(raw + '*.tiff'):

        lin_image = tif.imread(fn).astype('float32') / 2**16
        name = os.path.basename(fn).split('.')[0]
        lin_im_resized = cv2.resize(lin_image, (100, 100), interpolation=cv2.INTER_AREA)

        seg_im = cv2.imread(f'{seg_mask}{name}.png')
        adj_mat = adj.get_adj(seg_im)

        gt = cv2.imread(f'{gt_add}{name}.png').astype('float32') / 2 ** 8
        gt = cv2.resize(gt, (100, 100), interpolation=cv2.INTER_AREA)
        gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)

        labels = list(np.unique(seg_im))

        rgb_vec_x = np.reshape(lin_im_resized, [-1, 3]).astype('float32')
        rgb_vec_gt = np.reshape(gt, [-1, 3]).astype('float32')
        lum_x = np.matmul(lin_im_resized, lum_factor)
        lum_gt = np.matmul(gt, lum_factor)
        edge_idx = I2G.get_edgeindex(adj_mat)

        med_rgb_x, std_rgb_x = getrgbfeature(seg_im, lin_im_resized)
        med_lum_x = getlumfeature(seg_im, lum_x)
        med_lum_gt = getlumfeature(seg_im, lum_gt)
        histo = histogram(seg_im, lin_im_resized ** (1/2.2))

        seg_vector = seg_im[:, :, 0].ravel()
        np.save(f'{ds_add}{name}_segvec.npy', seg_vector)
        np.save(f'{ds_add}{name}_rgb_x.npy', rgb_vec_x)
        np.save(f'{ds_add}{name}_labels.npy', labels)
        np.save(f'{ds_add}{name}_rgb_gt.npy', rgb_vec_gt.astype('float32'))
        np.save(f'{ds_add}{name}_medseg_x.npy', med_lum_x)
        np.save(f'{ds_add}{name}_medseg_gt.npy', med_lum_gt)
        np.save(f'{ds_add}{name}_eidx.npy', edge_idx)
        np.save(f'{ds_add}{name}_medrgb_x.npy', med_rgb_x)
        np.save(f'{ds_add}{name}_stdrgb_x.npy', std_rgb_x)
        np.save(f'{ds_add}{name}_histo32.npy', histo.astype('float'))
