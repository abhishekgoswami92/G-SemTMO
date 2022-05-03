"""
This file computes a mini histogram for each node in the graph feature space using the input tif file and the semantic mask.
Bin size is adjustible.
"""

import numpy as np
import cv2
import tifffile as tif


semmap_address = 'Data/Input/Segmap/'        # Directory containign semantic maps with 9 classes
tif_address = 'Data/Input/Raw/'            # Raw images
namelist = list(np.load("Data/Preprocessed Dataset/Namelists/4205_names_FiveK.npy", allow_pickle=True))
save_add = f'Data/Output/'                # Directory to save histogram per class


def histogram(segmap, im):
    labels = np.unique(segmap)
    mask_rgb = np.zeros(im.shape)
    cumu_hist = []
    for l in labels:
        hist = []
        mask_rgb = np.where(segmap == l, im, mask_rgb)
        for i in range(3):
            channel = mask_rgb[np.array(mask_rgb[:, :, i] != 0)][:, i]
            hist_ch, bin_ch = np.histogram(channel, bins=32)
            sum_hist_ch = np.sum(hist_ch)
            norm_hist_ch = hist_ch / (sum_hist_ch / 32)
            for item in norm_hist_ch:
                hist.append(item)
        cumu_hist.append(hist)
    return np.array(cumu_hist)


if __name__ == '__main__':
    for file in namelist:
        tif_im = tif.imread(f'{tif_address}{file}.tiff').astype('float32') / 2 ** 16
        gamma_tif = tif_im ** (1 / 2.2)
        shape = (tif_im.shape[1], tif_im.shape[0])
        sem_map = cv2.imread(f'{semmap_address}{file}.png')
        sem_map = cv2.resize(sem_map, shape, interpolation=cv2.INTER_NEAREST)

        histo = histogram(sem_map, gamma_tif)
        np.save(f'{save_add}{file}_histo32.npy', histo.astype('float'))

