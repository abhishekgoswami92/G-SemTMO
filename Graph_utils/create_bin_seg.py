"""
This file merges the 150 semantic labels to 9. Given a semantic mask with unmerged semantic labels, it merges them and creates
n binary semantic maps each consisting a unique merged semantic label (class).
"""

import numpy as np
import cv2
import os
from matplotlib import pyplot as plt
from glob import glob
import imageio


def imshow(file):
    plt.imshow(file)
    plt.show()


# sky=[3]
# s = 1
# mountain=[14,17,35,47,69,92,95]
# m =2
# vegetation=[5,10,18,30,33,67,73,126]
# v=3
# water= [22,27,61,110,114,129]
# w=4
# human= [13,127]
# h=5
# nonhuman= [21,77,81,84,91,103,104,105,115,117,128]
# n=6
# city=[2,7,12,26,39,43,44,49,52,53,54,55,60,62,80,85,94,97,101,102,107,116,122]
# c=7
# room= [1,4,6,8,9,11,15,16,19,20,23,24,25,28,29,31,32,34,36,40,41,42,45,46,48,50,51,56,57,58,59,65,71,72,74,76,82,89,98,100,111,146]
# r=8
# other= [37,38,63,64,66,68,70,75,78,79,83,86,87,88,93,96,99,106,108,109,112,113,118,119,120,121,123,124,125,130,131,132,134,135,136,138,139,140,142,143,144,145,147,148,149,150]
# o=9

dictionary = {
    1: [3],
    2: [14, 17, 35, 47, 69, 92, 95],
    3: [5, 10, 18, 30, 33, 67, 73, 126],
    4: [22, 27, 61, 110, 114, 129],
    5: [13, 127],
    6: [21, 77, 81, 84, 91, 103, 104, 105, 115, 117, 128, 133],
    7: [2, 7, 12, 26, 39, 43, 44, 49, 52, 53, 54, 55, 60, 62, 80, 85, 94, 97, 101, 102, 107, 116, 122, 137],
    8: [1, 4, 6, 8, 9, 11, 15, 16, 19, 20, 23, 24, 25, 28, 29, 31, 32, 34, 36, 40, 41, 42, 45, 46, 48, 50, 51, 56, 57,
        58, 59, 65, 71, 72, 74, 76, 82, 89, 98, 100, 111, 146],
    9: [37, 38, 63, 64, 66, 68, 70, 75, 78, 79, 83, 86, 87, 88, 90, 93, 96, 99, 106, 108, 109, 112, 113, 118, 119, 120,
        121, 123, 124, 125, 130, 131, 132, 134, 135, 136, 138, 139, 140, 141, 142, 143, 144, 145, 147, 148, 149, 150]
}

lookuptable = np.zeros(151, dtype='uint8')
for val, llist in dictionary.items():
    for i in range(1, 151):
        if i in llist:
            lookuptable[i] = val

img_address = 'Data/Input/Gamma Corrected/'
save_address = 'Data/Input/Segmap/'
for fn in glob(img_address + '*.png'):
    im = cv2.imread(fn)
    fname = os.path.basename(fn).split('.')[0]
    if not os.path.exists(save_address + str(fname)):
        os.mkdir(save_address + str(fname))
    im = im[:, :, 0]
    im_label = lookuptable[im + 1]
    unique = np.unique(im_label)
    print('Mapped labels: ', unique)
    fn = os.path.join(save_address + fname, '_segmap_fullsize.png'.format(i))
    imageio.imsave(fn, im_label)

    if 0:            # To obtain binary segmented maps per semantic merged class
        for i in unique:
            bin_mask = im_label.copy()
            bin_mask[bin_mask != i] = 0
            bin_mask[bin_mask == i] = 255
            fn = os.path.join(save_address + fname, 'map_{}.png'.format(i))
            imageio.imsave(fn, bin_mask)
