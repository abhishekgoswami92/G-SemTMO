"""
This file computes a multilevel contrast score for each image to approximate how punchy or contrasty they appear.

"""

import cv2
import numpy as np
from matplotlib import pyplot as plt
from glob import glob
import os
import pandas as pd


def imshow(f):
    plt.imshow(f)
    plt.show()


def multilev_contrast(test_im):
    score = []
    for p in range(1, 6):
        M = test_im.shape[0] // p
        N = test_im.shape[1] // p
        patch_contrast = []
        B = [test_im[x:x + M, y:y + N] for x in range(0, test_im.shape[0], M) for y in range(0, test_im.shape[1], N)]
        for i in range(len(B)):
            if B[i].shape[1] == N and B[i].shape[0] == M:
                patch_contrast.append(np.var(B[i]))
        measure = np.sqrt(np.mean(patch_contrast))
        score.append(measure)
    return np.mean(score)


def ch_contrast_score(test_im):
    M = 8
    N = 8
    patch_contrast = []
    ch_patch_contrast = []
    dr_list = []
    ch_dr_list = []
    P = [test_im[x:x + M, y:y + N] for x in range(0, test_im.shape[0], M) for y in range(0, test_im.shape[1], N)]
    for ch in range(3):
        for i in range(len(P)):
            patch = P[i]
            patch = patch[:, :, ch]
            hl = np.max(patch)
            sh = np.min(patch)
            dr = hl - sh
            patch_contrast.append(np.var(patch))
            dr_list.append(dr)
        ch_patch_contrast.append(np.sqrt(np.mean(patch_contrast)))
        ch_dr_list.append(np.mean(dr_list))
    measure = np.max(ch_patch_contrast)
    dr_measure = np.max(ch_dr_list)
    print(measure)
    return measure, dr_measure


def contrast_score(test_im):
    M = 8
    N = 8
    patch_contrast = []
    dr_list = []
    B = [test_im[x:x + M, y:y + N] for x in range(0, test_im.shape[0], M) for y in range(0, test_im.shape[1], N)]
    for i in range(len(B)):
        hl = np.max(B[i])
        sh = np.min(B[i])
        dr = hl - sh
        patch_contrast.append(np.var(B[i]))
        dr_list.append(dr)
    measure = np.sqrt(np.mean(patch_contrast))
    dr_measure = np.mean(dr_list)
    print(measure)
    return measure, dr_measure


if __name__ == "__main__":

    im_dir = 'Data/Output/'
    csv_out = 'Data/Output/csv/'
    namelist_dir = 'Data/Preprocessed Dataset/Namelists/'
    lum_factor = np.array([.299, .587, .114])

    if 1:  # Contrast score computation
        contrast_scorelist = []
        dr_scorelist = []
        namelist = []
        scores_contrast = pd.DataFrame(columns=['Name'])
        for file in glob(im_dir + '*.jpg'):
            im = cv2.imread(file)[:, :, ::-1] / 255.0
            fname = os.path.basename(file).split('.')[0]
            lum_x = np.matmul(im, lum_factor)
            mul_lev_contrast = multilev_contrast(lum_x)

            namelist.append(fname)
            contrast_scorelist.append(mul_lev_contrast)

    if 0:  # contrast score visualisation
        contrast_csv = pd.read_csv(csv_out + '781_multilev_contrast_scores.csv')
        # LC-> Low contrast, HC-> High contrast, invert the sort order
        LC_200 = contrast_csv.sort_values('Multilev contrast score').head(200)
        LC_200 = LC_200.sort_values('Name')['Name'].values
        np.save(f'{namelist_dir}/200_filtered_Lcontrast.npy', LC_200)

        scores = contrast_csv['Multilev contrast score'].values
        plt.hist(scores, bins=100)
        plt.axvline(x=0.162, color='red')
        plt.title('Multi-Level Contrast Histogram')
        plt.savefig('Data/Output/plots/200 Filtered LC images.png')
