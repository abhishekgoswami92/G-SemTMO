"""
This file is primarily for computing HyAB colour distance of results from GT. It also contains the PSNR ad MS-SSIM
metric computations.

The first part of the code creates score csv files. The second part can be used to generate plots of the same.
"""

import cv2
import numpy as np
import pandas as pd
from pytorch_msssim import ms_ssim
from math import log10, sqrt
import torch
import os
import torchvision

totensor = torchvision.transforms.ToTensor()


def msssim_compare(image1, image2):
    it1 = totensor(np.array(image1)).unsqueeze(0)
    it2 = totensor(np.array(image2)).unsqueeze(0)
    val = ms_ssim(it1, it2, data_range=1, size_average=False)
    return val.cpu().detach().numpy()


def PSNR(imgref, imgpred):
    mse = np.mean((imgref - imgpred) ** 2)
    if mse == 0:
        return 100
    max_pixel = 1  # 255 if un-normalised, 1 if normalised
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr


def hyAB_torch(imgref, imgpred):
    hyab = torch.sum(torch.abs(imgref[:, 0] - imgpred[:, 0])) + torch.norm(imgref[:, 1:] - imgpred[:, 1:], 2)
    return hyab


def hyAB(imgref, imgpred):
    hyab = np.sum(np.abs(imgref[:, :, 0] - imgpred[:, :, 0])) + np.linalg.norm(imgref[:, :, 1:] - imgpred[:, :, 1:])
    return hyab


# ---------- Change as per code of TMO
codelist = [
    'durand02',
    'mantiuk08',
    'reinhard02',
    'reinhard05',
    'hdrnet16',
    'gstmo'
]

scores_hyab = pd.DataFrame(columns=['Name'])
scores_psnr = pd.DataFrame(columns=['Name'])
scores_mssim = pd.DataFrame(columns=['Name'])

csv_out_dir = 'Data/Output/csv/'
comparison_dir = f'Data/Output/Compare/'


namelist = list(np.load("Data/Preprocessed Dataset/Namelists/4205_names_FiveK", allow_pickle=True))


if __name__ == '__main__':
    if 1:                           # Compute metric scores
        for modelcode in codelist:
            print(modelcode)

            temp_hyab = []
            temp_psnr = []
            temp_mssim = []
            temp_namelist = []
            for testfile in namelist:
                print(testfile)
                if os.path.exists(f'{comparison_dir}{testfile}{modelcode}.png'):
                    imgExp = cv2.imread(f'{comparison_dir}{testfile}.jpg').astype('float32') / 2 ** 8
                    imgExp = cv2.cvtColor(imgExp, cv2.COLOR_BGR2RGB)

                    imgPred = cv2.imread(f'{comparison_dir}{testfile}{modelcode}.png').astype('float32') / 2 ** 8
                    imgPred = cv2.cvtColor(imgPred, cv2.COLOR_BGR2RGB)

                    h, w, _ = imgExp.shape
                    imgExp_lab = cv2.cvtColor(imgExp, cv2.COLOR_RGB2LAB)
                    imgPred_lab = cv2.cvtColor(imgPred, cv2.COLOR_RGB2LAB)

                    error = hyAB(imgExp_lab, imgPred_lab) / (h * w)
                    psnr = PSNR(imgExp, imgPred)
                    mssim = msssim_compare(imgExp, imgPred)
                    temp_hyab.append(error)
                    temp_psnr.append(psnr)
                    temp_mssim.append(mssim)
                    temp_namelist.append(testfile)

            scores_hyab['Name'] = temp_namelist
            scores_mssim['Name'] = temp_namelist
            scores_psnr['Name'] = temp_namelist

            scores_hyab[f'{modelcode}'] = temp_hyab
            scores_mssim[f'{modelcode}'] = temp_mssim
            scores_psnr[f'{modelcode}'] = temp_psnr

        scores_hyab.to_csv(csv_out_dir + 'hyab.csv', index=False)
        scores_mssim.to_csv(csv_out_dir + 'mssim.csv', index=False)
        scores_psnr.to_csv(csv_out_dir + 'psnr.csv', index=False)

    if 0:                           # Plot graphs of the metric scores.
        csv_table = pd.read_csv(f"{csv_out_dir}hyab.csv")
        gstmo_fold = csv_table[codelist[1]]
        gstmo = csv_table[codelist[0]]
        w = .3
        plt.hist(gstmo_fold, bins=np.arange(0, 10 + w, w), alpha=.6, label='G-SemTMO (contrast-filtered)', zorder=3)
        plt.hist(gstmo, bins=np.arange(0, 10 + w, w), alpha=.6, label='G-SemTMO (whole)', zorder=2)

        plt.axvline(x=np.median(gstmo_fold.values), color='red', label='G-SemTMO (contrast-filtered) - median', zorder=5)
        plt.axvline(x=np.median(gstmo.values), color='black', label='G-SemTMO (whole) - median', zorder=5)

        plt.title('HyAB Histogram on LocHDR')

        plt.legend(loc='upper left', fontsize=8)
        plt.grid()
        plt.show()
