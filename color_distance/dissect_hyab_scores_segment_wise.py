"""
Break HyAB scores per semantic segment of image into luminance and chrominance scores.
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt
from glob import glob
from pytorch_msssim import ms_ssim
from math import log10, sqrt
import torch
import os
import torchvision

totensor = torchvision.transforms.ToTensor()


def get_mask_median(image, fwk):
    imshow(image)
    masked = np.where(fwk > 0, image, np.nan)
    median = np.nanmedian(masked)
    return median


def imshow(im):
    plt.imshow(im)
    plt.show()


def get_mask_count(fwk):
    x = [i for i in fwk[:, :, 0].flatten() if i > 0]
    return len(x)


def get_mask_rgb(image, fwk):
    masked = np.zeros(image.shape).astype('float32')
    masked = np.where(fwk > 0, image, masked)
    return masked


def msssim_compare(image1, image2):
    it1 = totensor(np.array(image1)).unsqueeze(0)
    it2 = totensor(np.array(image2)).unsqueeze(0)
    val = ms_ssim(it1, it2, data_range=1, size_average=False)
    return val.cpu().detach().numpy()


def PSNR(gt, pred):
    mse = np.mean((gt - pred) ** 2)
    if mse == 0:
        return 100
    max_pixel = 1  # 255 if un-normalised, 1 if normalised
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr


def hyAB_torch(imgref, imgpred):
    hyab = torch.sum(torch.abs(imgref[:, 0] - imgpred[:, 0])) + torch.norm(imgref[:, 1:] - imgpred[:, 1:], 2)
    return hyab


def hyAB(imgref, imgpred):
    hyab = np.sum(np.abs(imgref[:, :, 0] - imgpred[:, :, 0])) + np.sum(np.linalg.norm(imgref[:, :, 1:] - imgpred[:, :, 1:], axis=(0, 1)))
    return hyab


def L_error(imgref, imgpred):
    l_err = np.sum(np.abs(imgref[:, :, 0] - imgpred[:, :, 0]))
    return l_err


def Chroma_error(imgref, imgpred):
    chrm_err = np.sum(np.linalg.norm(imgref[:, :, 1:] - imgpred[:, :, 1:], axis=(0, 1)))
    return chrm_err


codelist = [
    '_200 filtered HC_ExpI_ab2_fold3',
    '_200 filtered HC_ExpI_fold3_gstmo',
]

csv_out_dir = 'Data/Output/csv/'
semfwk_add = 'Data/Input/Semfwk/'
comparison_dir = 'Data/Output/Compare/'
lum_factor = np.array([.299, .587, .114])

namelist = ['a2381']
if __name__ == '__main__':
    if 1:
        for modelcode in codelist:
            print(modelcode)

            temp_hyab = []
            temp_psnr = []
            temp_mssim = []
            temp_namelist = []
            for testfile in namelist:
                if os.path.exists(f'{comparison_dir}{testfile}{modelcode}.png'):
                    for file in glob(semfwk_add + f'{testfile}/*.png'):
                        sem_name = os.path.basename(file).split('.')[0]
                        sem_fwk = cv2.imread(file)
                        pixel_count_seg = get_mask_count(sem_fwk)
                        if pixel_count_seg > 0:

                            imgExp = cv2.imread(f'{comparison_dir}{testfile}.jpg').astype('float32')/2**8
                            w, h, _ = imgExp.shape
                            imgExp = cv2.cvtColor(imgExp, cv2.COLOR_BGR2RGB)
                            Lab_expert_whole = cv2.cvtColor(imgExp, cv2.COLOR_RGB2LAB)
                            maskedExp = get_mask_rgb(imgExp, sem_fwk)

                            imgPred = cv2.imread(f'{comparison_dir}{testfile}{modelcode}.png').astype('float32')/2**8
                            imgPred = cv2.cvtColor(imgPred, cv2.COLOR_BGR2RGB)
                            Lab_pred_whole = cv2.cvtColor(imgPred, cv2.COLOR_RGB2LAB)
                            maskedPred = get_mask_rgb(imgPred, sem_fwk)

                            masked_imgExp_lab = cv2.cvtColor(maskedExp, cv2.COLOR_RGB2LAB)
                            masked_imgPred_lab = cv2.cvtColor(maskedPred, cv2.COLOR_RGB2LAB)

                            error_whole = hyAB(Lab_expert_whole, Lab_pred_whole) / (w*h)
                            l_whole = L_error(Lab_expert_whole, Lab_pred_whole) / (w*h)
                            ab_whole = Chroma_error(Lab_expert_whole, Lab_pred_whole) / (w*h)

                            error = hyAB(masked_imgExp_lab, masked_imgPred_lab) / pixel_count_seg
                            l = L_error(masked_imgExp_lab, masked_imgPred_lab) / pixel_count_seg
                            ab = Chroma_error(masked_imgExp_lab, masked_imgPred_lab) / pixel_count_seg

                            print(f'Whole \n HyAB: {error_whole}, L*: {l_whole}, AB*: {ab_whole}')
                            print(f'Seg \n HyAB: {error}, L*: {l}, AB*: {ab}\n')
