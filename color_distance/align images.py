"""
This file aligns images based on centre repositioning.
"""

import cv2
import rawpy
import tifffile as tif

namelist = ['a2372']

for name in namelist:
    dng = f'Data/Input/Raw/{name}.dng'  # RAW images
    exp = f'Data/Input/Groundtruth/{name}.jpg'  # Expert modified images
    exp_resz = f'Data/Input/Groundtruth/{name}.jpg'  # Resized expert images
    with rawpy.imread(dng) as raw:
        dng = raw.postprocess(half_size=False, use_camera_wb=True, no_auto_bright=True, output_bps=16, gamma=(1, 1))
        expim = cv2.imread(exp)
        expim_resz = cv2.imread(exp_resz)

        top_margin = (dng.shape[1] - expim.shape[1]) // 2
        bottom_margin = (dng.shape[1] - expim.shape[1]) - top_margin
        right_margin = (dng.shape[0] - expim.shape[0]) // 2
        left_margin = (dng.shape[0] - expim.shape[0]) - right_margin
        dng = dng[right_margin:-left_margin, top_margin:-bottom_margin]

        dim = (expim_resz.shape[1], expim_resz.shape[0])
        dng = cv2.resize(dng, dim, interpolation=cv2.INTER_AREA)

        tif.imsave(f'Data/Output/{name}.tiff', dng)
