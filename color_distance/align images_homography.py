"""
This file aligns and crops image based on homography of reference image
"""

import cv2 as cv
import numpy as np
import os
import rawpy
import tifffile as tif

namelist = list(np.load('O:/10-TESTS_ALGOS_B2C/SemanticTonemapping/Training GNN/Compile DS/selected codes.npy'))

for name in namelist[-119:]:
    dng_filename = f'Data/Input/Raw/{name}.dng'  # RAW images
    exp_filename = f'Data/Input/Groundtruth/{name}.jpg'  # Expert modified images
    exp_rsz_filename = f'Data/Input/Groundtruth/{name}.jpg'  # Resized expert images

    reduction_factor = 1 / 2.
    with rawpy.imread(dng_filename) as raw:
        fname = os.path.basename(dng_filename).split('.')[0]
        print(fname)
        dng_im = raw.postprocess(half_size=False, use_camera_wb=True, no_auto_bright=True, output_bps=8)
        dng_im = cv.resize(dng_im, dsize=None, fx=reduction_factor, fy=reduction_factor)

        exp_im = cv.imread(exp_filename)[:, :, ::-1]
        exp_shape = exp_im.shape
        exp_im = cv.resize(exp_im, dsize=None, fx=reduction_factor, fy=reduction_factor)
        print(exp_rsz_filename)
        exp_rsz_im = cv.imread(exp_rsz_filename)[:, :, ::-1]

        new_dng_im = raw.postprocess(half_size=False, use_camera_wb=True, no_auto_bright=True, output_bps=16,
                                     gamma=(1, 1))
        if dng_im.shape == exp_im.shape:
            new_dng_im_crop = cv.resize(new_dng_im, dsize=(exp_rsz_im.shape[1], exp_rsz_im.shape[0]))

        else:

            # Initiate SIFT detector
            sift = cv.SIFT_create()

            # find the keypoints and descriptors with SIFT
            kp1, des1 = sift.detectAndCompute(dng_im, None)
            kp2, des2 = sift.detectAndCompute(exp_im, None)

            # match points (Nearest Neighboors)
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)
            flann = cv.FlannBasedMatcher(index_params, search_params)
            matches = flann.knnMatch(des1, des2, k=2)

            # store all the good matches as per Lowe's ratio test.
            good_match = []
            for m, n in matches:
                if m.distance < 0.7 * n.distance:
                    good_match.append(m)

            do_reg_crop = False
            if len(good_match) < 4:
                do_reg_crop = True
            else:

                dng_pts = np.float32([kp1[m.queryIdx].pt for m in good_match]).reshape(-1, 1, 2)
                exp_pts = np.float32([kp2[m.trainIdx].pt for m in good_match]).reshape(-1, 1, 2)
                M, mask = cv.findHomography(exp_pts, dng_pts, cv.RANSAC, 5.0)

                # plot the projection
                h, w, d = exp_im.shape
                pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
                dst = cv.perspectiveTransform(pts, M)
                img2 = cv.polylines(dng_im, [np.int32(dst)], True, 255, 3, cv.LINE_AA)  # here !!!

                dst_m = np.int32(np.squeeze(dst) / reduction_factor)

                if np.max(dst_m[:, 0]) < dng_im.shape[0] and np.min(dst_m[:, 0]) > 0 and np.max(dst_m[:, 1]) < \
                        dng_im.shape[1] & np.min(dst_m[:, 1]) > 0:
                    new_dng_im_crop = new_dng_im[dst_m[0][1]:dst_m[2][1], dst_m[0][0]:dst_m[2][0]]
                else:
                    do_reg_crop = True
            if do_reg_crop:
                top_margin = int((new_dng_im.shape[1] - exp_shape[1]) // 2)
                bottom_margin = int((new_dng_im.shape[1] - exp_shape[1])) - top_margin
                right_margin = int((new_dng_im.shape[0] - exp_shape[0]) // 2)
                left_margin = int((new_dng_im.shape[0] - exp_shape[0])) - right_margin

                new_dng_im_crop = new_dng_im[right_margin:-left_margin, top_margin:-bottom_margin]

            h, w, _ = exp_rsz_im.shape
            new_dng_im_crop = cv.resize(new_dng_im_crop, dsize=(w, h))

        tif.imsave(f'Data/Output/{name}.tiff', new_dng_im_crop)
