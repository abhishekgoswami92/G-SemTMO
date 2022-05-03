"""
Infer results trained using Ablation 1 or Ablation 2
"""

import torch
import numpy as np
import GNN_utils.networks as net
from matplotlib import pyplot as plt
import cv2
import GNN_utils.utils as ut
import tifffile as tif
from glob import glob
import os


# Define Hyper-parameters
device = 'cuda'
model_code = 'abl2'
dataset_root_dir = 'Data/Preprocessed Dataset/'

# ------------- namelist is the list of names of all images in the dataset.
namelist = list(np.load("Data/Preprocessed Dataset/Namelists/4205_names_FiveK", allow_pickle=True))
# --------------------------------------
# Needed for semantic specific tonecurves or reconstruction data
semantic_code_files = []
semantic_code = 'sky'
# ---------------------------------------
model_path_fcn = f'Model/Pretrained/Ablation/ablation2_FiveK.pth'
write_path = f'Data/Output/'
inputlin_im = 'Data/Input/Raw/'
seg_add = 'Data/Input/Segmap/'
sem_fwk_add = 'Data/Input/Semfwk/'
tc_add = f'Output/tc/'

input_size_GNN = 9 + 3 + 3 + 1          # label, features
output_size_GNN = 18
input_size_FCN = 3 + input_size_GNN
output_size_FCN = 3

modelFCNCT = net.SimpleFCNCT(input_size_FCN, output_size_FCN).to(device)
modelFCNCT.load_state_dict(torch.load(model_path_fcn))


def imshow(image, title):
    plt.imshow(image)
    plt.title(title)
    plt.show()


def broadcast_hint(segv, hints):                        # for infering without DL. tensors manually fix dim.
    segv = torch.unsqueeze(segv, -1)
    segv = torch.flatten(segv)
    labels = torch.unique(segv)
    hintvec = torch.zeros([segv.shape[0], hints.shape[1]]).type(torch.float).to(device)
    for l in labels:
        hintvec[torch.where(segv == l)] = hints[torch.where(labels == l)]
    return hintvec


legend = ['sky', 'mountain', 'veg', 'water', 'human', 'non-human', 'city', 'room', 'other']
colour = ['cyan', 'brown', 'green', 'blue', 'orange', 'red', 'grey', 'magenta', 'black']
semantic_pos = legend.index(semantic_code)


def infer_whole(rgb_in, hint, segvec):
    """
    Infer the whole image with rgb and semantic hint values
    """
    rgb_in = rgb_in.to(device)
    hint = hint.to(device)
    hint_vector = broadcast_hint(segvec, hint)
    rgbh = torch.cat((rgb_in, hint_vector), 1)
    rgb_out = modelFCNCT(rgbh)
    return rgb_out


def infer_rgb_per_hint(rgb_in, hint, numpixl):
    """
    Infer the rgb values only for specific hint values for each semantic node. Used for Tone curve generation
    """
    rgb_in = rgb_in.to(device)
    hint = hint.to(device)
    hint_vector = hint.repeat(numpixl, 1)
    rgbh = torch.cat((rgb_in, hint_vector), 1)
    rgb_out = modelFCNCT(rgbh)
    return rgb_out


def plot_tc(hint, axis, color, lgnd):

    dummyin = np.linspace(.000001, 1, 1000)
    dummy = torch.tensor(np.array([[i, i, i] for i in dummyin]), dtype=torch.float)
    dummy = dummy.to(device)
    hint = hint.to(device)
    hint_vector = hint.repeat(1000, 1)
    dummy = torch.cat((dummy, hint_vector), 1)

    dummyrgb = modelFCNCT(dummy)
    dummyrgb = dummyrgb.cpu().detach().numpy()
    dummyout = np.mean(dummyrgb, axis=1)
    dummyout = np.clip(dummyout, 0, 1)
    axis.plot(np.log10(dummyin + .000001), np.log10(dummyout + .000001), color=color, label=lgnd)
    axis.set_xlim([-3, 0])
    axis.set_ylim([-2, 0])

    return 0


meanloss = 0

for image_name in namelist[-81:]:
    file_name = inputlin_im + image_name + '.tiff'

    # -------------- RGB_input/output
    im = tif.imread(file_name).astype('float32') / 2**16

    image_name = os.path.basename(file_name).split('.')[0]

    w = int(im.shape[1] * 1)
    h = int(im.shape[0] * 1)
    im = cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA)
    rgbx = torch.tensor(np.reshape(im, [-1, 3]), dtype=torch.float)

    # -------------- Input features
    labels = np.load(f'{dataset_root_dir}{image_name}_labels.npy')
    if len(labels) == 1:
        continue
    hot_mapped_labels = ut.codeonehot((labels - 1), num_classes=9)  # -1 to adjust for idx starting at 0
    medx = np.load(f'{dataset_root_dir}{image_name}_medseg_x.npy')
    medx = np.reshape(medx, (medx.shape[0], 1))
    medrgbx = np.load(f'{dataset_root_dir}{image_name}_medrgb_x.npy')
    stdrgbx = np.load(f'{dataset_root_dir}{image_name}_stdrgb_x.npy')
    feat_in_gnn = torch.tensor(np.append(medrgbx, hot_mapped_labels, axis=1), dtype=torch.float)
    feat_in_gnn = torch.tensor(np.append(stdrgbx, feat_in_gnn, axis=1), dtype=torch.float)
    feat_in_gnn = torch.tensor(np.append(medx, feat_in_gnn, axis=1), dtype=torch.float)

    # -------------- Edge indices
    edges = np.load(f'{dataset_root_dir}{image_name}_eidx.npy')
    edges = np.transpose(edges)
    for i in range(len(labels)):
        edges = np.where(edges == labels[i], i, edges)
    edges = torch.from_numpy(edges.astype('int64'))

    # -------------- Normalising Semantic Framework
    sem_fwk = []
    RGB = np.zeros(im.shape)
    for file in glob(f'{sem_fwk_add}{image_name}/*.png'):
        matte = cv2.imread(file)[:, :, 0].astype(float) / 255.0
        matte = cv2.resize(matte, (w, h), interpolation=cv2.INTER_AREA)
        matte = cv2.bilateralFilter(matte.astype('float32'), d=60, sigmaColor=20, sigmaSpace=50)
        sem_fwk.append(matte)
    sem_fwk = sem_fwk / np.sum(sem_fwk, axis=0)


# ------------------------------------------------ ABLATION 2
    # ----------------- For tonecurves
    fig, ax = plt.subplots()
    ax.set(xlabel='Input log luminance', ylabel='Output luma',
           )
    rgb_fwk = []
    for i in range(feat_in_gnn.shape[0]):
        if labels[i] == semantic_pos + 1:
            semantic_code_files.append(image_name)
        plot_tc(feat_in_gnn[i], ax, color=colour[labels[i]-1])         # to save tonecurves
        rgbout_vec = infer_rgb_per_hint(rgbx, feat_in_gnn[i], w*h)
        rgbout_vec = rgbout_vec.cpu().detach().numpy()
        rgbout_vec = np.clip(rgbout_vec, 0, 1)
        pred_im = np.reshape(rgbout_vec, (h, w, 3))
        rgb_fwk.append(pred_im)

    ax.legend(loc='lower right', prop={'size': 16})
    ax.tick_params(axis='both', which='major', labelsize=15)
    plt.grid()
    plt.savefig(f'{tc_add}{image_name}_{model_code}_tc.png')
    plt.clf()

    # ----------------- Blending
    for i in range(len(sem_fwk)):
        matte = sem_fwk[i]
        rgb_hinted = rgb_fwk[i]
        matte = np.dstack([matte]*3)
        RGB = RGB + rgb_hinted * matte
    RGB = np.clip(RGB, 0, 1)
    plt.imsave(f'{write_path}{image_name}_{model_code}.png', RGB.astype('float32'))


# ------------------------------------------------ ABLATION 1
    # ----------------- For tonecurves

    rgb_fwk = []
    rgbx = rgbx.to(device)
    rgb_out = modelFCNCT(rgbx)
    rgbout_vec = rgb_out.cpu().detach().numpy()
    rgbout_vec = np.clip(rgbout_vec, 0, 1)
    RGB = np.reshape(rgbout_vec, (h, w, 3))

    plt.imsave(f'{write_path}{image_name}_MC_150_abl_1.png', RGB.astype('float32'))



























