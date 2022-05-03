"""
Custom Data Loader script used to read the pre-proessed dataset from the disk with the help/in the order of the Dataset Info file.
Then the data is converted to tensors and pass back to the calling function.
"""

import torch
import numpy as np
from torch.utils.data import Dataset
import os
import GNN_utils.utils as ut


class custom_DL(Dataset):
    def __init__(self, datasetinfo, root_dir):
        # 2 parameters are passed, _info is a csv file with the data information. root_dir is the dir that
        # has all tha data

        self.eidx = datasetinfo[:, 0]
        self.hist = datasetinfo[:, 1]
        self.lbl = datasetinfo[:, 2]
        self.medrgb_x = datasetinfo[:, 3]
        self.medlum_gt = datasetinfo[:, 4]
        self.medlum_x = datasetinfo[:, 5]
        self.rgb_gt = datasetinfo[:, 6]
        self.rgb_x = datasetinfo[:, 7]
        self.seg = datasetinfo[:, 8]
        self.stdrgb_x = datasetinfo[:, 9]

        self.root_dir = root_dir

    # load dataset...

    def __len__(self):
        return len(self.eidx)

    def __getitem__(self, idx):

        # -------------- X
        labels = np.load(os.path.join(self.root_dir, str(self.lbl[idx])))

        hot_mapped_labels = ut.codeonehot((labels - 1), num_classes=9)  # -1 to adjust for idx starting at 0
        medlumx = np.load(os.path.join(self.root_dir, str(self.medlum_x[idx])))
        medlumx = np.reshape(medlumx, (medlumx.shape[0], 1))
        medrgbx = np.load(os.path.join(self.root_dir, str(self.medrgb_x[idx])))
        stdrgbx = np.load(os.path.join(self.root_dir, str(self.stdrgb_x[idx])))

        feat_in_gnn = np.append(medrgbx, hot_mapped_labels, axis=1)
        feat_in_gnn = np.append(stdrgbx, feat_in_gnn, axis=1)
        feat_in_gnn = np.append(medlumx, feat_in_gnn, axis=1)

        # -------------- Edge indices
        edges = np.load(os.path.join(self.root_dir, str(self.eidx[idx])))
        edges = np.transpose(edges)
        for i in range(len(labels)):
            edges = np.where(edges == labels[i], i, edges)
        segv = np.load(os.path.join(self.root_dir, str(self.seg[idx]))).transpose()

        # -------------- Y
        feat_out_gnn = np.load(os.path.join(self.root_dir, str(self.medlum_gt[idx])))

        # -------------- RGB_input/output for Loss
        rgbx = np.load(os.path.join(self.root_dir, str(self.rgb_x[idx])))
        rgbgt = np.load(os.path.join(self.root_dir, str(self.rgb_gt[idx])))

        gnn_in = torch.tensor(feat_in_gnn, dtype=torch.float)
        gnn_gt = torch.tensor(feat_out_gnn, dtype=torch.float)
        edge_tensor = torch.tensor(edges, dtype=torch.long)

        return gnn_in, gnn_gt, edges, rgbx, rgbgt, segv


class ToTensor(object):
    # Convert ndarrays to Tensors
    def __call__(self, sample):

        gnn_in, gnn_gt, edge_tensor, rgbx, rgbgt, segv = sample[0], sample[1], sample[2], sample[3], sample[4], sample[5]
        return torch.from_numpy(gnn_in), torch.from_numpy(gnn_gt), torch.from_numpy(edge_tensor), \
               torch.from_numpy(rgbx), torch.from_numpy(rgbgt), torch.from_numpy(segv)
