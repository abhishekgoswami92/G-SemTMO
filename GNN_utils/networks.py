"""
This file details the various architectures for GNN and FC networks tried.
FCNCT - Fully Connected Network for Color Transfer
GCNet - Graph Convol. Network
BNR - Batch Normalised
Dr - Dropouts added
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import dropout_adj


class ShallowFCNCT(nn.Module):
    def __init__(self, input_size, num_classes):
        super(ShallowFCNCT, self).__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.lrelu1 = nn.LeakyReLU()
        self.fc2 = nn.Linear(32, 64)
        self.lrelu2 = nn.LeakyReLU()
        self.fc3 = nn.Linear(64, 32)
        self.lrelu3 = nn.LeakyReLU()
        self.fc4 = nn.Linear(32, num_classes)

    def forward(self, x):
        x = (torch.sign(x) * x) ** (1 / 2.2)
        out = self.fc1(x)
        out = self.lrelu1(out)
        out = self.fc2(out)
        out = self.lrelu2(out)
        out = self.fc3(out)
        out = self.lrelu3(out)
        out = self.fc4(out)

        return out


class SimpleFCNCT(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SimpleFCNCT, self).__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.lrelu1 = nn.LeakyReLU()
        self.fc2 = nn.Linear(32, num_classes)

    def forward(self, x):
        x = (torch.sign(x) * x) ** (1 / 2.2)
        out = self.fc1(x)
        out = self.lrelu1(out)
        out = self.fc2(out)

        return out


class DeepFCNCT_BNR(nn.Module):
    def __init__(self, input_size, num_classes):
        super(DeepFCNCT_BNR, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.lrelu1 = nn.LeakyReLU()
        self.fc2 = nn.Linear(64, 128)
        self.lrelu2 = nn.LeakyReLU()
        self.fc3 = nn.Linear(128, 128)
        self.lrelu3 = nn.LeakyReLU()
        self.fc4 = nn.Linear(128, 128)
        self.lrelu4 = nn.LeakyReLU()
        self.fc5 = nn.Linear(128, 64)
        self.lrelu5 = nn.LeakyReLU()
        self.fc6 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = (torch.sign(x) * x) ** (1 / 2.2)
        out = self.fc1(x)
        out = self.lrelu1(out)
        out = self.fc2(out)
        out = self.lrelu2(out)
        out = self.fc3(out)
        out = self.lrelu3(out)
        out = self.fc4(out)
        out = self.lrelu4(out)
        out = self.fc5(out)
        out = self.lrelu5(out)
        out = self.fc6(out)

        return out


class DeepFCNCT_BNR_Dr(nn.Module):
    def __init__(self, input_size, num_classes):
        super(DeepFCNCT_BNR_Dr, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.lrelu1 = nn.LeakyReLU()
        self.fc2 = nn.Linear(64, 128)
        self.lrelu2 = nn.LeakyReLU()
        self.d1 = nn.Dropout(.2)
        self.fc3 = nn.Linear(128, 128)
        self.lrelu3 = nn.LeakyReLU()
        self.fc4 = nn.Linear(128, 128)
        self.lrelu4 = nn.LeakyReLU()
        self.d2 = nn.Dropout(.2)
        self.fc5 = nn.Linear(128, 64)
        self.lrelu5 = nn.LeakyReLU()
        self.fc6 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = (torch.sign(x) * x) ** (1 / 2.2)
        out = self.fc1(x)
        out = self.lrelu1(out)
        out = self.fc2(out)
        out = self.lrelu2(out)
        out = self.d1(out)
        out = self.fc3(out)
        out = self.lrelu3(out)
        out = self.fc4(out)
        out = self.lrelu4(out)
        out = self.d2(out)
        out = self.fc5(out)
        out = self.lrelu5(out)
        out = self.fc6(out)

        return out


class ShallowGCNet(torch.nn.Module):
    def __init__(self, num_ft, num_class):
        super(ShallowGCNet, self).__init__()
        self.conv1 = GCNConv(num_ft, 256)
        self.conv2 = GCNConv(256, 256)
        self.conv3 = GCNConv(256, 128)
        self.conv4 = GCNConv(128, 64)
        self.post_mp = nn.Sequential(
            nn.Linear(64, num_class))

    def forward(self, x, edge_index):
        # ---------------------------------
        x = self.conv1(x, edge_index)
        x = F.leaky_relu(x)
        # ---------------------------------
        x = self.conv2(x, edge_index)
        x = F.leaky_relu(x)
        # ---------------------------------
        x = self.conv3(x, edge_index)
        x = F.leaky_relu(x)
        # ---------------------------------
        x = self.conv4(x, edge_index)
        x = F.leaky_relu(x)
        # ---------------------------------
        x = F.dropout(x, .5)
        x = self.post_mp(x)
        return x


class DeepGCNet(torch.nn.Module):
    def __init__(self, num_ft, num_class):
        super(DeepGCNet, self).__init__()
        self.conv1 = GCNConv(num_ft, 128)
        self.conv2 = GCNConv(128, 128)
        self.conv3 = GCNConv(128, 256)
        self.conv4 = GCNConv(256, 256)
        self.conv5 = GCNConv(256, 128)
        self.conv6 = GCNConv(128, 64)
        self.post_mp = nn.Sequential(
            nn.Linear(64, num_class))

    def forward(self, x, edge_index):
        # ---------------------------------
        x = self.conv1(x, edge_index)
        x = F.leaky_relu(x)
        # ---------------------------------
        x = self.conv2(x, edge_index)
        x = F.leaky_relu(x)
        # ---------------------------------
        x = self.conv3(x, edge_index)
        x = F.leaky_relu(x)
        # ---------------------------------
        x = self.conv4(x, edge_index)
        x = F.leaky_relu(x)
        # ---------------------------------
        x = self.conv5(x, edge_index)
        x = F.leaky_relu(x)
        # ---------------------------------
        x = self.conv6(x, edge_index)
        x = F.leaky_relu(x)
        # ---------------------------------
        x = F.dropout(x, .5)
        x = self.post_mp(x)
        return x


class SimpleGCNet(torch.nn.Module):
    def __init__(self, num_ft, num_class):
        super(SimpleGCNet, self).__init__()
        self.conv1 = GCNConv(num_ft, 32)
        self.conv2 = GCNConv(32, 64)
        self.conv3 = GCNConv(64, 32)
        self.post_mp = nn.Sequential(
            nn.Linear(32, num_class))

    def forward(self, x, edge_index):
        # ---------------------------------
        x = self.conv1(x, edge_index)
        x = F.leaky_relu(x)
        # ---------------------------------
        x = self.conv2(x, edge_index)
        x = F.leaky_relu(x)
        # ---------------------------------
        x = self.conv3(x, edge_index)
        x = F.leaky_relu(x)
        # ---------------------------------
        x = self.post_mp(x)
        return x


class DeepGCNet_Dr(torch.nn.Module):
    def __init__(self, num_ft, num_class):
        super(DeepGCNet_Dr, self).__init__()
        self.conv1 = GCNConv(num_ft, 128)
        self.conv2 = GCNConv(128, 128)
        self.conv3 = GCNConv(128, 256)
        self.conv4 = GCNConv(256, 256)
        self.conv5 = GCNConv(256, 128)
        self.conv6 = GCNConv(128, 64)
        self.post_mp = nn.Sequential(
            nn.Linear(64, num_class))

    def forward(self, x, edge_index):
        edge_index, _ = dropout_adj(edge_index, p=0.2,
                                    force_undirected=True,
                                    training=self.training)
        x = F.dropout(x, p=0.2, training=self.training)

        # ---------------------------------
        x = self.conv1(x, edge_index)
        x = F.leaky_relu(x)
        # ---------------------------------
        x = self.conv2(x, edge_index)
        x = F.leaky_relu(x)
        # ---------------------------------
        x = self.conv3(x, edge_index)
        x = F.leaky_relu(x)
        # ---------------------------------
        x = self.conv4(x, edge_index)
        x = F.leaky_relu(x)
        # ---------------------------------
        x = self.conv5(x, edge_index)
        x = F.leaky_relu(x)
        # ---------------------------------
        x = self.conv6(x, edge_index)
        x = F.leaky_relu(x)
        # ---------------------------------
        x = F.dropout(x, .5)
        x = self.post_mp(x)
        return x
