"""
GCNCT uses DeepFCNCT_BNR, and GCNet. GNN predicts hint
(RGB,hint) -> (RGB) is predicted by FCNCT. Loss is sum of abs diff

"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import GNN_utils.customDL_GCNCT as dl
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from CT_utils.CT_library import rgb2lab
import GNN_utils.networks as net
from color_distance.Colour_distance import hyAB_torch as hyab


# Define Hyper-parameters

device = 'cuda'
model_code = 'G-SemTMO_ExpE'                                                # model code to identify the training
dataset_info = np.genfromtxt(open('Data/Preprocessed Dataset/Dataset_Info/dataset_info_4205.csv'),
                             delimiter=',', dtype=str)                      # csv file defining the dataset index info
dataset_root_dir = 'Data/Preprocessed Dataset/'         # root directory of dataset
writer = SummaryWriter(f'Log/{model_code}/')
model_path_gnn = f'Model/GCNCT/{model_code}_gnn.pth'                        # path for gcn model
model_path_fcn = f'Model/FCNCT/{model_code}_fcn.pth'                        # path for fcn model

input_size_GNN = 9 + 3 + 3 + 1   # label, features
output_size_GNN = 18
input_size_FCN = 3 + output_size_GNN + input_size_GNN
output_size_FCN = 3
batch_size = 1
learning_rate = 0.001
decay = 5e-4
min_loss = 999999


def customloss(pred, gt):
    pred_lab = rgb2lab(pred * (2 ** 8))
    gt_lab = rgb2lab(gt * (2 ** 8))
    loss = hyab(gt_lab, pred_lab)
    return loss / 10000


def broadcast_hint(segv, hints):
    segv = torch.squeeze(segv, 1)
    labels = torch.unique(segv)
    hintvec = torch.zeros([segv.shape[0], hints.shape[1]]).type(torch.float).to(device)
    for l in labels:
        hintvec[torch.where(segv == l)] = hints[torch.where(labels == l)]
    return hintvec


def gcnct_collate(batch):
    gnn_inp = [item[0] for item in batch]
    gnn_gt = [item[1] for item in batch]
    ei = [item[2] for item in batch]
    rgb_inp = [item[3] for item in batch]
    rgb_out = [item[4] for item in batch]
    segv = [item[5] for item in batch]

    gnn_inp = np.vstack(gnn_inp).astype('float32')
    gnn_gt = np.vstack(gnn_gt).astype('float32')
    ei = np.hstack(ei).astype('int64')
    rgb_inp = np.vstack(rgb_inp).astype('float32')
    rgb_out = np.vstack(rgb_out).astype('float32')
    segv = np.vstack(segv).transpose().astype(int)

    return torch.from_numpy(gnn_inp), torch.from_numpy(gnn_gt), torch.from_numpy(ei), torch.from_numpy(rgb_inp), \
           torch.from_numpy(rgb_out), torch.from_numpy(segv)


def train(t_loader, opt):
    modelGNN.train()
    modelFCNCT.train()
    sum_loss = 0

    for gnn_in, gnn_gt, edge_tensor, rgb_in, rgb_gt, segvec in t_loader:
        if len(edge_tensor) == 0:
            continue
        opt.zero_grad()
        gnn_in.requires_grad = True
        rgb_gt.requires_grad = True
        rgb_in.requires_grad = True

        gnn_in = gnn_in.to(device)
        edge_tensor = edge_tensor.to(device)
        rgb_in = rgb_in.to(device)
        rgb_gt = rgb_gt.to(device)
        hint = modelGNN(gnn_in, edge_tensor)
        skip_hint = torch.cat((hint, gnn_in), 1)
        hint_vector = broadcast_hint(segvec, skip_hint)
        rgbh = torch.cat((rgb_in, hint_vector), 1)
        rgb_out = modelFCNCT(rgbh)

        loss = torch.sum(torch.abs(rgb_out - rgb_gt))
        loss.backward()
        opt.step()
        sum_loss += loss
    return sum_loss / len(t_loader)


def val(v_loader):
    modelGNN.eval()
    modelFCNCT.eval()
    sum_loss = 0
    for gnn_in, gnn_gt, edge_tensor, rgb_in, rgb_gt, segvec in v_loader:
        if len(edge_tensor) == 0:
            continue
        gnn_in.requires_grad = False
        rgb_gt.requires_grad = False
        rgb_in.requires_grad = False

        gnn_in = gnn_in.to(device)
        edge_tensor = edge_tensor.to(device)
        rgb_in = rgb_in.to(device)
        rgb_gt = rgb_gt.to(device)
        hint = modelGNN(gnn_in, edge_tensor)
        skip_hint = torch.cat((hint, gnn_in), 1)
        hint_vector = broadcast_hint(segvec, skip_hint)
        rgbh = torch.cat((rgb_in, hint_vector), 1)
        rgb_out = modelFCNCT(rgbh)
        loss = torch.sum(torch.abs(rgb_out - rgb_gt))
        sum_loss += loss
    return sum_loss / len(v_loader)


if __name__ == "__main__":

    modelFCNCT = net.SimpleFCNCT(input_size_FCN, output_size_FCN).to(device)
    modelGNN = net.DeepGCNet_Dr(input_size_GNN, output_size_GNN).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(list(modelFCNCT.parameters()) + list(modelGNN.parameters()), lr=learning_rate,
                                weight_decay=decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[75, 150], gamma=0.1)

    data_train = dl.custom_DL(datasetinfo=dataset_info[np.r_[0:4000]], root_dir=dataset_root_dir)
    data_val = dl.custom_DL(datasetinfo=dataset_info[np.r_[4050:4100]], root_dir=dataset_root_dir)
    # ------------------------------------------------------------------------------------------   data collate and load
    data_train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True, collate_fn=gcnct_collate)
    data_val_loader = DataLoader(data_val, batch_size=batch_size, shuffle=True, collate_fn=gcnct_collate)

    # Train the model
    writer.add_graph(modelGNN, torch.Tensor([i for i in range(input_size_GNN)]).to(device))
    writer.add_graph(modelFCNCT, torch.Tensor([i for i in range(input_size_FCN)]).to(device))

    for epoch in range(0, 251):
        train_loss = train(data_train_loader, optimizer)

        if epoch % 5 == 0:
            val_loss = val(data_val_loader)
        log = 'Epoch: {:01d}, Train: {:.5f}, Val: {:.5f}, LR: {:.6f}'
        print(log.format(epoch, train_loss, val_loss, learning_rate))
        writer.add_scalars('T_Loss and V_Loss', {'TL': train_loss, 'VL': val_loss}, epoch)
        writer.add_scalars('LR', {'LR': scheduler.get_last_lr()[0]}, epoch)
        if val_loss < min_loss:
            torch.save(modelGNN.state_dict(), model_path_gnn)
            torch.save(modelFCNCT.state_dict(), model_path_fcn)
            min_loss = val_loss
        scheduler.step()

        for tag, param in modelFCNCT.named_parameters():
            writer.add_histogram(f"fcnct/{tag}/gradient", param.grad.data.cpu().numpy(), epoch)
            writer.add_histogram(f"fcnct/{tag}/value", param.data.cpu().numpy(), epoch)
        for tag, param in modelGNN.named_parameters():
            writer.add_histogram(f"gnn/{tag}/gradient", param.grad.data.cpu().numpy(), epoch)
            writer.add_histogram(f"gnn/{tag}/value", param.data.cpu().numpy(), epoch)

    writer.close()

    final_model_gnn = f'Model/GCNCT/{model_code}_finalgnn.pth'
    final_model_fcn = f'Model/GCNCT/{model_code}_finalfcn.pth'

    torch.save(modelGNN.state_dict(), final_model_gnn)
    torch.save(modelFCNCT.state_dict(), final_model_fcn)
