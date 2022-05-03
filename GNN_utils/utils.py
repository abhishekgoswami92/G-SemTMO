"""
Utils required by custom dataloader (customDL)
"""

import numpy as np
import torch


def custom_collate(batch):
    train_x = [item[0] for item in batch]
    train_y = [item[1] for item in batch]
    edges = [item[2] for item in batch]

    train_x = np.vstack(train_x).astype(float)
    train_y = np.vstack(train_y).astype(float)
    edges = np.hstack(edges).astype(int)

    # -----------------------------------Padding RGBxy
    len_arr = []
    for item in batch:
        for x in range(len(item[3])):
            len_arr.append(len(item[3][x]))
    max_len = np.max(len_arr)
    padded_rgbx = np.zeros([train_x.shape[0], max_len], dtype=float)
    c = 0
    for item in batch:
        for x in range(len(item[3])):
            padded_rgbx[c, :len_arr[c]] += item[3][x]
            c += 1

    padded_rgby = np.zeros([train_x.shape[0], max_len], dtype=float)
    c = 0
    for item in batch:
        for x in range(len(item[4])):
            padded_rgby[c, :len_arr[c]] += item[4][x]
            c += 1
    # --------------------------------------

    return torch.from_numpy(train_x), torch.from_numpy(train_y), torch.from_numpy(edges), \
           torch.from_numpy(padded_rgbx), torch.from_numpy(padded_rgby), torch.from_numpy(np.array(len_arr))


def codeonehot(vector, num_classes=None):
    """
    Converts an input 1-D vector of integers into an output
    2-D array of one-hot vectors, where an i'th input value
    of j will set a '1' in the i'th row, j'th column of the
    output array.

    Example:
        v = np.array((1, 0, 4))
        one_hot_v = convertToOneHot(v)
        print one_hot_v

        [[0 1 0 0 0]
         [1 0 0 0 0]
         [0 0 0 0 1]]
    """

    assert isinstance(vector, np.ndarray)
    assert len(vector) > 0

    if num_classes is None:
        num_classes = np.max(vector) + 1
    else:
        assert num_classes > 0
        assert num_classes >= np.max(vector)

    result = np.zeros(shape=(len(vector), num_classes))
    result[np.arange(len(vector)), vector] = 1
    return result.astype(int)
