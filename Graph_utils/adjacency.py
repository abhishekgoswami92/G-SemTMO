"""
This file creates the adjacency matrix for the semantic labels from the semantic masks of the image
"""

from matplotlib import pyplot as plt
import numpy as np
import cv2
import scipy


def imshow(file):
    plt.imshow(file)
    plt.show()


def union(lst1, lst2):
    if lst1 is None:
        lst1 = lst2
    final_list = list(set(lst1) | set(lst2))
    return final_list


def nb_vals(matrix, indices):
    matrix = scipy.array(matrix)
    indices = tuple(scipy.transpose(scipy.atleast_2d(indices)))
    arr_shape = scipy.shape(matrix)
    dist = scipy.ones(arr_shape)
    dist[indices] = 0
    dist = scipy.ndimage.distance_transform_cdt(dist, metric='chessboard')
    nb_indices = scipy.transpose(scipy.nonzero(dist == 1))
    return [matrix[tuple(ind)] for ind in nb_indices]


def ret_neigh(m, indices):
    i, j = indices
    if i == 0 and j == 0:
        return m[i, j + 1], m[i + 1, j]
    elif i == 0 and j == m.shape[1]-1:
        return m[i, j - 1], m[i + 1, j]
    elif i == m.shape[0]-1 and j == 0:
        return m[i-1, j], m[i, j+1]
    elif i == m.shape[0]-1 and j == m.shape[1]-1:
        return m[i-1, j], m[i, j-1]
    elif i == 0:
        return m[i, j-1], m[i, j+1], m[i+1, j]
    elif j == 0:
        return m[i-1, j], m[i, j+1], m[i+1, j]
    elif i == m.shape[0]-1:
        return m[i, j-1], m[i-1, j], m[i, j+1]
    elif j == m.shape[1]-1:
        return m[i-1, j], m[i, j-1], m[i+1, j]
    else:
        return m[i-1, j], m[i, j-1], m[i, j+1], m[i+1, j]


def get_adj(seg):
    seg_mask = seg[:, :, 0]
    labels = np.unique(seg_mask)
    adj_dict = {label: None for label in labels}
    for i in range(seg_mask.shape[0]):
        for j in range(seg_mask.shape[1]):
            adjacent_elements = []
            self_val = seg_mask[i, j]
            neigbours = ret_neigh(seg_mask, [i, j])
            for val in neigbours:
                if val != self_val:
                    adjacent_elements.append(val)
            values = adj_dict.get(self_val, "")
            adj_dict[self_val] = union(values, np.unique(adjacent_elements))
    return adj_dict


if __name__ == '__main__':
    seg_mask_add = 'Data/Input/Segmap/'               # Read a semantic mask image
    seg_im = cv2.imread(seg_mask_add + 'pecklake.png')
    get_adj(seg_im)





