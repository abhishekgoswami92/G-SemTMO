"""
Create a dataset index in form of csv for the dataloader to consult while loading data fro training
"""

import numpy as np
import os

flist = os.listdir('Data/Preprocessed Dataset/')
dataset_count = 3               # number of samples (images) in the dataset
n_features = 10                 # number of features extracted for each sample

list_arr = np.empty((dataset_count, n_features), dtype=object)
for i in range(n_features):
    for j in range(dataset_count):
        list_arr[j, i] = flist[j * n_features + i]
np.savetxt('Data/Preprocessed Dataset/Dataset_Info/ds_info.csv', list_arr, delimiter=',', fmt='%s')
