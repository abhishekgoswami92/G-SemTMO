
# Graph-Semantic Tonemapping and LocHDR-HC200 Dataset 

Copyright Abhishek Goswami, Erwan Bernard
Copyright DxO Labs

Test images in the repository are taken from the [MIT-Adobe FiveK Dataset (Expert E)](https://data.csail.mit.edu/graphics/fivek/)

To get in touch email at goswamiabhishek92 (at) gmail (dot) com

This is a repository for the G-SemTMO implementation and the [LocHDR-HC200 dataset](<Data/LocHDR and HC200/LocHDR LR catalog.zip>) described in the thesis Chapter 5 :
>[Content-aware HDR tone mapping algorithms](https://www.theses.fr/s222174)
  Abhishek Goswami

## Install Dependencies

### Global dependencies

minimum requirement : python >= 3.7

Python modules can be install with
```
pip install -r requirements.txt
```

### pytorch

All the code expect pytorch to support GPU
Installation instruction to install pytorch with GPU support can be found [here](https://pytorch.org/get-started/locally/)

The code use Pytorch 1.7.0, should be install with :
```
pip3 install torch==1.7.0 -f https://download.pytorch.org/whl/torch/
pip3 install torchvision==0.8.1 -f https://download.pytorch.org/whl/torchvision/
pip3 install torchaudio==0.7.0 -f https://download.pytorch.org/whl/torchaudio/
```

### pytorch-geometric

Intallation instruction available [here](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)
```
pip install torch-scatter==2.0.5 torch-sparse==0.6.8 torch-spline-conv==1.2.0 torch-cluster==1.5.8 torch-geometric==1.6.3 -f https://data.pyg.org/whl/torch-1.7.0%2Bcu110.html
```

### Preprocesing dependencies

Raw Preprocessing is done using external library :
- [FastFCN](https://github.com/wuhuikai/FastFCN)
- [Alpha Matting](https://github.com/np-csu/AlphaMatting)


## 1 Pre-process the images in the dataset for training or inference

Pre-processing images involves representing the segmentating maps of each image as a connected graph and saving the node (semantic label)
specific features as numpy binary files as well as the graph in form of adjacency matrix in COO format.

- **IMPORTANT** Semantic segmentation can be obtained using [FastFCN](https://github.com/wuhuikai/FastFCN) classifier pretrained over ADE20k dataset.
- Run `Graph_utils/create_bin_seg.py` to get segmented maps with merged semantic labels. FastFCN labels are merged from 150 to 9 classes.
- Downsample all maps to 100x100px and store in `Data/Input/Segmap`.
- Full resolution segmented maps must be made pixel precise using [Alpha Matting](https://github.com/np-csu/AlphaMatting) technique to create binary mattes of each label (store in `Data/Input/Semfwk`).
- Run `Preprocess_Dataset_Gsemtmo.py` to convert image data to numpy data. Script provides support for .TIFF files. To use other RAW file formats python's RAWPY library is recommended.
- Creating Dataset Info -- While training the order and index of pre-processed dataset is required for the dataloader. Run `CreateDatasetInfo.py` to create
a csv file containing the dataset info.


## 2 Training

- `Gsemtmo_train.py` runs the training.

The script calls the customDL dataloader and uses the pre-processed data for the training. Hyperparameters for the training can be set to
choice.

## 3 Inference

- `Gsemtmo_infer.py` runs the inference.
- The script doesn't use a custom dataloader. Pre-processed data is loaded from disk and passed on to the network for inference. In case
a pretrained network is to be used for inference, use the Preprocess_dataset_Gsemtmo.py to create the binary files for the test set of
images.
- The inference script also saves for each image - a plot with tonecurves observed per semantic label. This part can be further tweaked to
obtain tone curves for individual segments only.

## 4 Quality of Inference

In the colour distance folder the `Colour_distance.py` script has three metrics HyAB, PSNR and MS-SSIM defined.
Given a 'Comparison' folder containing Ground truth and Inferred images, the script produces 3 csv files with 3 metric scores for each
image.

## Some other relevant Folders

*Ablation* --
The Ablation folder contains scripts for the two ablation studies -- their training and respective inference script. The previously used
script for preprocessing image data shall be used before training.

*Contrast analysis* --
The folder contains `contrast_scores.py` which computes a multi-scale contrast score for each image.

*Model* --
The folder has a subfolder 'pretrained' containing pretrained GCN and FC networks. The pretraining is conducted on the 5 experts of FiveK
dataset - A, B, C, D and E.
For the MIT experts, 4000 training images are used.
Furthermore, models trained on the LocHDR, locally curated dataset by our expert retoucher Ishani, is presented as Expert I. GCN and FCN
are trained over 680 images.
Finally, we present another set of models trained over the HC200 dataset, only the high contrast-punchy appearing images from LocHDR.
This training was conducted in a K-Fold cross validation format with 4 folds.

# Some important variables

namelist - List of the names/index of all the image files from the dataset. For training or inference.

codelist - List of codes added as suffix to the output image denoting the pretrained network. For learning based TMOs the codes represent
the model of training.

Datasetinfo - File index (in csv) of pre-processed dataset.

seg - semantic segmentation maps. The map is classified with 9 semantic classes.

sem_fwk - Semantic framework. Binary mattes of the seg maps.

tc - Tone curve plotting input vs output log luminance of images.

## Images and Enhancement Licenses

The repository uses images from third-party :
- Global tone mapping uses images from [Adobe FiveK](https://data.csail.mit.edu/graphics/fivek/),
  - License [LicenseAdobe.txt](https://data.csail.mit.edu/graphics/fivek/legal/LicenseAdobe.txt) covers files listed in [filesAdobe.txt](https://data.csail.mit.edu/graphics/fivek/legal/filesAdobe.txt)
  - License [LicenseAdobeMIT.txt](https://data.csail.mit.edu/graphics/fivek/legal/LicenseAdobeMIT.txt) covers files listed in [filesAdobeMIT.txt](https://data.csail.mit.edu/graphics/fivek/legal/filesAdobeMIT.txt)

- Local tone mapping uses images from [Adobe FiveK](https://data.csail.mit.edu/graphics/fivek/) with custom enhancement owned by DxO Labs under [Attribution-NonCommercial-ShareAlike 4.0 International
](LICENSE)


## Acknowledgments

This work is part of the doctoral research funded by the European Union’s Horizon 2020 research and innovation programme under the [Marie Sklodowska-Curie Grant Agreement No. 765911](https://www.realvision-itn.eu/).

We are grateful to DxO labs, University of Paris-Saclay|CNRS|CentraleSupelec and University of Cambridge for supporting this research work. We thank Mr. Wolf Hauser, Prof. Frederic Dufaux and Prof. Rafal Mantiuk
for their invaluable advice and guidance in the project.  
We would like to thank Jayawardhana GJKIU (Gunaseela Jayawardhana Kankanamge Ishani Uthpala Jayawardhana) for her incredible patience while retouching hundreds of RAW images using local editing tools for our [LocHDR dataset](<Data/LocHDR and HC200/LocHDR LR catalog.zip>).