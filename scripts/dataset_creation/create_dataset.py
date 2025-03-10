
import os
from functools import wraps
import gc
import math
import random
from pathlib import Path      
from datetime import datetime
from typing import List,Tuple


# scientific
import numpy as np    
import matplotlib.pyplot as plt
import plotly.express as px 
from matplotlib import cm
import matplotlib          
import tifffile as tiff   
import pandas as pd        
from scipy.ndimage import zoom 
from sklearn.metrics import r2_score 
from sklearn.linear_model import LinearRegression

# torch
import torch   
from torch.utils.data import Dataset
from torch import nn
from torch import Tensor
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, random_split
from torch.optim import (SGD,
                         Adam, )

from torchvision.transforms import (ToTensor,   
                                    Compose,
                                    RandomHorizontalFlip,
                                    RandomVerticalFlip,
                                    RandomRotation,
                                    Normalize,
                                    ToPILImage)
import time
#%% Determine base path project and import data address paths:
base_dir = Path(r"D:\Iman_Nabipour\ToTuneCNNs2\Base-dir")# project root
base_output_dir = Path(r"D:\Iman_Nabipour\ToTuneCNNs2\Base-dir\cpt")
cu = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
output_dir = base_dir.joinpath("result").joinpath(cu)
output_dir.mkdir(parents=True, exist_ok=True)

# checkpoint_dir = "/content/drive/MyDrive/StoneRegression/20211215-163923" 
checkpoint_dir =None     
if checkpoint_dir is None:
    checkpoint_dir = base_output_dir.joinpath(cu)
    checkpoint_dir.mkdir(parents=True,exist_ok=True)
    has_cpt = False
else:
    checkpoint_dir = Path(checkpoint_dir)
    has_cpt = True

# new 3d images
## 1um_ High Resolution(HR) images paths:
### (Binary and Euclidian distanse transform(EDT) images)
### Original images
base_path = "D:/Iman_Nabipour/ToTuneCNNs2/"
path1 = "NewDataset_150/Dataset_Binary_150/"
path2 = "NewDataset_150/Dataset_NormalizedDistField_150/"
um1_bin = Path(base_path + path1 + "HR_Bin_150/HR_Bin_Original_150")
um1_dist = Path(base_path + path2 + "HR_NormDist_150/HR_NormDistField_Original_150")
### Dilated
um1_bin_dilated=Path(base_path + path1 + "HR_Bin_150/HR_Bin_Dilated_150")
um1_dist_dilated = Path(base_path + path2 + "HR_NormDist_150/HR_Dilated_NormDistField_150")
### Eroded
um1_bin_eroded = Path(base_path + path1 + "HR_Bin_150/HR_Bin_Eroded_150")
um1_dist_eroded = Path(base_path + path2 + "HR_NormDist_150/HR_Eroded_NormDistField_150")
### opening
um1_bin_opening = Path(base_path + path1 + "HR_Bin_150/HR_Bin_Opening_150")
um1_dist_opening = Path(base_path + path2 + "HR_NormDist_150/HR_Opening_NormDistField_150")

## 2um_ Midlle Resolution(MR) images paths:
### (Binary and Euclidian distanse transform(EDT) images)
### Original images
um2_bin = Path(base_path + path1 + "MR_Bin_150/MR_Bin_Original_150")
um2_dist = Path(base_path + path2 + "MR_NormDist_150/MR_NormDistField_Original_150")
### Dilated
um2_bin_dilated=Path(base_path + path1 + "MR_Bin_150/MR_Bin_Dilated_150")
um2_dist_dilated = Path(base_path + path2 + "MR_NormDist_150/MR_Dilated_NormDistField_150")
### Eroded
um2_bin_eroded = Path(base_path + path1 + "MR_Bin_150/MR_Bin_Eroded_150")
um2_dist_eroded = Path(base_path + path2 + "MR_NormDist_150/MR_Eroded_NormDistField_150")
### opening
um2_bin_opening = Path(base_path + path1 + "MR_Bin_150/MR_Bin_Opening_150")
um2_dist_opening = Path(base_path + path2 + "MR_NormDist_150/MR_Opening_NormDistField_150")

## 3um_ Low Resolution(LR) images paths:
### (Binary and Euclidian distanse transform(EDT) images)
### Original images
um3_bin = Path(base_path + path1 + "LR_Bin_150/LR_Bin_Original_150")
um3_dist = Path(base_path + path2 + "LR_NormDist_150/LR_NormDistField_Original_150")
### Dilated
um3_bin_dilated=Path(base_path + path1 + "LR_Bin_150/LR_Bin_Dilated_150")
um3_dist_dilated = Path(base_path + path2 + "LR_NormDist_150/LR_Dilated_NormDistField_150")
### Eroded
um3_bin_eroded = Path(base_path + path1 + "LR_Bin_150/LR_Bin_Eroded_150")
um3_dist_eroded = Path(base_path + path2 + "LR_NormDist_150/LR_Eroded_NormDistField_150")
### opening
um3_bin_opening = Path(base_path + path1 + "LR_Bin_150/LR_Bin_Opening_150")
um3_dist_opening = Path(base_path + path2 + "LR_NormDist_150/LR_Opening_NormDistField_150")

# Labels file path:
lb_file_2 = Path(base_path + "NewDataset_150/myLabels_150.xlsx")

## aggregation
um1_total_paths = (um1_bin,um1_dist,um1_bin_dilated,um1_dist_dilated,um1_bin_eroded,um1_dist_eroded,um1_bin_opening,um1_dist_opening)
um2_total_paths = (um2_bin,um2_dist,um2_bin_dilated,um2_dist_dilated,um2_bin_eroded,um2_dist_eroded,um2_bin_opening,um2_dist_opening)
um3_total_paths = (um3_bin,um3_dist,um3_bin_dilated,um3_dist_dilated,um3_bin_eroded,um3_dist_eroded,um3_bin_opening,um3_dist_opening)

## binary
um1_bin_paths = (um1_bin,um1_bin_dilated,um1_bin_eroded,um1_bin_opening)
um2_bin_paths = (um2_bin,um2_bin_dilated,um2_bin_eroded,um2_bin_opening)
um3_bin_paths = (um3_bin,um3_bin_dilated,um3_bin_eroded,um3_bin_opening)

## dist
# um1_dist_paths = (um1_dist,um1_dist_dilated,um1_dist_eroded,um1_dist_opening)
# um2_dist_paths = (um2_dist,um2_dist_dilated,um2_dist_eroded,um2_dist_opening)
# um3_dist_paths = (um3_dist,um3_dist_dilated,um3_dist_eroded,um3_dist_opening)
um1_dist_paths = (um1_dist,)
um2_dist_paths = (um2_dist,)
um3_dist_paths = (um3_dist,)
#%% Utility
def fix_all_seeds(seed):
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def flush_and_gc(f):
    @wraps(f)
    def g(*args, **kwargs):
        torch.cuda.empty_cache()
        gc.collect()
        return f(*args, **kwargs)

    return g
#%% Hyperparameters
EPOCHS = 60 #60
BATCH_SIZE = 1
MOMENTUM = 0.8
WEIGHT_DECAY = 5e-3
N_FEATURE_MAP = 32
LR = 1e-5 
#%% Data Augmentation
random_h_flip_prob = .5
random_v_flip_prob = .5
random_degree_rotate_prob = 10.
n_channel = 120
#%% Runtime
train_size_per = .9
dev_size_per = .1
seed = 2021
save_iter = 5
n_worker =0     
device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
print(device)
#%% Fix seed
fix_all_seeds(seed=seed)
#%% DataLoader
class Stone(Dataset):
    def __init__(self, images_dir: List[Path], label_xlx: Path, transformers):
        self._transformers = transformers
        self._ds_root = images_dir
        self._label_root = label_xlx
        self._lb = np.squeeze(pd.read_excel(str(self._label_root)).to_numpy(), axis=-1)
        lb = [self._lb for _ in range(len(images_dir))]
        self._lb = np.concatenate(lb,axis=0)
        self._f_list = []
        for d_path in self._ds_root:

            f_list = list(d_path.glob("*.tif"))
            f_list.sort(key=lambda p: int(p.stem.split("-")[0]))
            self._f_list+= f_list

        assert len(self._f_list) == len(self._lb), f" is {len(self._f_list)} == {len(self._lb)} ? "

    def __len__(self):
        return len(self._f_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        n_d_im = tiff.imread(str(self._f_list[idx]))

        n_d_im = zoom(n_d_im, (0.4, 0.4, 0.4))
#         print(n_d_im.shape)
        lb = torch.as_tensor(self._lb[idx]*10**15)
#         print(lb.shape)
        n_d_im = (n_d_im - n_d_im.min())/(n_d_im.max() - n_d_im.min())
        if self._transformers is not None:
            n_d_im = torch.unsqueeze(self._transformers(n_d_im.astype(np.float32)), 0)
            
        return n_d_im, lb
#%% Transforms
def get_transforms(p_hor=.5, p_ver=.5, r_degree=10, mean=.5, std=.5, n_channel=300):
    return Compose([
        ToTensor(),
        Normalize(mean=[mean] * n_channel, std=[std] * n_channel),
        RandomHorizontalFlip(p=p_hor),
        RandomVerticalFlip(p=p_ver),
#         RandomRotation(degrees=r_degree), 
    ])

def get_simple_transformers(n_channel=300):
      return Compose([
            ToTensor(),
            Normalize(mean=[0.5] * n_channel, std=[0.5] * n_channel)
      ])
#%% Show one sample

ts_ds = Stone(images_dir=um3_dist_paths,
                label_xlx=lb_file_2,
                transformers=None)
ts_data,ts_lb = ts_ds[np.random.randint(0,len(ts_ds))]
print("Permeability: ",ts_lb.item())
fig = px.imshow(ts_data, animation_frame=0, binary_string=True, labels=dict(animation_frame="scan"), height=600)
fig.show()
#%%
# save numpy array as csv file
from numpy import asarray
from numpy import savetxt
# load numpy array from csv file
from numpy import loadtxt
#
from sklearn.model_selection import train_test_split
#
#%%
data_all = np.array([])

len_ts_ds = len(ts_ds) #  150

print('start')

Control =3 # 3 for Um3, 2 Um2, 1 Um1 
#%%
if 1 == Control:
    data_all = np.array([])
    for i in range(len_ts_ds):
        X1,y1 = ts_ds[i]
        y1_ok = y1.numpy()
        len_X1 = len(X1) # 120
        print(i)
        for j in range(len_X1):
            img1 = X1[j,:,:]
            XX = np.reshape(img1, (1, 14400)) # image is : 120 * 120 = 14400
            XXy = np.append(y1_ok, XX)
            data_all =  np.append([data_all],[XXy]) # in for loop
    nn = len_ts_ds * len_X1 # 150 * 120 = 18000
    data_all = data_all.reshape(nn,14401) # 120 * 14401
    savetxt('yX_um1_dist150.csv',data_all,delimiter=',')

elif 2 == Control: #all is yX.csv um2_bin_paths
    data_all = np.array([])
    for i in range(len_ts_ds):
        X1,y1 = ts_ds[i]
        y1_ok = y1.numpy()
        len_X1 = len(X1) # 120
        print(i)
        for j in range(len_X1):
            img1 = X1[j,:,:]
            XX = np.reshape(img1, (1, 14400)) # image is : 120 * 120 = 14400
            XXy = np.append(y1_ok, XX)
            data_all =  np.append([data_all],[XXy]) # in for loop
    nn = len_ts_ds * len_X1 # 150 * 120 = 18000
    data_all = data_all.reshape(nn,14401) # 120 * 14401
    savetxt('yX_um2_dist150.csv',data_all,delimiter=',')

elif 3 == Control: #all is yX.csv um3_bin_paths
    data_all = np.array([])
    for i in range(len_ts_ds):
        X1,y1 = ts_ds[i]
        y1_ok = y1.numpy()
        len_X1 = len(X1) # 120
        print(i)
        for j in range(len_X1):
            img1 = X1[j,:,:]
            XX = np.reshape(img1, (1, 14400)) # image is : 120 * 120 = 14400
            XXy = np.append(y1_ok, XX)
            data_all =  np.append([data_all],[XXy]) # in for loop
    nn = len_ts_ds * len_X1 # 150 * 120 = 18000
    data_all = data_all.reshape(nn,14401) # 120 * 14401
    savetxt('yX_um3_dist150.csv',data_all,delimiter=',')
     
else:
    print(f'control is {Control}')

print('end')
