#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 12:42:37 2022

@author: sagar
"""

import torch

import numpy as np
from scipy.io import loadmat,savemat
from torch.utils.data import Dataset,DataLoader


class VQC(Dataset):
    def __init__(self,name,root_dir,mos):
        self.name     = name
        self.root_dir = root_dir
        self.mos      = mos
    def __len__(self):
        return len(self.name)
    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        file_name = self.root_dir+str(self.name[idx].split(".")[0])+".npy"
        features = np.load(file_name)
        score   = self.mos[idx]
        return (np.asarray(features,dtype=np.float32),np.asarray(score,dtype=np.float32)) 


