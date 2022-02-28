#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 12:44:55 2022

@author: sagar
"""


import numpy as np
import torch
from scipy.io import loadmat
from torch.utils.data import Dataset, DataLoader


class LiveIQ(Dataset):
    def __init__(self,name,root_dir,scores):
        self.name=name
        self.root_dir=root_dir
        self.scores=scores
        
    def __len__(self):
        return len(self.name)
    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
#         print(self.name[idx],idx,MOS[idx])    
        string,format=self.name[idx].split('.')  
        
        file_name = self.root_dir+str(string)+".mat"
        image = loadmat(file_name)
        image = image['feature']
        image = np.transpose(image)
        image = np.squeeze(image)
#         print(image.shape)
        mos_score=self.scores[idx]
        return (image.astype(np.float32),mos_score.astype(np.float32))
