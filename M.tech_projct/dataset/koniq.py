#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 12:44:57 2022

@author: sagar
"""

import torch

import numpy as np
from scipy.io import loadmat,savemat
from torch.utils.data import Dataset,DataLoader


class Koniq(Dataset):
    def __init__(self,name,root_dir,scores):
        self.name=name
        self.root_dir=root_dir
        self.scores=scores
        
    def __len__(self):
        return len(self.name)
    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        file_name = self.root_dir+str(self.name[idx])+".mat"
        image = loadmat(file_name)
        image = image['feature']
        mos_score=self.scores[idx]
        return (image.astype(np.float32),mos_score.astype(np.float32))