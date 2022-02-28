#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 12:51:53 2022

@author: sagar
"""


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import torch
from torch import nn, einsum
import torch.nn.functional as F
from scipy.io import loadmat
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr


class FeatureToMOS(nn.Module):
    def __init__(self):
        super(FeatureToMOS, self).__init__()
        self.dense = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.BatchNorm1d(2048),
            nn.Linear(in_features=2048, out_features=512, bias=True),
            nn.BatchNorm1d(512),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=512, out_features=128, bias=True),
            nn.BatchNorm1d(128),
            nn.Dropout(p=0.3),
            nn.Linear(in_features=128, out_features=1, bias=True),

        )
             

    def forward(self, feat):
        return self.dense(feat)
       