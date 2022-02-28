#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 13:06:16 2022

@author: sagar
"""



import torch
import torch.nn as nn




class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x

    
class ConvMixer(nn.Module):
    def __init__(self,dim,depth,kernel_size,patch_size,n_classes):
        super().__init__()
        self.convmixer = nn.Sequential(
                    nn.Conv2d(2048, dim, kernel_size=patch_size, stride=patch_size),
                    nn.GELU(),
                    nn.Dropout(0.2),
                    nn.BatchNorm2d(dim),
                    *[nn.Sequential(
                            Residual(nn.Sequential(
                                nn.Conv2d(dim, dim, kernel_size, groups=dim, padding="same"),
                                nn.GELU(),
                                nn.Dropout(0.2),
                                nn.BatchNorm2d(dim)
                            )),
                            nn.Conv2d(dim, dim, kernel_size=1),
                            nn.GELU(),
                            nn.BatchNorm2d(dim)
                    ) for i in range(depth)],
                    nn.AdaptiveAvgPool2d((1,1)),
                    nn.Flatten(),
                    nn.Dropout(0.2),
                    nn.Linear(dim,n_classes)
            
              )
    
    def forward(self,x):
#         print(x.shape)
        x = torch.transpose(x,3,1)
        x = self.convmixer(x)
        return x