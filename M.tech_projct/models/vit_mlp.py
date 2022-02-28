#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 12:53:53 2022

@author: sagar
"""

from vit_pytorch import ViT
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat



class ViT_regress(nn.Module):
    def __init__(self , seq_pool, avg_pool,class_token):
        super().__init__()
        self.model = ViT(
                  image_size=256,
                  patch_size=32,
                  num_classes=1000,
                  dim = 32,
                  depth = 4,
                  heads = 8,
                  mlp_dim = 64,
                  dropout = 0.1,
                  emb_dropout = 0.1
                )
        self.seq_pool = seq_pool
        self.avg_pool = avg_pool
        self.class_token = class_token
    
    # 32 dimension embedding is required as per paper.
        self.linear = nn.Linear(2048,32,bias=True)
        
        
        self.pos_embedding = nn.Parameter(torch.randn(1,193,32))
        self.cls_token = nn.Parameter(torch.randn(1, 1, 32))
    

        self.regress = nn.Linear(32,16,bias=True)
        self.linear2 = nn.Linear(16,1,bias=True)
        self.drop    = nn.Dropout(p=0.3)
        self.af1     = nn.ReLU()
        self.attention_pool = nn.Linear(32,1,bias=True)
    
    
    def token(self,x):
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        return x

        

    def forward(self,x):
        x  =  self.linear(x)
        if self.class_token and self.avg_pool==False and self.seq_pool==False:
            x = self.token(x)
        elif self.avg_pool and self.class_token==False:
            pass
        elif self.avg_pool and self.class_token==True:
            x = self.token(x)
        elif self.seq_pool and self.class_token==False:
            pass
        elif self.seq_pool and self.class_token==True:
            x = self.token(x)
            
        x  =  self.model.dropout(x)
        x  =  self.model.transformer(x)
        if self.seq_pool:
            x =   torch.matmul(F.softmax(self.attention_pool(x), dim=1).transpose(-1, -2), x).squeeze(-2)
        elif self.avg_pool:
            x =   torch.mean(x,1)
        else:
            x  =  x.mean(dim = 1) if self.model.pool == 'mean' else x[:, 0]
        
        
        x  =  self.model.to_latent(x)
        x  =  self.regress(x)
        x  =  self.drop(x)
        x  =  self.linear2(x)
        return x
    
        
