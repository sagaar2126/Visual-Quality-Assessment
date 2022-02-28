#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 13:22:31 2022

@author: sagar
"""


import torch
from scipy.stats import spearmanr


device = "cuda" if torch.cuda.is_available() else "cpu"
def eval_srocc(dataloader,model,srocc_list,string,fold):
    y_pred = []
    y_true = []
    model.eval()
    with torch.no_grad():
        for X,y in dataloader:
            X     = X.to(device)
            pred  = model(X)
            pred  = torch.squeeze(pred)
            Z     = pred.cpu().detach().numpy()
            Z     = list(Z)
            y     = y.cpu().detach().numpy()
            y     = list(y)

            for i,j in zip(Z,y):
                y_pred.append(i)
                y_true.append(j)

    
    srocc  = spearmanr(y_pred , y_true)
    if(string=="train"):
        print(f"SROCC on the training set for fold {fold} is:----------> {srocc[0]:>3f}")
    else:
        print(f"SROCC on the validation set for fold {fold} is:--------> {srocc[0]:>3f}")
    srocc_list.append(srocc)

