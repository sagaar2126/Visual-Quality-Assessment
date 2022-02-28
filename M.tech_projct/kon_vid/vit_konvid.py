#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 13:31:44 2022

@author: sagar
"""


import torch
import torch.nn as nn
import pandas as pd
from eval import eval_srocc 
from tqdm import tqdm
from dataset.konvid import Konvid
from models.vit_mlp import ViT_regress
from torch.utils import DataLoader



device = "cuda" if torch.cuda.is_available() else "cpu"



"""""
specify the path
"""""

path    = "/"
path3   = "/"




def train_loop(dataloader,model,optimizer,loss_fn,train_loss_list,fold):
#     print(type(train_loss_list))
    total_loss = 0
    model.train()
    size = len(dataloader.dataset)
#     print(size)
    for X , y in dataloader:
        X , y = X.to(device) , y.to(device)
        pred  = model(X)
        pred  = torch.squeeze(pred)
        loss  = loss_fn(pred,y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss+=loss.item()*X.shape[0]
#         print(total_loss)
#     print(total_loss/size)
    train_loss_list.append(total_loss/size)
    print(f"Total Average loss on training-set for fold {fold} is:{total_loss/size:>3f}") 


def test_loop(dataloader , model , loss_fn , test_loss_list,fold):
    total_loss = 0
    model.eval()
    size = len(dataloader.dataset)
    with torch.no_grad():
        for X , y in dataloader:
            X , y = X.to(device) , y.to(device)
            pred  = model(X)
            pred  = torch.squeeze(pred)
            loss  = loss_fn(pred,y)
            total_loss+=loss.item()*X.shape[0]
    test_loss_list.append(total_loss/size)
    print(f"Total Average loss on validation-set for fold {fold} is:{total_loss/size:>3f}") 


best_srocc = []
for i in tqdm(range(1,6)):
    train = pd.read_csv(path+"/"+"train_"+str(i)+".csv")
    test  = pd.read_csv(path+"/"+"test_"+str(i)+".csv")
    batch_size = 20
    x_train, x_test , y_train , y_test = train['names'],test['names'],train['mos'],test['mos']
    train_set    =  Konvid(x_train , path3+"/" , y_train)
    test_set     =  Konvid(x_test , path3+"/" , y_test)
    train_loader =  DataLoader(train_set , batch_size=batch_size , shuffle = True)
    test_loader  =  DataLoader(test_set , batch_size=batch_size , shuffle = False)
    train_srocc  =  []
    test_srocc   =  []
    train_error  =  []
    test_error   =  []
    epochs = 50
    
    
    seq_pool = True
    avg_pool = False
    class_token = True

    loss_fn = nn.MSELoss()
    
    model_1 = ViT_regress(seq_pool,avg_pool,class_token)
    model_2 = ViT_regress(seq_pool,avg_pool,class_token)
    model_3 = ViT_regress(seq_pool,avg_pool,class_token)
    model_4 = ViT_regress(seq_pool,avg_pool,class_token)
    model_5 = ViT_regress(seq_pool,avg_pool,class_token)
    model = [model_1,model_2,model_3,model_4,model_5]
    optimizer = torch.optim.Adam(model[i-1].parameters(), lr=0.001)
    print(f"Fold {i}\n---------------------------")
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_loader,model[i-1].to(device),optimizer,loss_fn, train_error,i)
        eval_srocc(train_loader,model[i-1].to(device),train_srocc,"train",i)
        test_loop(test_loader,model[i-1].to(device),loss_fn,test_error,i)
        eval_srocc(test_loader,model[i-1].to(device),test_srocc,"test",i)
    best_srocc.append(max(test_srocc)[0])

    
