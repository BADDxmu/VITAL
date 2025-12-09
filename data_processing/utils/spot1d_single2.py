#!/usr/bin/env python3
# Original file derived from: https://github.com/jas-preet/SPOT-1D-Single
# Copyright (c) Original SPOT-1D Authors
# Modifications Copyright (c) 2025, BADD-XMU
#
# This source code is licensed under the MIT License.
# The original license and copyright notice are preserved.


import torch
import numpy as np
from .dataset.dataset_inference import ProteinDataset, text_collate_fn
from .dataset.data_functions import pickle_load
from torch.utils.data import DataLoader
from .main import classification, regression, write_csv
import argparse
import os
from tqdm import tqdm
import pandas as pd



def spot1d_single(file_list, device, save_path):
    with open(file_list, 'r') as f:
        path_list= f.read().splitlines()
    # path_list = read_list(file_list)
    dataset = ProteinDataset(path_list)
    fm12_loader = DataLoader(dataset, batch_size=1, collate_fn=text_collate_fn, num_workers=4)
    CURRENT_DIR = os.path.dirname(__file__)
    means = pickle_load(os.path.join(CURRENT_DIR, "means_single.pkl"))
    stds = pickle_load(os.path.join(CURRENT_DIR, "stds_single.pkl"))
    means = torch.tensor(means, dtype=torch.float32)
    stds = torch.tensor(stds, dtype=torch.float32)

    if device == "cpu":
        model1_class = torch.jit.load(os.path.join(CURRENT_DIR, "jits/model1_class_cpu.pth"))
        model2_class = torch.jit.load(os.path.join(CURRENT_DIR, "jits/model2_class_cpu.pth"))
        model3_class = torch.jit.load(os.path.join(CURRENT_DIR, "jits/model3_class_cpu.pth"))

        model1_reg = torch.jit.load(os.path.join(CURRENT_DIR, "jits/model1_reg_cpu.pth"))
        model2_reg = torch.jit.load(os.path.join(CURRENT_DIR, "jits/model2_reg_cpu.pth"))
        model3_reg = torch.jit.load(os.path.join(CURRENT_DIR, "jits/model3_reg_cpu.pth"))

    elif device == "cuda:0":
        model1_class = torch.jit.load(os.path.join(CURRENT_DIR, "jits/model1_class_gpu.pth"))
        model2_class = torch.jit.load(os.path.join(CURRENT_DIR, "jits/model2_class_gpu.pth"))
        model3_class = torch.jit.load(os.path.join(CURRENT_DIR, "jits/model3_class_gpu.pth"))

        model1_reg = torch.jit.load(os.path.join(CURRENT_DIR, "jits/model1_reg_gpu.pth"))
        model2_reg = torch.jit.load(os.path.join(CURRENT_DIR, "jits/model2_reg_gpu.pth"))
        model3_reg = torch.jit.load(os.path.join(CURRENT_DIR, "jits/model3_reg_gpu.pth"))
        

    else:
        print("please check the arguments passed and refer to help associated with the arguments")

    class_out = classification(fm12_loader, model1_class, model2_class, model3_class, means, stds, device)
    names, seq, ss3_pred_list, ss8_pred_list, ss3_prob_list, ss8_prob_list = class_out

    reg_out = regression(fm12_loader, model1_reg, model2_reg, model3_reg, means, stds, device)
    psi_list, phi_list, theta_list, tau_list, hseu_list, hsed_list, cn_list, asa_list = reg_out
    print(len(ss3_pred_list), len(psi_list))
    write_csv(class_out, reg_out, save_path)

    ## conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
    ## conda install pandas


def binary_segment(path_load, path_seg, path_save, window):

    #处理SPOT1D的预测数据
    RES_MAX_ACC = {'A': 129.0, 'R': 274.0, 'N': 195.0, 'D': 193.0, 
                   'C': 167.0, 'Q': 225.0, 'E': 223.0, 'G': 104.0, 
                   'H': 224.0, 'I': 197.0, 'L': 201.0, 'K': 236.0, 
                   'M': 224.0, 'F': 240.0, 'P': 159.0, 'S': 155.0, 
                   'T': 172.0, 'W': 285.0, 'Y': 263.0, 'V': 174.0}

    list_seq = os.listdir(path_load)
    list_seq.sort()

    for pdbx in tqdm(list_seq):

        # 读取数据
        spot = pd.read_csv(f'{path_load}{pdbx}', keep_default_na=False).iloc[:,[1,2,4]]
        spot['rsa'] = spot['ASA'] / spot['AA'].map(RES_MAX_ACC)
        
        # 构建二分类特征
        H = ((spot['SS3'] == 'H')*1).to_list()
        E = ((spot['SS3'] == 'E')*1).to_list()
        C = ((spot['SS3'] == 'C')*1).to_list()
        b = ((spot['rsa'] < 0.1)*1).to_list()
        m = (((spot['rsa'] >= 0.1) & (spot['rsa'] < 0.4))*1).to_list()
        e = ((spot['rsa'] >= 0.4)*1).to_list()


        # 构建dataframe
        df_binary = pd.DataFrame({'H':H,'E':E,'C':C,'b':b,'m':m,'e':e})
        

        ### 进行切片操作 ###
        # 创建空dataframe
        column = ['seg']

        for i in range(1, window+1):
            for j in ['H','E','C']:
                column.append('ss_'+j+'_'+str(i))
        for i in range(1, window+1):
            for j in ['b','m','e']:
                column.append('rsa_'+j+'_'+str(i))

        df_binary_segment = pd.DataFrame(columns=column)

        # 切片且填入数据
        segment = pd.read_table(f'{path_seg}{pdbx[:-4]}.fasta', header=None, keep_default_na=False)
        for index in range(0, len(segment), 2):
            start, end = segment[0][index].split('_')[-2:]

            row = [pdbx+'_'+start+'_'+end]
            
            row.extend(df_binary[int(start)-1:int(end)].stack().to_list())

            df_binary_segment.loc[index] = row

        # 保存
        df_binary_segment.to_csv(f'{path_save}{pdbx[:-4]}.tsv', sep='\t', index=False)




# if __name__ == "__main__":

#     spot1d_single(file_list=args.file_list, device=args.device, save_path=args.save_path)