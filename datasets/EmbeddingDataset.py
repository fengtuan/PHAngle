#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: weiyang
"""

from torch.utils.data  import Dataset
import torch
import os

class EmbeddingDataset(Dataset):
  def __init__(self, root_dir, data_list):
        self.root_dir=root_dir
        self.data_list=data_list        
  def __len__(self):
       return len(self.data_list)
   
  def __getitem__(self, index):
      file_path=self.root_dir+self.data_list[index].PDB_ID+".pt"
      feats=torch.load(file_path ,weights_only=True)
      PHI=self.data_list[index].PHI
      PSI=self.data_list[index].PSI
      return feats,PHI,PSI
  def delete_feature_files(self):
      for i in range(len(self.data_list)):
          file_path=self.root_dir+self.data_list[i].PDB_ID+".pt"
          os.remove(file_path)

def pad_collate(batch):
    n=len(batch)
    (feats,PHIs,PSIs)=zip(*batch)
    maxLength=0
    for i in range(n):
        if feats[i].shape[0]>maxLength:
            maxLength=feats[i].shape[0]
    feature_dim=feats[0].shape[1]
    inputs=torch.zeros(size=(n,maxLength,feature_dim),dtype=torch.float32)
    masks=torch.zeros(size=(n,maxLength),dtype=torch.bool)
    targets=torch.zeros(size=(n,maxLength,2),dtype=torch.float32)
    targets.fill_(float('nan'))
    for i in range(n):
        length=feats[i].shape[0]
        inputs[i,:length,:]=feats[i]
        targets[i,:length,0]=PHIs[i]
        targets[i,:length,1]=PSIs[i]         
        masks[i,:length]=True
    return inputs,targets,masks
    
