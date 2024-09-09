#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 16:43:28 2020

@author: weiyang
"""
import numpy as np
import torch
from torch.amp import autocast
from networks import BatchNorm1d

def embedding_feature_iterate_minibatches(ProteinLists,batchsize,shuffle=False):
    num_features=ProteinLists[0].feats.shape[1]
    N=len(ProteinLists)
    indices = np.arange(N)
    if shuffle:        
        np.random.shuffle(indices)    
    maxLength=0
    inputs=torch.zeros(size=(batchsize,4096,num_features),dtype=torch.float32)
    masks=torch.zeros(size=(batchsize,4096),dtype=torch.bool)
    targets=torch.zeros(size=(batchsize,4096,2),dtype=torch.float32)
    for idx in range(N):
        if idx % batchsize==0:
            inputs.fill_(0)
            masks.fill_(False)
            targets.fill_(float('nan'))
            batch_idx=0          
            maxLength=0
        length=ProteinLists[indices[idx]].ProteinLen        
        inputs[batch_idx,:length,:]=ProteinLists[indices[idx]].feats[:,:]
        targets[batch_idx,:length,0]=ProteinLists[indices[idx]].PHI
        targets[batch_idx,:length,1]=ProteinLists[indices[idx]].PSI         
        masks[batch_idx,:length]=True
        batch_idx+=1
        if length>maxLength:
                maxLength=length
        if (idx+1) % batchsize==0:
            yield inputs[:,:maxLength,:].clone(),targets[:,:maxLength,:].clone(),masks[:,:maxLength].clone()
    if N % batchsize!=0:        
        yield inputs[:batch_idx,:maxLength,:].clone(),targets[:batch_idx,:maxLength,:].clone(),masks[:batch_idx,:maxLength].clone()

def esm_iterate_minibatches(ProteinLists, batchsize,shuffle=True):
    N= len(ProteinLists)
    last_size=N % batchsize
    indices = np.arange(N)
    if shuffle:        
        np.random.shuffle(indices)   
    maxLength=0
    masks=torch.zeros(size=(batchsize,4096),dtype=torch.long)
    targets=torch.zeros(size=(batchsize,4096,2),dtype=torch.float32)   
    PrimarySeqs=torch.zeros(size=(batchsize,4096),dtype=torch.long)
    for idx in range(N):
        if idx % batchsize==0:           
            masks.fill_(0)
            PrimarySeqs.fill_(1)
            targets.fill_(float('nan'))
            batch_idx=0          
            maxLength=0
        ProteinLen=ProteinLists[indices[idx]].ProteinLen       
        masks[batch_idx,:ProteinLen+2]=1      
        targets[batch_idx,1:ProteinLen+1,0]=ProteinLists[indices[idx]].PHI
        targets[batch_idx,1:ProteinLen+1,1]=ProteinLists[indices[idx]].PSI        
        PrimarySeqs[batch_idx,1:ProteinLen+1]=ProteinLists[indices[idx]].PrimarySeq
        PrimarySeqs[batch_idx,0]=0
        PrimarySeqs[batch_idx,ProteinLen+1]=2        
        batch_idx+=1
        if ProteinLen+2>maxLength:
                maxLength=ProteinLen+2
        if (idx+1) % batchsize==0:
            yield PrimarySeqs[:,:maxLength].clone(),targets[:,1:maxLength,:].clone(),masks[:,:maxLength].clone()
    if last_size !=0:        
        yield PrimarySeqs[:last_size,:maxLength].clone(),targets[:last_size,1:maxLength,:].clone(),masks[:last_size,:maxLength].clone()

def ProtTrans_iterate_minibatches(ProteinLists, batchsize,shuffle=True):
    N= len(ProteinLists)
    last_size=N % batchsize
    indices = np.arange(N)
    if shuffle:        
        np.random.shuffle(indices)   
    maxLength=0
    masks=torch.zeros(size=(batchsize,4096),dtype=torch.long)
    targets=torch.zeros(size=(batchsize,4096,2),dtype=torch.float32)   
    PrimarySeqs=torch.zeros(size=(batchsize,4096),dtype=torch.long)
    for idx in range(N):
        if idx % batchsize==0:           
            masks.fill_(0)
            PrimarySeqs.fill_(0)
            targets.fill_(float('nan'))
            batch_idx=0          
            maxLength=0
        ProteinLen=ProteinLists[indices[idx]].ProteinLen       
        masks[batch_idx,:ProteinLen+1]=1      
        targets[batch_idx,:ProteinLen,0]=ProteinLists[indices[idx]].PHI
        targets[batch_idx,:ProteinLen,1]=ProteinLists[indices[idx]].PSI        
        PrimarySeqs[batch_idx,:ProteinLen]=ProteinLists[indices[idx]].PrimarySeq        
        PrimarySeqs[batch_idx,ProteinLen]=1        
        batch_idx+=1
        if ProteinLen+1>maxLength:
                maxLength=ProteinLen+1
        if (idx+1) % batchsize==0:
            yield PrimarySeqs[:,:maxLength].clone(),targets[:,:maxLength,:].clone(),masks[:,:maxLength].clone()
    if last_size !=0:        
        yield PrimarySeqs[:last_size,:maxLength].clone(),targets[:last_size,:maxLength,:].clone(),masks[:last_size,:maxLength].clone()


def generate_embedding_feature(args,model,device,data_list,batch_size):
    model.eval()
    idx=0
    if args.pretrained_model[:3]=='esm':
        iterate_minibatches=esm_iterate_minibatches
    else:
        iterate_minibatches=ProtTrans_iterate_minibatches   
    with torch.no_grad():
        for batch in iterate_minibatches(data_list,batch_size, shuffle=False):
            inputs, targets,masks = batch
            n=inputs.shape[0]
            inputs=inputs.to(device)           
            masks=masks.to(device)
            with autocast(enabled=args.use_amp):
               outputs=model(inputs,masks)[0]
            outputs=outputs.cpu()
            print(idx)
            for i in range(n):
                if args.pretrained_model[:3]=='esm':
                    data_list[idx+i].feats=outputs[i,1:data_list[idx+i].ProteinLen+1,:].clone()
                else:
                    data_list[idx+i].feats=outputs[i,:data_list[idx+i].ProteinLen,:].clone()
            idx+=n 

def save_generating_embedding_feature(args,model,device,data_list,batch_size):
    model.eval()
    idx=0
    if args.pretrained_model[:3]=='esm':
        iterate_minibatches=esm_iterate_minibatches
    else:
        iterate_minibatches=ProtTrans_iterate_minibatches   
    with torch.no_grad():
        for batch in iterate_minibatches(data_list,batch_size, shuffle=False):
            inputs, targets,masks = batch
            n=inputs.shape[0]
            inputs=inputs.to(device)           
            masks=masks.to(device)
            with autocast(enabled=args.use_amp):
               outputs=model(inputs,masks)[0]
            outputs=outputs.cpu()
            print(idx)
            for i in range(n):
                if args.pretrained_model[:3]=='esm':
                    feats=outputs[i,1:data_list[idx+i].ProteinLen+1,:].clone()
                else:
                    feats=outputs[i,:data_list[idx+i].ProteinLen,:].clone()
                torch.save(feats,args.embedding_feature_path+data_list[idx+i].PDB_ID+".pt")
            idx+=n

def save_generating_embedding_feature_esm_1b(args,model,device,data_list,batch_size):
    model.eval()
    idx=0
    if args.pretrained_model[:3]=='esm':
        iterate_minibatches=esm_iterate_minibatches
    else:
        iterate_minibatches=ProtTrans_iterate_minibatches   
    with torch.no_grad():
        for batch in iterate_minibatches(data_list,batch_size, shuffle=False):
            inputs, targets,masks = batch
            n=inputs.shape[0]
            inputs=inputs.to(device)            
            with torch.no_grad():
                results = model(inputs, repr_layers=[33], return_contacts=False)
            outputs = results["representations"][33]
            print(idx)
            for i in range(n):
                if args.pretrained_model[:3]=='esm':
                    feats=outputs[i,1:data_list[idx+i].ProteinLen+1,:].clone()
                else:
                    feats=outputs[i,:data_list[idx+i].ProteinLen,:].clone()
                torch.save(feats,args.embedding_feature_path+data_list[idx+i].PDB_ID+".pt")
            idx+=n

def save_generating_embedding_feature_for_15B(args,model,device,data_list,batch_size,submodule_state_dicts):
    model.eval()
    idx=0
    if args.pretrained_model[:3]=='esm':
        iterate_minibatches=esm_iterate_minibatches
    else:
        iterate_minibatches=ProtTrans_iterate_minibatches   
    with torch.no_grad():
        for batch in iterate_minibatches(data_list,batch_size, shuffle=False):
            inputs, targets,masks = batch
            n=inputs.shape[0]
            inputs=inputs.to(device)           
            masks=masks.to(device)
            with autocast(enabled=args.use_amp):
               outputs=model(input_ids=inputs,attention_mask=masks,state_dicts=submodule_state_dicts)[0]               
            outputs=outputs.cpu()
            print(idx)
            for i in range(n):
                if args.pretrained_model[:3]=='esm':
                    feats=outputs[i,1:data_list[idx+i].ProteinLen+1,:].clone()
                else:
                    feats=outputs[i,:data_list[idx+i].ProteinLen,:].clone()
                torch.save(feats,args.embedding_feature_path+data_list[idx+i].PDB_ID+".pt")
            idx+=n

def load_embedding_feature(args,data_list,target_dir):
    infos=args.embedding_feature_path.split(sep="/")
    length=len(infos[-2])+1
    path=args.embedding_feature_path[:-length]+target_dir+"/"
    for i in range(len(data_list)):
        file_path=path+data_list[i].PDB_ID+".pt"        
        data_list[i].feats=torch.load(file_path,weights_only=True)
            
def embedding_feature_train(args,model,device,data_list,optimizer,criterion):
    model.train()
    total_loss=0.0
    count=0    
    for batch in embedding_feature_iterate_minibatches(data_list,args.batch_size,shuffle=True):
        inputs,targets,masks = batch
        inputs=inputs.to(device)
        targets=targets.to(device)      
        masks=masks.to(device)
        masks=masks.unsqueeze(dim=1)
        optimizer.zero_grad()
        with autocast(enabled=args.use_amp): 
           outputs=model(inputs,masks)
           loss = criterion(outputs, targets)
        total_loss+=loss.item()
        loss.backward()           
        optimizer.step()
        count+=1
    return total_loss/count

def embedding_feature_train_with_loader(args,model,scaler,device,train_loader,optimizer,criterion):
    model.train()
    total_loss=0.0
    count=0
    l=0
    length=len(train_loader)    
    for i,(inputs,targets,masks)  in enumerate(train_loader): 
        inputs=inputs.to(device)
        targets=targets.to(device)      
        masks=masks.to(device)
        masks=masks.unsqueeze(dim=1)        
        if l==0:
            optimizer.zero_grad()
        with autocast(enabled=args.use_amp, device_type="cuda"):                           
            outputs=model(inputs,masks)                
            loss = criterion(outputs, targets)/args.gradient_accumulation_steps
        total_loss+=loss.item()
        scaler.scale(loss).backward()         
        l+=1
        if l==args.gradient_accumulation_steps or i==length-1:                                          
            scaler.step(optimizer)
            scaler.update()    
            count+=1
            l=0  
    return total_loss/count

@torch.no_grad()
def update_bn(loader, model, device=None):
    r"""Updates BatchNorm running_mean, running_var buffers in the model.

    It performs one pass over data in `loader` to estimate the activation
    statistics for BatchNorm layers in the model.
    Args:
        loader (torch.utils.data.DataLoader): dataset loader to compute the
            activation statistics on. Each data batch should be either a
            tensor, or a list/tuple whose first element is a tensor
            containing data.
        model (torch.nn.Module): model for which we seek to update BatchNorm
            statistics.
        device (torch.device, optional): If set, data will be transferred to
            :attr:`device` before being passed into :attr:`model`.

    Example:
        >>> # xdoctest: +SKIP("Undefined variables")
        >>> loader, model = ...
        >>> torch.optim.swa_utils.update_bn(loader, model)

    .. note::
        The `update_bn` utility assumes that each data batch in :attr:`loader`
        is either a tensor or a list or tuple of tensors; in the latter case it
        is assumed that :meth:`model.forward()` should be called on the first
        element of the list or tuple corresponding to the data batch.
    """
    momenta = {}
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm) or isinstance(module, BatchNorm1d):
            module.running_mean = torch.zeros_like(module.running_mean)
            module.running_var = torch.ones_like(module.running_var)
            momenta[module] = module.momentum

    if not momenta:
        return

    was_training = model.training
    model.train()
    for module in momenta.keys():
        module.momentum = 0.1

    for input in loader:
        if isinstance(input, (list, tuple)):
            x = input[0]
            masks = input[2]
        if device is not None:
            x = x.to(device)
            masks=masks.to(device)
            masks=masks.unsqueeze(dim=1) 

        model(x,masks)

    for bn_module in momenta.keys():
        bn_module.momentum = momenta[bn_module]
    model.train(was_training)

def embedding_feature_eval(args,model,device,data_list):
    model.eval()
    MSE_loss_PHI=0.0
    Num_PHI=0
    MSE_loss_PSI=0.0
    Num_PSI=0
    with torch.no_grad():
        for batch in embedding_feature_iterate_minibatches(data_list,args.batch_size,shuffle=False):
            inputs,targets,masks = batch
            inputs=inputs.to(device)
            targets=targets.to(device)      
            masks=masks.to(device)
            masks=masks.unsqueeze(dim=1)
            with autocast(enabled=args.use_amp, device_type="cuda"):
               outputs=model(inputs,masks)
            if args.degree_transfer:
                pred_PHI=torch.rad2deg(torch.arctan2(outputs[:,:,0],outputs[:,:,1]))/180.0
                pred_PSI=torch.rad2deg(torch.arctan2(outputs[:,:,2],outputs[:,:,3]))/180.0
            else:
                pred_PHI=outputs[:,:,0]
                pred_PSI=outputs[:,:,1]                
            targets_PHI=targets[:,:,0].flatten()
            targets_PSI=targets[:,:,1].flatten()       
            masks_PHI=torch.isnan(targets_PHI).logical_not()
            masks_PSI=torch.isnan(targets_PSI).logical_not()
                
            valid_pred_PHI=torch.masked_select(pred_PHI.flatten(),masks_PHI)
            valid_targets_PHI=torch.masked_select(targets_PHI,masks_PHI)
            diff_PHI=torch.abs(valid_targets_PHI-valid_pred_PHI)
            diff_PHI=torch.where(diff_PHI>1.0, 2.0-diff_PHI,diff_PHI)
            MSE_loss_PHI+=diff_PHI.sum()
                
            valid_pred_PSI=torch.masked_select(pred_PSI.flatten(),masks_PSI)
            valid_targets_PSI=torch.masked_select(targets_PSI,masks_PSI)
            diff_PSI=torch.abs(valid_targets_PSI-valid_pred_PSI)
            diff_PSI=torch.where(diff_PSI>1.0, 2.0-diff_PSI,diff_PSI)
            MSE_loss_PSI+=diff_PSI.sum()                

            Num_PHI+=valid_targets_PHI.shape[0]
            Num_PSI+=valid_targets_PSI.shape[0]         
    return 180*MSE_loss_PHI/Num_PHI,180*MSE_loss_PSI/Num_PSI


def vmap_ensemble_embedding_feature_eval(args,models,device,data_list):
    from torch.func  import stack_module_state
    from torch.func  import functional_call
    import copy
    from torch import vmap
    params, buffers = stack_module_state(models)
    base_model = copy.deepcopy(models[0])
    base_model = base_model.to('meta')
    def fmodel(params, buffers, inputs,masks):
        return functional_call(base_model, (params, buffers), (inputs,masks))
    

    MSE_loss_PHI=0.0
    Num_PHI=0
    MSE_loss_PSI=0.0
    Num_PSI=0    
    with torch.no_grad():
        for batch in embedding_feature_iterate_minibatches(data_list,args.batch_size,shuffle=False):
            inputs,targets,masks = batch
            inputs=inputs.to(device)
            targets=targets.to(device)      
            masks=masks.to(device)
            masks=masks.unsqueeze(dim=1)
            with autocast(enabled=args.use_amp, device_type="cuda"):
                pred_vmap = vmap( fmodel, in_dims=(0,0,None,None),randomness='error')(params,buffers,inputs,masks)                
                outputs,_=  pred_vmap.median(dim=0) 
            if args.degree_transfer:
                pred_PHI=torch.rad2deg(torch.arctan2(outputs[:,:,0],outputs[:,:,1]))/180.0
                pred_PSI=torch.rad2deg(torch.arctan2(outputs[:,:,2],outputs[:,:,3]))/180.0
            else:
                pred_PHI=outputs[:,:,0]
                pred_PSI=outputs[:,:,1]                
            targets_PHI=targets[:,:,0].flatten()
            targets_PSI=targets[:,:,1].flatten()       
            masks_PHI=torch.isnan(targets_PHI).logical_not()
            masks_PSI=torch.isnan(targets_PSI).logical_not()
                
            valid_pred_PHI=torch.masked_select(pred_PHI.flatten(),masks_PHI)
            valid_targets_PHI=torch.masked_select(targets_PHI,masks_PHI)
            diff_PHI=torch.abs(valid_targets_PHI-valid_pred_PHI)
            diff_PHI=torch.where(diff_PHI>1.0, 2.0-diff_PHI,diff_PHI)
            MSE_loss_PHI+=diff_PHI.sum()
                
            valid_pred_PSI=torch.masked_select(pred_PSI.flatten(),masks_PSI)
            valid_targets_PSI=torch.masked_select(targets_PSI,masks_PSI)
            diff_PSI=torch.abs(valid_targets_PSI-valid_pred_PSI)
            diff_PSI=torch.where(diff_PSI>1.0, 2.0-diff_PSI,diff_PSI)
            MSE_loss_PSI+=diff_PSI.sum()                

            Num_PHI+=valid_targets_PHI.shape[0]
            Num_PSI+=valid_targets_PSI.shape[0]         
    return 180*MSE_loss_PHI/Num_PHI,180*MSE_loss_PSI/Num_PSI



def Ensemble_embedding_feature_eval(args,models,device,data_list):
    num_models=len(models)
    MSE_loss_PHI=0.0
    Num_PHI=0
    MSE_loss_PSI=0.0
    Num_PSI=0
    with torch.no_grad():
        for batch in embedding_feature_iterate_minibatches(data_list,args.batch_size,shuffle=False):
            inputs,targets,masks = batch
            inputs=inputs.to(device)
            targets=targets.to(device)      
            masks=masks.to(device)
            masks=masks.unsqueeze(dim=1)
            N=targets.shape[0]
            L=targets.shape[1]
            pred_PHIs=torch.zeros(size=(N,L,num_models),dtype=torch.float32,device=device)
            pred_PSIs=torch.zeros(size=(N,L,num_models),dtype=torch.float32,device=device)            
            for i in range(num_models):                
                model=models[i].to(device)
                model.eval()
                with autocast(enabled=args.use_amp, device_type="cuda"):
                   outputs=model(inputs,masks)
                if args.degree_transfer:
                    PHI=torch.rad2deg(torch.arctan2(outputs[:,:,0],outputs[:,:,1]))/180.0
                    PSI=torch.rad2deg(torch.arctan2(outputs[:,:,2],outputs[:,:,3]))/180.0
                else:
                    PHI=outputs[:,:,0]
                    PSI=outputs[:,:,1]
                pred_PHIs[:,:,i]=PHI
                pred_PSIs[:,:,i]=PSI
                # del model
                
            pred_PHI,_= pred_PHIs.median(dim=- 1)   
            pred_PSI,_= pred_PSIs.median(dim=- 1)
            
            targets_PHI=targets[:,:,0].flatten()
            targets_PSI=targets[:,:,1].flatten()       
            masks_PHI=torch.isnan(targets_PHI).logical_not()
            masks_PSI=torch.isnan(targets_PSI).logical_not()
                
            valid_pred_PHI=torch.masked_select(pred_PHI.flatten(),masks_PHI)
            valid_targets_PHI=torch.masked_select(targets_PHI,masks_PHI)
            diff_PHI=torch.abs(valid_targets_PHI-valid_pred_PHI)
            diff_PHI=torch.where(diff_PHI>1.0, 2.0-diff_PHI,diff_PHI)
            MSE_loss_PHI+=diff_PHI.sum()
                
            valid_pred_PSI=torch.masked_select(pred_PSI.flatten(),masks_PSI)
            valid_targets_PSI=torch.masked_select(targets_PSI,masks_PSI)
            diff_PSI=torch.abs(valid_targets_PSI-valid_pred_PSI)
            diff_PSI=torch.where(diff_PSI>1.0, 2.0-diff_PSI,diff_PSI)
            MSE_loss_PSI+=diff_PSI.sum()                

            Num_PHI+=valid_targets_PHI.shape[0]
            Num_PSI+=valid_targets_PSI.shape[0]         
    return 180*MSE_loss_PHI/Num_PHI,180*MSE_loss_PSI/Num_PSI




