#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: weiyang
"""
import torch
from networks import PHAngle
from loss.loss import MAE_Loss,MSE_Loss_SinCos,MSE_Loss,MAE_Loss_SinCos
from  datasets import load_train_valid_test
from datasets import EmbeddingDataset,pad_collate
from torch.utils.data  import DataLoader
from torch.amp import GradScaler
import os
import argparse
import utils
import time
import numpy as np

def str2bool(v):
    """
    Converts string to bool type; enables command line 
    arguments in the format of '--arg1 true --arg2 false'
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
def str2list(string):
    ''' Method to convert underscore separated values into list in argparse '''
    return [int(item) for item in string.split("_")]
parser = argparse.ArgumentParser()
LookupChoices = type('', (argparse.Action, ), dict(__call__ = lambda a, p, n, v, o: setattr(n, a.dest, a.choices[v])))
parser.add_argument('--seed', default = 0, type = int)
# parser.add_argument('--pretrained_model',type=str,default="esm2_t48_15B_UR50D")
# parser.add_argument('--pretrained_model',type=str,default="esm2_t36_3B_UR50D")
# parser.add_argument('--pretrained_model',type=str,default="prot_t5_xl_uniref50")
# parser.add_argument('--pretrained_model',type=str,default="esm2_t30_150M_UR50D")
# parser.add_argument('--pretrained_model',type=str,default="esm2_t33_650M_UR50D")
#parser.add_argument('--pretrained_model',type=str,default="ankh-base")
parser.add_argument('--pretrained_model',type=str,default="ankh-large")
# parser.add_argument('--pretrained_model',type=str,default="esm1b_t33_650M_UR50S")
parser.add_argument('--depth',type=int,default=2) 
parser.add_argument('--dropout', default = 0.2, type = float)
parser.add_argument('--r', default =96, type = int)
parser.add_argument('--maxEpochs', default =1000, type = int)
parser.add_argument('--hidden_size', default = 1536, type = int)
parser.add_argument('--batch_size', default = 32, type = int)
parser.add_argument('--gradient_accumulation_steps',type=int,default=1)
parser.add_argument('--lr', default = 1e-3, type = float, help = 'Learning rate.')
parser.add_argument('--weight_decay', type=float, default=1e-5)
parser.add_argument('--res_dir',type=str,default="PHAngle_Results")
parser.add_argument('--loss_type',type=str,default="MSE")  
parser.add_argument('--network_type',type=str,default="PHAngle") 
parser.add_argument('--path',type=str,default="/embeddingData/")
parser.add_argument('--embedding_feature_path',type=str,default="")
parser.add_argument('--degree_transfer',type=str2bool, default=True)


args = parser.parse_args()
args.use_amp=False
torch.manual_seed(args.seed)
np.random.seed(args.seed) 
use_cuda =torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
if use_cuda:     
    torch.cuda.manual_seed_all(args.seed)
 
model_path=args.res_dir
if(os.path.isdir(model_path)==False ):
  os.mkdir(model_path)

args.path=args.path+args.pretrained_model+"/"
args.embedding_feature_path=args.path+"train/"
 
if args.pretrained_model[:3]=='esm':
    train_list,valid_list,Test2016_list,Test2018_list=load_train_valid_test(IsESM=True)
else:
    train_list,valid_list,Test2016_list,Test2018_list=load_train_valid_test(IsESM=False)
    
train_set=EmbeddingDataset(args.embedding_feature_path,train_list)
train_loader=DataLoader(train_set,shuffle=True,batch_size=args.batch_size,num_workers=0,collate_fn=pad_collate)
utils.load_embedding_feature(args,valid_list,"valid")
utils.load_embedding_feature(args,Test2016_list,"Test2016")
utils.load_embedding_feature(args,Test2018_list,"Test2018")
  
args.input_dim=valid_list[0].feats.shape[1]
args.hidden_size=args.input_dim

model=PHAngle(args).cuda()
if args.degree_transfer:
    if args.loss_type=="MSE":
       criterion= MSE_Loss_SinCos()
    else:
       criterion= MAE_Loss_SinCos()    
    FileName=model_path+'/'+args.pretrained_model+'_'+str(args.seed)+'_'+str(args.r)+'_'+args.loss_type+'_'+args.network_type+'_'+str(args.hidden_size)+'_sincos_'+time.strftime('Results_%Y-%m-%d-%H-%M-%S',time.localtime(time.time()))
    ModelName='%s/%s_%d_%d_%s_%s_%d_sincos.pth'%(model_path,args.pretrained_model,args.seed,args.r,args.loss_type,args.network_type,args.hidden_size)
else:
    if args.loss_type=="MSE":
       criterion= MSE_Loss()
    else:
       criterion= MAE_Loss() 
    FileName=model_path+'/'+args.pretrained_model+'_'+str(args.seed)+'_'+str(args.r)+'_'+args.loss_type+'_'+args.network_type+'_'+str(args.hidden_size)+'_'+time.strftime('Results_%Y-%m-%d-%H-%M-%S',time.localtime(time.time()))
    ModelName='%s/%s_%d_%d_%s_%s_%d.pth'%(model_path,args.pretrained_model,args.seed,args.r,args.loss_type,args.network_type,args.hidden_size)

param_dict={}
for k,v in model.named_parameters():
    param_dict[k]=v
bn_params=[v for n,v in param_dict.items() if ('ln' in n or 'bn' in n or 'bias' in n)]
rest_params=[v for n,v in param_dict.items() if not ('ln' in n or 'bn' in n or 'bias' in n)]
optimizer = torch.optim.AdamW([{'params':bn_params,'weight_decay':0},
                              {'params':rest_params,'weight_decay':args.weight_decay} ],
                              lr=args.lr,amsgrad=False)
scaler=GradScaler(enabled=args.use_amp)
#############################################################
f = open(FileName+'.txt', 'w')
# early-stopping parameters
decrease_patience=5
best_loss = 1e9
best_epoch = 0
epoch = 0
Num_Of_decrease=0
done_looping = False   
while (epoch < args.maxEpochs) and (not done_looping):
    epoch = epoch + 1
    start_time = time.time()
    average_loss=utils.embedding_feature_train_with_loader(args,model,scaler,device,train_loader,optimizer,criterion) 
    print("{}th Epoch took {:.1f}s".format(epoch, time.time() - start_time))
    f.write("{}th Epoch took {:.1f}s\n".format(epoch, time.time() - start_time))
    print("  Average train class loss:\t\t{:.4f}".format(average_loss))
    f.write("  Average train class loss:\t\t{:.4f}\n".format(average_loss))
    
    PHI_MSE,PSI_MSE=utils.embedding_feature_eval(args,model,device,valid_list)
    eval_loss=(PHI_MSE+PSI_MSE)/2.0
    print("  PHI_MSE:{:.2f},PSI_MSE:{:.2f}".format(PHI_MSE,PSI_MSE))
    f.write("  PHI_MSE:{:.2f},PSI_MSE:{:.2f}\n".format(PHI_MSE,PSI_MSE))    
    f.flush()
    if eval_loss < best_loss:
        best_loss = eval_loss
        best_epoch = epoch
        Num_Of_decrease=0        
        torch.save(model.state_dict(),ModelName)
    else:
        Num_Of_decrease=Num_Of_decrease+1
    if (Num_Of_decrease>decrease_patience):
        done_looping = True    
print("best evaluation loss:{:.2f},best epoch:{}".format(best_loss,best_epoch))
f.write("best evaluation loss:{:.2f},best epoch:{}\n".format(best_loss,best_epoch))
 
model.load_state_dict(torch.load(ModelName,weights_only=True))
PHI_MSE,PSI_MSE=utils.embedding_feature_eval(args,model,device,Test2016_list)
print("Test2016  PHI_MSE:{:.2f},PSI_MSE:{:.2f}".format(PHI_MSE,PSI_MSE))
f.write("Test2016  PHI_MSE:{:.2f},PSI_MSE:{:.2f}\n".format(PHI_MSE,PSI_MSE)) 
PHI_MSE,PSI_MSE=utils.embedding_feature_eval(args,model,device,Test2018_list)
print("Test2018  PHI_MSE:{:.2f},PSI_MSE:{:.2f}".format(PHI_MSE,PSI_MSE))
f.write("Test2018  PHI_MSE:{:.2f},PSI_MSE:{:.2f}\n".format(PHI_MSE,PSI_MSE)) 
f.close() 
