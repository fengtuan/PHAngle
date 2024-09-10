 #!/usr/bin/env python3
#10 n -*- coding: utf-8 -*-
""" 
@author: weiyang
"""
from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.nn.functional as F
from .ph_layers import PHConv1D,PHMLinear

class BatchNorm1d(nn.Module):
    "Construct a BatchNorm1d module."
    def __init__(self, num_features, eps=1e-5,momentum=0.1):
        super(BatchNorm1d, self).__init__()
        self.weight = nn.Parameter(torch.ones(num_features,1))
        self.bias = nn.Parameter(torch.zeros(num_features,1))
        self.eps = eps
        self.momentum=momentum
        self.register_buffer('running_mean',torch.zeros(num_features,1))
        self.register_buffer('running_var',torch.ones(num_features,1))       
        self.C=num_features
   
    def forward(self, x,masks):
        if self.training:
            m_x=torch.masked_select(x.transpose(0,1),masks.transpose(0,1)).view(self.C,-1).contiguous()
            var,mean=torch.var_mean(m_x,dim=1,keepdim=True)            
            self.running_mean=(1-self.momentum)*self.running_mean+self.momentum*mean.detach()
            self.running_var=(1-self.momentum)*self.running_var+self.momentum*var.detach() 
        else:
            mean=self.running_mean
            var=self.running_var
        out=self.weight *( (x - mean) / (var+self.eps).sqrt()) + self.bias
        return torch.where(masks,out,torch.zeros(size=(1,),device=out.device)) 


def Conv1d(in_channels, out_channels, kernel_size, r=0):
    if r==0:
      conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=(kernel_size-1)//2, bias=False)
    else:
      conv = PHConv1D(r, in_channels, out_channels, kernel_size, padding=(kernel_size-1)//2, bias=False)
    return conv

def Linear(in_channels, out_channels, r=0):
    if r==0:
      linear = nn.Linear(in_channels, out_channels, bias=False)
    else:
      linear = PHMLinear(r, in_channels, out_channels, bias=False)
    return linear

class BasicConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, r=0):
        super(BasicConv1d, self).__init__()
        self.conv = Conv1d(in_channels, out_channels, kernel_size, r)      
        self.bn = BatchNorm1d(in_channels)
        self.kernel_size = kernel_size
        if in_channels != out_channels:
           self.identity=Conv1d(in_channels, out_channels,1, r)
        else:
           self.identity = nn.Sequential() 

    def forward(self,x,masks):
        x_ = self.identity(x)
        out = self.bn(x,masks)        
        out = self.conv(x)
        out = F.hardswish(out)               
        out = out+x_
        return out 
    
class build_block(nn.Module):
    def __init__(self, BasicConv,in_channels, out_channels,dropout=0.2,r=0):
        super(build_block, self).__init__()       
        self.conv =BasicConv(in_channels, out_channels, 1,r)
        self.branch_dropout = nn.Dropout(dropout)
        self.branch_conv =BasicConv(out_channels//2, out_channels//2, 3,r)        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x,masks=None):                
        out = self.conv(x,masks)
        branch1,branch2 = out.chunk(2, dim=1)
        branch2 = self.branch_conv(self.branch_dropout(branch2),masks)        
        out=torch.cat([branch1, branch2], 1)
        out=self.dropout(out)
        return out

class inception_unit(nn.Module):
    def __init__(self, BasicConv, in_channels,out_channels,dropout=0.2, r=0):
        super(inception_unit, self).__init__()       
        self.input_layers = nn.ModuleList(
            [build_block(BasicConv, in_channels,out_channels//2,dropout,r) for i in range(2)])
        self.intermediate_layer =build_block(BasicConv, out_channels//2,out_channels//2,dropout,r)
        

    def forward(self, x,masks=None):
        branch1 = self.input_layers[0](x,masks)
        branch2 = self.input_layers[1](x,masks)
        out = self.intermediate_layer(branch2,masks)
        branch2=out+branch2
        output = torch.cat([branch1, branch2], 1)
        return output


class PHAngle(nn.Module):
    def __init__(self,args):
        super(PHAngle, self).__init__()
        self.layer1 = inception_unit(BasicConv1d,args.input_dim,args.hidden_size,args.dropout, args.r)
        self.layer2 = nn.ModuleList(
                [inception_unit(BasicConv1d,args.hidden_size,args.hidden_size,args.dropout,args.r) for i in range(args.depth-1)])
        self.bn = BatchNorm1d(args.hidden_size)
        self.fc1 = Conv1d(args.hidden_size, 512, 1, args.r)
        if args.degree_transfer:
            self.fc2 =  nn.Conv1d(512, 4, 1, padding=0,bias=False)
        else:
            self.fc2 =  nn.Conv1d(512, 2, 1, padding=0,bias=False)
        self.depth=args.depth
        self.dropout=nn.Dropout(args.dropout)
    def forward(self, x,masks=None):
        out = x.transpose(1,2).contiguous()
        out = self.dropout(out)
        out=self.layer1(out,masks)
        for i in range(self.depth-1):
            out = self.layer2[i](out,masks)
        out=self.bn(out,masks)
        out=F.hardswish(self.fc1(out))
        out=self.fc2(out)
        out=out.transpose(1,2).contiguous()
        out = F.hardtanh(out,-6,6)/6.0
        return out
 
    
#for base feature
class PHAngle_base(nn.Module):
    def __init__(self,args):
        super(PHAngle_base, self).__init__()       
        self.proj = Linear(args.input_dim, args.hidden_size, 0)        
        self.layer = nn.ModuleList(
                [inception_unit(BasicConv1d,args.hidden_size,args.hidden_size,args.dropout,args.r) for i in range(args.depth)])        
        self.bn = BatchNorm1d(args.hidden_size)
        self.fc1 = Conv1d(args.hidden_size, 512, 1, args.r)             
        if args.degree_transfer:            
            self.fc2 =  nn.Conv1d(512, 4, 1, padding=0,bias=False)
        else:                       
            self.fc2 =  nn.Conv1d(512, 2, 1, padding=0,bias=False)
        self.depth=args.depth
        self.dropout=nn.Dropout(args.dropout)
    def forward(self, x,masks=None):        
        out = F.hardswish(self.proj(x))
        out = self.dropout(out)         
        out = out.transpose(1,2).contiguous()        
        for i in range(self.depth):
            out = self.layer[i](out,masks)         
        out=self.bn(out,masks)
        out=F.hardswish(self.fc1(out))              
        out=self.fc2(out)
        out=out.transpose(1,2).contiguous()        
        out = F.hardtanh(out,-6,6)/6.0       
        return out







    
    
