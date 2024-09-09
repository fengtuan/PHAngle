import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module, init
from torch.nn.parameter import Parameter

def permute(mat, A_dim1, A_dim2, B_dim1, B_dim2):
    ans = torch.zeros((A_dim1 * A_dim2, B_dim1 * B_dim2),dtype=mat.dtype,device=mat.device)
    for i in range(A_dim1):
        for j in range(A_dim2):
            ans[A_dim2 * i + j, :] = mat[i * B_dim1 : (i + 1) * B_dim1,
                                         j * B_dim2 : (j + 1) * B_dim2].reshape((-1,))
    return ans

def permute_3D(mat, A_dim1, A_dim2, B_dim1, B_dim2):
    kernel_size=mat.shape[-1]
    ans = torch.zeros((kernel_size,A_dim1 * A_dim2, B_dim1 * B_dim2),dtype=mat.dtype,device=mat.device)
    for i in range(A_dim1):
        for j in range(A_dim2):
            ans[:,A_dim2 * i + j, :] = mat[i * B_dim1 : (i + 1) * B_dim1,
                                         j * B_dim2 : (j + 1) * B_dim2,:].reshape((-1,kernel_size)).transpose(0,1)
    return ans

def kron_decomp(mat, A_dim1, A_dim2, B_dim1, B_dim2, rank):    
    mat_= permute(mat, A_dim1, A_dim2, B_dim1, B_dim2)
    u, s, v  = torch.linalg.svd(mat_, full_matrices=False)
    A_hat = torch.zeros((rank, A_dim1, A_dim2),dtype=mat.dtype,device=mat.device)
    B_hat = torch.zeros((rank, B_dim1, B_dim2),dtype=mat.dtype,device=mat.device)
    for r in range(rank):
        A_hat[r, :, :] = torch.sqrt(s[r]) * u[:, r].reshape([A_dim1, A_dim2])
        B_hat[r, :, :] = torch.sqrt(s[r]) * v[r, :].reshape([B_dim1, B_dim2])
    return A_hat,B_hat

def kron_decomp_3D(mat, A_dim1, A_dim2, B_dim1, B_dim2, rank):
    kernel_size=mat.shape[-1]    
    mat_= permute_3D(mat, A_dim1, A_dim2, B_dim1, B_dim2)
    u, s, v  = torch.linalg.svd(mat_, full_matrices=False)
    A_hat = torch.zeros((rank, A_dim1, A_dim2,kernel_size),dtype=mat.dtype,device=mat.device)
    B_hat = torch.zeros((rank, B_dim1, B_dim2,kernel_size),dtype=mat.dtype,device=mat.device)
    for r in range(rank):
        A_hat[r, :, :,:] = torch.sqrt(s[:,r]) * u[:,:, r].transpose(0,1).reshape([A_dim1, A_dim2 ,kernel_size])
        B_hat[r, :, :,:] = torch.sqrt(s[:,r]) * v[:,r, :].transpose(0,1).reshape([B_dim1, B_dim2 ,kernel_size])
    return A_hat,B_hat

class PHMLinear(nn.Module):
  def __init__(self, r,in_features, out_features, bias=True,device=None, dtype=None):
    factory_kwargs = {'device': device, 'dtype': dtype}
    super(PHMLinear, self).__init__()
    d=math.sqrt(max(in_features, out_features))
    n = np.power(2,math.floor(np.log2(d)))
    self.in_features = in_features
    self.out_features = out_features
    self.t=self.in_features//n
    self.s=self.out_features//n
    self.n=n
    # self.rank=(in_features*out_features)//(r*(in_features+out_features))
    self.rank=r
    if bias:
        self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
    else:
        self.register_parameter('bias', None)   
    self.A = nn.Parameter(torch.empty((self.rank, n, self.t), **factory_kwargs))
    self.B = nn.Parameter(torch.empty((self.rank, self.s, n), **factory_kwargs))
    self.reset_parameters()


  def kronecker_product(self):
    H = torch.einsum("bij,bkm->ikjm",self.A, self.B).reshape(self.out_features, self.in_features)
    return H

  def forward(self, input):
    weight = self.kronecker_product()
    # weight = torch.sum(self.kronecker_product1(self.A, self.B), dim=0)
    return F.linear(input, weight=weight, bias=self.bias)

  def extra_repr(self) -> str:
    return 'in_features={}, out_features={}, bias={}'.format(
      self.in_features, self.out_features, self.bias is not None)
    
  def reset_parameters(self) -> None:
     bound = 1 / math.sqrt(self.in_features)
     weight = torch.empty((self.out_features, self.in_features))   
     init.uniform_(weight, a=-bound,b=bound)
     weight=weight.to(self.A.device)
     A_hat,B_hat=kron_decomp(weight, self.n, self.t, self.s, self.n, self.rank)
     self.A.data[:,:,:]=A_hat
     self.B.data[:,:,:]=B_hat 
     if self.bias is not None:
         init.uniform_(self.bias, -bound, bound)

class PHConv1D(Module):
  def __init__(self, r, in_features, out_features, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,device=None, dtype=None):
    factory_kwargs = {'device': device, 'dtype': dtype}
    super(PHConv1D, self).__init__()
    d=math.sqrt(max(in_features//groups, out_features))
    n = np.power(2,math.floor(np.log2(d)))
    self.in_features = in_features//groups
    self.out_features = out_features
    self.stride = stride
    self.padding = padding
    self.dilation=dilation
    self.groups=groups
    self.kernel_size = kernel_size
    self.n=n
    self.t=self.in_features//n
    self.s=self.out_features//n   
    self.rank=r
    if bias:
        self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
    else:
        self.register_parameter('bias', None)
    self.A = nn.Parameter(torch.empty((self.rank, n, self.t, kernel_size), **factory_kwargs))
    self.B = nn.Parameter(torch.empty((self.rank, self.s, n, kernel_size), **factory_kwargs)) 
    self.reset_parameters()

  def kronecker_product(self):
    H = torch.einsum("bijl,bkml->ikjml",self.A, self.B).reshape(self.out_features, self.in_features,self.kernel_size)
    return H

  def forward(self, input):
    weight =self.kronecker_product()
    return F.conv1d(input, weight, self.bias, self.stride, self.padding,self.dilation,self.groups)
    
  def extra_repr(self) -> str:
    return 'in_features={}, out_features={}, bias={}'.format(
        self.in_features, self.out_features, self.bias is not None)
       
  def reset_parameters(self) -> None:
     k=float(self.groups)/(self.in_features*self.kernel_size)
     bound=math.sqrt(k)
     weight = torch.empty((self.out_features, self.in_features,self.kernel_size))   
     init.uniform_(weight, a=-bound,b=bound)
     weight=weight.to(self.A.device)
     self.A.data,self.B.data=kron_decomp_3D(weight, self.n, self.t, self.s, self.n, self.rank)  
     if self.bias is not None:
         init.uniform_(self.bias, -bound, bound)  
        
  
