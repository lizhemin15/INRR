import torch as t
from torch import nn
import torch.nn.functional as F
import numpy as np
import os
import sys
from .mfn import GaborNet,FourierNet

current_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(current_dir)
sys.path.append("../..")

from config import settings
cuda_if = settings.cuda_if
cuda_num = settings.cuda_num

class MSNBase(nn.Module):
    """
    Multi Scale Neural Network
    This Network Combine the mfn with multi-scale fuzing
    To solve the problem of lack training data
    """
    def __init__(self,n_layers=3,scale_factor=2,params=[256,256],samp_mode='nearest',mainnet_name='fourier'):
        super().__init__()
        size_list = []
        size_list.append((params[0],params[1]))
        size_row,size_col = params[0],params[1]
        for _ in range(n_layers):
            size_row = size_row//scale_factor
            size_col = size_col//scale_factor
            size_list.append((size_row,size_col))
        size_list.reverse()
        if mainnet_name == 'fourier':
            self.mainnet = FourierNet(2,256,1)
        else:
            self.mainnet = GaborNet(2,256,1)
        self.dis = nn.ModuleList([Dis_Layer(self.mainnet,size) for size in size_list])
        self.up = Up_Layer(scale_factor=scale_factor, mode=samp_mode, align_corners=None)
        self.weight = t.autograd.Variable(t.rand(len(self.dis))*1e-3,requires_grad=True)

    def forward(self):
        y = 0
        for i,dis in enumerate(self.dis):
            w = self.weight[i]
            if i < len(self.dis)-1:
                y = self.up(y*(1-w)+dis()*w)
            else:
                y = y*(1-w)+dis()*w
        return y



class Dis_Layer(nn.Module):
    """
    Discrete from the continuou mfn neural network
    Input: Neural network
    Output: A discrete matrix size = size
    """
    def __init__(self,net,size):
        """
        All the input are scaled to [-0.5,0.5]
        """
        super().__init__()
        self.net = net
        self.size = size

    def img2cor(self,size):
        # 给定m*n灰度图像，返回mn*2
        self.m,self.n = size[0],size[1]
        x = np.linspace(0,1,self.n)-0.5
        y = np.linspace(0,1,self.m)-0.5
        xx,yy = np.meshgrid(x,y)
        self.xyz = np.stack([xx,yy],axis=2).astype('float32')
        if cuda_if:
            self.input = t.tensor(self.xyz).cuda(cuda_num).reshape(-1,2)
        else:
            self.input = t.tensor(self.xyz).reshape(-1,2)

    def cor2img(self,img):
        # 给定形状为mn*1的网络输出，返回m*n的灰度图像
        return img.reshape(self.m,self.n)

    def forward(self):
        self.img2cor(self.size)
        out = self.cor2img(self.net(self.input))
        return out
    



class Up_Layer(nn.Module):
    """
    Upsampling the specific mfn neural network
    Input: A discrete matrix
    Output: A Up or Down sampled matrix
    """
    def __init__(self,size=None, scale_factor=None, mode='nearest', align_corners=None):
        super().__init__()
        self.upper = nn.Upsample(size=size, scale_factor=scale_factor, mode=mode, align_corners=align_corners)
    
    def forward(self,x):
        x = t.unsqueeze(x,0)
        x = t.unsqueeze(x,0)
        x = self.upper(x)
        x = t.squeeze(x,0)
        return t.squeeze(x,0)


