import os
import sys
current_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(current_dir)
sys.path.append("..")

from config import settings
import torch.nn as nn
import torch as t

import numpy as np


cuda_if = settings.cuda_if
cuda_num = settings.cuda_num

def mse(pre,rel,mask=None):
    if mask == None:
        mask = t.ones(pre.shape)
    if cuda_if:
        mask = mask.cuda(cuda_num)
    return ((pre-rel)*mask).pow(2).mean()

def rmse(pre,rel,mask=None):
    if mask == None:
        mask = t.ones(pre.shape)
    if cuda_if:
        mask = mask.cuda(cuda_num)
    return t.sqrt(((pre-rel)*mask).pow(2).mean())

def psnr(pre,rel):
    MSE = mse(pre,rel)
    return 10*t.log10(1/MSE)

def ssim(pre,rel):
    pass

def nmae(pre,rel,mask=None):
    if mask == None:
        mask = t.ones(pre.shape)
    if cuda_if:
        mask = mask.cuda(cuda_num)
    def translate_mask(mask):
        u,v = t.where(mask == 1)
        return u,v
    u,v = translate_mask(1-mask)
    return t.abs(pre-rel)[u,v].mean()/(t.max(rel)-t.min(rel))

def mse_inv(pre,rel,mask=None):
    # loss = (rel-rel*pre*rel)\odot mask
    # rel \in R^{m\times n}, pre\in R{n\times m}
    if mask == None:
        mask = t.ones(pre.shape)
    if cuda_if:
        mask = mask.cuda(cuda_num)
    pre_now = t.mm(t.mm(rel,pre),rel)
    rel_now = rel
    return mse(pre_now,rel_now,mask)

def mse_id(pre,rel,mask=None,direc='left'):
    # if direc == 'left': pre\in R^{m\times m}, rel \in R^{m\times n} calculate pre*rel-rel
    # else pre\in R^{n\times n}, rel \in R^{m\times n} calculate rel*pre-rel
    if mask == None:
        mask = t.ones(pre.shape)
    if cuda_if:
        mask = mask.cuda(cuda_num)
    if direc == 'left':
        pre_now = t.mm(pre,rel)
    else:
        pre_now = t.mm(rel,pre)
    rel_now = rel
    return mse(pre_now,rel_now,mask)


def gen_loss(gen,dis,pic,mask):
    if mask == None:
        mask = t.ones(pic.shape)
    if cuda_if:
        mask = mask.cuda(cuda_num)
    m,n = mask.shape
    x = np.linspace(0,1,n)-0.5
    y = np.linspace(0,1,m)-0.5
    xx,yy = np.meshgrid(x,y)
    xyz = np.stack([xx,yy],axis=2).astype('float32')
    real_xy = t.tensor(xyz[mask==1]).reshape(-1,2)
    rand_xy = t.rand(m*n,2)-0.5
    if cuda_if:
        real_xy = real_xy.cuda(cuda_num)
        rand_xy = rand_xy.cuda(cuda_num)
    return -(1-dis(gen(real_xy)).mean())-dis(gen(rand_xy)).mean()



def dis_loss(gen,dis,pic,mask):
    if mask == None:
        mask = t.ones(pic.shape)
    if cuda_if:
        mask = mask.cuda(cuda_num)
    m,n = mask.shape
    rand_xy = t.rand(m*n,2)-0.5
    if cuda_if:
        rand_xy = rand_xy.cuda(cuda_num)
    return dis(gen(rand_xy)).mean()