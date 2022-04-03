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

import loss,net

class basic_demo(object):
    def __init__(self,para=[6,6,6],reg=None):
        self.net = net.dmf(para)
        self.reg = reg
        self.loss_dict={'loss_fid':[],'loss_all':[],'nmae_test':[]}
        for reg_now in self.reg:
            self.loss_dict['loss_'+reg_now.type] = []
    
    def get_loss(self,fid_name,pic,mask_in,eta,mu):
        if fid_name == None:
            loss_fid = loss.mse(self.net.data,pic,mask_in)
        elif fid_name == 'inv':
            loss_fid = loss.mse_inv(self.net.data,pic,mask_in)
        elif fid_name == 'idl':
            loss_fid = loss.mse_id(self.net.data,pic,mask_in,direc='left')
        elif fid_name == 'idr':
            loss_fid = loss.mse_id(self.net.data,pic,mask_in,direc='right')
        else:
            raise('Wrong fid_name=',fid_name)
        loss_reg_list = []
        index_list = []
        j = 0
        for i,reg in enumerate(self.reg):
            if eta[i] != None:
                index_list.append(j)
                j+=1
                if reg.type == 'hc_reg':
                    loss_reg_list.append(reg.loss(self.net.data))
                    self.loss_dict['loss_'+reg.type].append(loss_reg_list[-1].detach().cpu().numpy())
                else:
                    loss_reg_list.append(reg.init_data(self.net.data))
                    self.loss_dict['loss_'+reg.type].append(loss_reg_list[-1].detach().cpu().numpy())
                    reg.opt.zero_grad()
            else:
                index_list.append(None)

        loss_all = mu*loss_fid
        for i in range(len(self.reg)):
            if eta[i] != None:
                loss_all = loss_all + eta[i]*loss_reg_list[index_list[i]]
        with t.no_grad():
            self.loss_dict['loss_fid'].append(loss_fid.detach().cpu().numpy())
            self.loss_dict['loss_all'].append(loss_all.detach().cpu().numpy())
            pic_know = pic*mask_in.cuda(cuda_num)
            if fid_name == 'inv':
                final_img = t.mm(t.mm(pic_know,self.net.data),pic_know)
            elif fid_name == 'idl':
                final_img = t.mm(pic_know,self.net.data)
            elif fid_name == 'idr':
                final_img = t.mm(self.net.data,pic_know)
            else:
                final_img = self.net.data
            self.loss_dict['nmae_test'].append(loss.nmae(final_img,pic,mask_in).detach().cpu().numpy())
        self.net.opt.zero_grad()
        return loss_all
    
    def train(self,pic,mu=1,eta=[0],mask_in=None,fid_name=None,train_reg_if=True):
        # loss_all = mu*loss_fid +  eta*loss_reg 
        # (Specially, when we choose mu=1, eta=0, We train the mdoel without regularizer)
        # If we set mu=0, this means we only train the regularizer term 
        loss_all = self.get_loss(fid_name,pic,mask_in,eta,mu)
        loss_all.backward()
        self.net.update()
        if train_reg_if:
            for reg in self.reg:
                if reg.type != 'hc_reg':
                    reg.update(self.net.data)

class basic_dmf(basic_demo):
    def __init__(self,para=[6,6,6],reg=None,std_w=1e-3):
        self.net = net.dmf(para,std_w)
        self.reg = reg
        self.loss_dict={'loss_fid':[],'loss_all':[],'nmae_test':[]}
        for reg_now in self.reg:
            self.loss_dict['loss_'+reg_now.type] = []

class air_net(basic_demo):
    def __init__(self,para=[6,6,6],reg=None,def_type=0,hadm_lr=1e-3,img=None,net_lr=1e-3):
        #self.net = net.dmf(para)
        img = img.unsqueeze(dim=0)
        img = t.repeat_interleave(img.unsqueeze(dim=1), repeats=para[0], dim=1)
        self.net = net.dip(para,img=img,lr=net_lr)
        self.reg = reg
        self.hadm = net.hadm([img.shape[2],img.shape[3]],def_type=def_type,hadm_lr=hadm_lr)
        self.loss_dict={'loss_fid':[],'loss_all':[],'nmae_test':[]}
        for reg_now in self.reg:
            self.loss_dict['loss_'+reg_now.type] = []

    def train(self,pic,mu=1,eta=[0],mask_in=None,fid_name=None,train_hadm=True):
        # loss_all = mu*loss_fid +  eta*loss_reg 
        # (Specially, when we choose mu=1, eta=0, We train the mdoel without regularizer)
        # If we set mu=0, this means we only train the regularizer term 
        if train_hadm:
            fid_term = self.net.data+self.hadm.data
        else:
            fid_term = self.net.data+self.hadm.data.detach()
        loss_all = self.get_loss(fid_name,pic,mask_in,eta,mu)
        if train_hadm:
            self.hadm.opt.zero_grad()
        loss_all.backward()
        self.net.update()
        if train_hadm:
            self.hadm.update()
        for reg in self.reg:
            if reg.type != 'hc_reg':
                reg.update(self.net.data)

class fp(basic_demo):
    def __init__(self,para=[2,100,100,1],reg=None,def_type=0,hadm_lr=1e-3,img=None,net_lr=1e-3,std_b=1e-3):
        #self.net = net.dmf(para)
        self.net = net.fp(para,img=img,lr=net_lr,std_b=std_b)
        self.reg = reg
        self.loss_dict={'loss_fid':[],'loss_all':[],'nmae_test':[]}
        for reg_now in self.reg:
            self.loss_dict['loss_'+reg_now.type] = []
        
                
class dip(basic_demo):
    def __init__(self,para=[6,6,6],img=None,reg=None,net_lr=1e-3,input_mode='random',mask_in=None,opt_type='Adam'):
        #self.net = net.dmf(para)
        self.net = net.dip(para=para,img=img,lr=net_lr,input_mode=input_mode,mask_in=mask_in,opt_type=opt_type)
        self.reg = reg
        self.loss_dict={'loss_fid':[],'loss_all':[],'nmae_test':[]}
        for reg_now in self.reg:
            self.loss_dict['loss_'+reg_now.type] = []
            
class fc(basic_demo):
    def __init__(self,para=[2,100,100,1],reg=None,def_type=0,hadm_lr=1e-3,img=None,net_lr=1e-3,std_b=1e-3):
        #self.net = net.dmf(para)
        self.net = net.fc(para,img=img,lr=net_lr,std_b=std_b)
        self.reg = reg
        self.loss_dict={'loss_fid':[],'loss_all':[],'nmae_test':[]}
        for reg_now in self.reg:
            self.loss_dict['loss_'+reg_now.type] = []