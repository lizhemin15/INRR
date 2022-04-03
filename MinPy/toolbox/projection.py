import numpy as np
from numpy import linalg as la
import os
import sys

current_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(current_dir)
sys.path.append("..")

class projection(object):
    def __init__(self,img=None):
        self.img = img
        self.get_bas()
        self.pro_real = self.projection(img)
    
    def get_bas(self):
        pass

    def projection(self,img):
        cor_list = []
        for bas in self.bas:
            cor_list.append(np.trace(img@bas.T)/np.trace(bas@bas.T))
        return cor_list

    
class svd_pro(projection):
    def __init__(self,img=None):
        self.img = img
        self.u,self.sigma,self.vt = la.svd(img)
        self.get_bas()
        self.pro_real = self.projection(img)
    def get_bas(self):
        bas_list = []
        k = self.img.shape[0]//10
        def get_recov_img(img,s,e):
            S = np.zeros(img.shape)
            for i in range(s,e):
                S[i,i] = self.sigma[i]
            tmp = np.dot(self.u,S)
            img_return = np.dot(tmp,self.vt)
            return img_return
        for s,e in [(0,k),(k,self.img.shape[0])]:
            bas = get_recov_img(self.img,s=s,e=e)
            bas_list.append(bas)
        self.bas = bas_list

class mask_pro(projection):
    def __init__(self,img=None,mask=None):
        self.img = img
        self.mask = mask
        self.pro_real = self.projection(img)
        
    def projection(self,img):
        obs_index = np.where(self.mask==1)
        unk_index = np.where(self.mask==0)
        obs_mse = np.mean(((img-self.img)[obs_index])**2)
        unk_mse = np.mean(((img-self.img)[unk_index])**2)
        #print(obs_mse,unk_mse)
        return [obs_mse,unk_mse]