import os
import sys
current_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(current_dir)
sys.path.append("..")

from config import settings
import torch.nn as nn
import torch as t
import numpy as np
import loss
from third_party.models.siren_pytorch import SirenNet

cuda_if = settings.cuda_if
cuda_num = settings.cuda_num

class hc_reg(object):
    #使用torch写的正则化项
    #handcraft, hd
    def __init__(self,name='lap',kernel=None,p=2,model_path=None,sample_mode='random',sample_num=1000,L=None):
        if name == 'nn':
            self.model = t.load(model_path)
        if name == 'flap_row' or name == 'flap_col':
            self.L = L
        self.name = name
        self.__kernel = kernel
        self.__p = p
        self.type = 'hc_reg'
        self.sample_mode = sample_mode
        self.sample_num = sample_num

    def loss(self,M):
        self.__M = M
        if self.name == 'tv1':
            return self.tv(p=1)
        elif self.name == 'tv2':
            return self.tv(p=2)
        elif self.name == 'lap':
            return self.lap()
        elif self.name == 'kernel':
            return self.reg_kernel(kernel=self.__kernel,p=self.__p)
        elif self.name == 'de_row':
            return self.de('row')
        elif self.name == 'de_col':
            return self.de('col')
        elif self.name == 'l2':
            return self.lp(p=2)
        elif self.name == 'nn':
            return self.nn(sample_num=self.sample_num)
        elif self.name == 'flap_row':
            return self.flap(mode='row')
        elif self.name == 'flap_col':
            return self.flap(mode='col')
        else:
            raise('Please check out your regularization term')
    
    def lp(self,p=2):
        reg_loss = 0
        for name,w in self.__M.named_parameters():
            if name == 'layers.4.weight':
                reg_loss = reg_loss+t.norm(w,p=p)
        return reg_loss
        


    def tv(self,p):
        center = self.__M[1:self.__M.shape[0]-1,1:self.__M.shape[1]-1]
        up = self.__M[1:self.__M.shape[0]-1,0:self.__M.shape[1]-2]
        down = self.__M[1:self.__M.shape[0]-1,2:self.__M.shape[1]]
        left = self.__M[0:self.__M.shape[0]-2,1:self.__M.shape[1]-1]
        right = self.__M[2:self.__M.shape[0],1:self.__M.shape[1]-1]
        Var1 = 2*center-up-down
        Var2 = 2*center-left-right
        return (t.norm(Var1,p=p)+t.norm(Var2,p=p))/self.__M.shape[0]

            
    def lap(self):
        center = self.__M[1:self.__M.shape[0]-1,1:self.__M.shape[1]-1]
        up = self.__M[1:self.__M.shape[0]-1,0:self.__M.shape[1]-2]
        down = self.__M[1:self.__M.shape[0]-1,2:self.__M.shape[1]]
        left = self.__M[0:self.__M.shape[0]-2,1:self.__M.shape[1]-1]
        right = self.__M[2:self.__M.shape[0],1:self.__M.shape[1]-1]
        Var = 4*center-up-down-left-right
        return t.norm(Var,p=2)/self.__M.shape[0]
    
    def reg_kernel(self,kernel,p=2):
        center = self.__M[1:self.__M.shape[0]-1,1:self.__M.shape[1]-1]
        up = self.__M[1:self.__M.shape[0]-1,0:self.__M.shape[1]-2]
        down = self.__M[1:self.__M.shape[0]-1,2:self.__M.shape[1]]
        left = self.__M[0:self.__M.shape[0]-2,1:self.__M.shape[1]-1]
        right = self.__M[2:self.__M.shape[0],1:self.__M.shape[1]-1]
        lu = self.__M[0:self.__M.shape[0]-2,0:self.__M.shape[1]-2]
        ru = self.__M[2:self.__M.shape[0],0:self.__M.shape[1]-2]
        ld = self.__M[0:self.__M.shape[0]-2,1:self.__M.shape[1]-1]
        rd = self.__M[2:self.__M.shape[0],1:self.__M.shape[1]-1]
        Var = kernel[0][0]*lu+kernel[0][1]*up+kernel[0][2]*ru\
            +kernel[1][0]*left+kernel[1][1]*center+kernel[1][2]*right\
            +kernel[2][0]*ld+kernel[2][1]*down+kernel[2][2]*rd
        return t.norm(Var,p=p)/self.__M.shape[0]*8

    def de(self,mode='row'):
        if mode == 'col':
            M = self.__M.T
        else:
            M = self.__M
        Ones = t.ones(M.shape[1],1)
        Eyes = t.eye(M.shape[0])
        if cuda_if:
            Ones = Ones.cuda(cuda_num)
            Eyes = Eyes.cuda(cuda_num)
        V_M = t.sqrt(t.mm(M**2,Ones))
        cov = t.mm(M,M.T)/t.mm(V_M,V_M.T)
        lap = -cov+2*Eyes
        self.LAP = lap
        return t.trace(t.mm(M.T,t.mm(lap,M)))
    
    def nn(self,sample_num=1000):
        # 此时的self.__M为正在训练的网络结构
        # self.model 为加载的teacher结构
        # 在坐标范围内随机均匀采样 sample_num个点，计算两个网络在这些点上的MSE
        if self.sample_mode == 'random':
            input = t.rand(sample_num,2)*2-1
            if cuda_if:
                input = input.cuda(cuda_num)
            return loss.mse(self.model(input),self.__M(input))
        elif self.sample_mode == 'uniform':
            x = np.linspace(-1,1,sample_num)
            y = np.linspace(-1,1,sample_num)
            xx,yy = np.meshgrid(x,y)
            xyz = np.stack([xx,yy],axis=2).astype('float32')
            input = t.tensor(xyz).reshape(-1,2)
            if cuda_if:
                input = input.cuda(cuda_num)
            return loss.mse(self.model(input),self.__M(input))
        elif self.sample_mode == 'denoising':
            x = np.linspace(-1,1-2/sample_num,sample_num)
            y = np.linspace(-1,1-2/sample_num,sample_num)
            xx,yy = np.meshgrid(x,y)
            xyz = np.stack([xx,yy],axis=2).astype('float32')
            xyz += np.random.uniform(0,2/sample_num,(sample_num,sample_num,2))
            input = t.tensor(xyz).reshape(-1,2)
            if cuda_if:
                input = input.cuda(cuda_num)
            return loss.denoise_mse(input,self.__M)

    def flap(self,mode='row'):
        # L为加载的Laplacian矩阵
        if mode == 'col':
            M = self.__M.T
        else:
            M = self.__M
        return t.trace(t.mm(M.T,t.mm(self.L.to(M.device),M)))


class cair_reg(object):
    def __init__(self,r=256,mode='row'):
        self.type = 'cair_reg_'+mode
        self.mode = mode
        self.net = self.init_net(r)
        self.opt = self.init_opt()

    def init_net(self,r):
        unet = SirenNet(
            dim_in = 1,                        # input dimension, ex. 2d coor
            dim_hidden = 32,                  # hidden dimension
            dim_out = r,                       # output dimension, ex. rgb value
            num_layers = 5,                    # number of layers
            final_activation = nn.Softmax(),   # activation of final layer (nn.Identity() for direct output)
            w0_initial = 5.                   # different signals may require different omega_0 in the first layer - this is a hyperparameter
        ).cuda(cuda_num)
        if cuda_if:
            unet = unet.cuda(cuda_num)
        return unet

    def update(self,W):
        self.opt.step()
        self.data = self.init_data(W)

    def lap(self,A):
        n = A.shape[0]
        Ones = t.ones(n,1)
        I_n = t.from_numpy(np.eye(n)).to(t.float32)
        if cuda_if:
            Ones = Ones.cuda(cuda_num)
            I_n = I_n.cuda(cuda_num)
        A_1 = A * (t.mm(Ones,Ones.T)-I_n) # A_1 将中间的元素都归零，作为邻接矩阵
        L = -A_1+t.mm(A_1,t.mm(Ones,Ones.T))*I_n # A_2 将邻接矩阵转化为拉普拉斯矩阵
        return L

    def init_data(self,W):
        if self.mode == 'col':
            img = W
        else:
            img = W.T
        n = img.shape[1]
        coor = t.linspace(-1,1,n).reshape(-1,1)
        if cuda_if:
            coor = coor.cuda(cuda_num)
        self.A = self.net(coor)@self.net(coor).T
        self.L = self.lap(self.A)
        return t.trace(img@self.L@img.T)

    def init_opt(self):
        # Initial the optimizer of parameters in network
        optimizer = t.optim.Adam(self.net.parameters(),lr=1e-4)
        return optimizer



        

class auto_reg(object):
    def __init__(self,size,mode='row'):
        self.type = 'auto_reg_'+mode
        if mode == 'row':
            self.net = self.init_net(size,mode)
        else:
            self.net = self.init_net(size,mode)
        if cuda_if:
            self.net = self.net.cuda(cuda_num)
        self.opt = self.init_opt()

    def init_net(self,n,mode='row'):
        class net(nn.Module):
            def __init__(self,n,mode='row'):
                super(net,self).__init__()
                self.n = n
                self.A_0 = nn.Linear(n,n,bias=False)
                self.softmin = nn.Softmin(1)
                self.mode = mode

            def forward(self,W):
                Ones = t.ones(self.n,1)
                I_n = t.from_numpy(np.eye(self.n)).to(t.float32)
                if cuda_if:
                    Ones = Ones.cuda(cuda_num)
                    I_n = I_n.cuda(cuda_num)
                A_0 = self.A_0.weight # A_0 \in \mathbb{R}^{n \times n}
                A_1 = self.softmin(A_0) # A_1 中的元素的取值 \in (0,1) 和为1
                A_2 = (A_1+A_1.T)/2 # A_2 一定是对称的
                A_3 = A_2 * (t.mm(Ones,Ones.T)-I_n) # A_3 将中间的元素都归零，作为邻接矩阵
                A_4 = -A_3+t.mm(A_3,t.mm(Ones,Ones.T))*I_n # A_4 将邻接矩阵转化为拉普拉斯矩阵
                self.lap = A_4

                if self.mode == 'row':
                    return t.trace(t.mm(W.T,t.mm(A_4,W)))#+l1 #行关系
                elif self.mode == 'col':
                    return t.trace(t.mm(W,t.mm(A_4,W.T)))#+l1 #列关系
                elif self.mode == 'all':
                    return t.trace(t.mm(W.T,t.mm(self.A_0.weight,W)))#+l1 #所有L
        return net(n,mode)

    def update(self,W):
        self.opt.step()
        self.data = self.init_data(W)


    def init_data(self,W):
        return self.net(W)

    def init_opt(self):
        # Initial the optimizer of parameters in network
        optimizer = t.optim.Adam(self.net.parameters())
        return optimizer

class cnn_reg(object):
    def __init__(self):
        self.type = 'cnn_reg'
        self.net = self.init_net()
        if cuda_if:
            self.net = self.net.cuda(cuda_num)
        self.opt = self.init_opt()


    def init_net(self):
        class kernel(nn.Module):
            def __init__(self):
                super(kernel,self).__init__()
                self.K_0 = nn.Linear(3,3,bias=False)
                
                
            def forward(self):
                
                Ones = t.ones(3,1)
                I_n = t.from_numpy(np.eye(3)).to(t.float32)
                if cuda_if:
                    Ones = Ones.cuda(cuda_num)
                    I_n = I_n.cuda(cuda_num)
                K_1 = t.sigmoid(self.K_0.weight) # 使得K_1 中元素大于等于零小于等于一
                inner_12 = t.tensor([[1,1,1],[1,0,1],[1,1,1]]).to(t.float32)
                if cuda_if:
                    inner_12 = inner_12.cuda(cuda_num)
                K_2 = -inner_12 * K_1 # 挖空中间的元素，并取负
                inner_23 = t.tensor([[0,0,0],[0,1,0],[0,0,0]]).to(t.float32)
                if cuda_if:
                    inner_23 = inner_23.cuda(cuda_num)
                K_3 = K_2 - inner_23 * (t.mm(Ones.T,t.mm(K_2,Ones))) # 向中间填充进周围元素和相反数
                K_4 = K_3/t.norm(K_3,p='fro') # 将K_3归一化，防止收敛到0矩阵
                self.K_4 = K_4
                return K_4
            
        class net(nn.Module):
            def __init__(self):
                super(net,self).__init__()
                self.Kernel = kernel()
                
            def forward(self,M):
                return t.norm(nn.functional.conv2d(M.unsqueeze(dim=0).unsqueeze(dim=1),self.Kernel().unsqueeze(dim=0).unsqueeze(dim=1)),p=1)

        return net()


    def init_data(self,W):
        return self.net(W)

    def init_opt(self):
        # Initial the optimizer of parameters in network
        optimizer = t.optim.SGD(self.net.parameters(),lr=1e-1)
        return optimizer

    def update(self,W):
        self.opt.step()
        self.data = self.init_data(W)