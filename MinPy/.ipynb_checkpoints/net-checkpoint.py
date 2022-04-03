import os
import sys
current_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(current_dir)
sys.path.append("..")

from config import settings
import torch.nn as nn
import torch as t
from torch.autograd import Variable


from third_party.models import *
from third_party.utils.denoising_utils import *

cuda_if = settings.cuda_if
cuda_num = settings.cuda_num

class basic_net(object):
    # The basic network structure
    # Every network in MinPy at least include
    #     - self.init_para() and return a network module in pytorch
    #     - self.init_data() and return the output of neural network
    #     - self.init_opt() and put the network parameters into optimizer
    #     - self.update() and update the parameters in loss function
    def __init__(self):
        pass
    
    def init_para(self,params):
        pass

    def init_data(self):
        pass
    
    def init_opt(self,lr=1e-3,opt_type='Adam'):
        # Initial the optimizer of parameters in network
        if opt_type == 'Adadelta':
            optimizer = t.optim.Adadelta(self.net.parameters(),lr=lr)
        elif opt_type == 'Adagrad':
            optimizer = t.optim.Adagrad(self.net.parameters(),lr=lr)
        elif opt_type == 'Adam':
            optimizer = t.optim.Adam(self.net.parameters(),lr=lr)
        elif opt_type == 'AdamW':
            optimizer = t.optim.AdamW(self.net.parameters(),lr=lr)
        elif opt_type == 'SparseAdam':
            optimizer = t.optim.SparseAdam(self.net.parameters(),lr=lr)
        elif opt_type == 'Adamax':
            optimizer = t.optim.Adamax(self.net.parameters(),lr=lr)
        elif opt_type == 'ASGD':
            optimizer = t.optim.ASGD(self.net.parameters(),lr=lr)
        elif opt_type == 'LBFGS':
            optimizer = t.optim.LBFGS(self.net.parameters(),lr=lr)
        elif opt_type == 'SGD':
            optimizer = t.optim.SGD(self.net.parameters(),lr=lr)
        elif opt_type == 'NAdam':
            optimizer = t.optim.NAdam(self.net.parameters(),lr=lr)
        elif opt_type == 'RAdam':
            optimizer = t.optim.RAdam(self.net.parameters(),lr=lr)
        elif opt_type == 'RMSprop':
            optimizer = t.optim.RMSprop(self.net.parameters(),lr=lr)
        elif opt_type == 'Rprop':
            optimizer = t.optim.Rprop(self.net.parameters(),lr=lr)
        else:
            raise('Wrong optimization type')
        return optimizer

    def update(self):
        self.opt.step()
        self.data = self.init_data()
    

class dmf(basic_net):
    # Deep Matrix Factorization
    def __init__(self,params,std_w=1e-3):
        self.type = 'dmf'
        self.net = self.init_para(params,std_w)
        self.data = self.init_data()
        self.opt = self.init_opt()

    def init_para(self,params,std_w=1e-3):
        # Initial the parameter (Deep linear network)
        hidden_sizes = params
        layers = zip(hidden_sizes, hidden_sizes[1:])
        nn_list = []
        for (f_in,f_out) in layers:
            nn_list.append(nn.Linear(f_in, f_out, bias=False))
        model = nn.Sequential(*nn_list)
        if cuda_if:
            model = model.cuda(cuda_num)
        for m in model.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight,mean=0,std=std_w)
        return model

    def init_data(self):
        # Initial data
        def get_e2e(model):
            #获取预测矩阵
            weight = None
            for fc in model.children():
                assert isinstance(fc, nn.Linear) and fc.bias is None
                if weight is None:
                    weight = fc.weight.t()
                else:
                    weight = fc(weight)
            return weight
        return get_e2e(self.net)



class dmf_rand(basic_net):
    # Deep Matrix Factorization with random input
    def __init__(self,params):
        self.type = 'dmf_rand'
        self.net = self.init_para(params)
        self.input = t.eye(params[0],params[1])
        self.input = self.input.cuda(cuda_num)
        self.data = self.init_data()
        self.opt = self.init_opt()
        
    def init_para(self,params):
        # Initial the parameter (Deep linear network)
        hidden_sizes = params
        layers = zip(hidden_sizes, hidden_sizes[1:])
        nn_list = []
        for (f_in,f_out) in layers:
            nn_list.append(nn.Linear(f_in, f_out, bias=False))
        model = nn.Sequential(*nn_list)
        if cuda_if:
            model = model.cuda(cuda_num)
        for m in model.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight,mean=1e-3,std=1e-3)
        return model

    def init_data(self):
        # Initial data
        def get_e2e(model,input_data):
            #获取预测矩阵
            weight = input_data
            for fc in model.children():
                assert isinstance(fc, nn.Linear) and fc.bias is None
                if weight is None:
                    weight = fc.weight.t()
                else:
                    weight = fc(weight)
            return weight
        return get_e2e(self.net,self.input+t.randn(self.input.shape).cuda(cuda_num)*1e-2)
    
    def show_img(self):
        def get_e2e(model,input_data):
            #获取预测矩阵
            weight = input_data
            for fc in model.children():
                assert isinstance(fc, nn.Linear) and fc.bias is None
                if weight is None:
                    weight = fc.weight.t()
                else:
                    weight = fc(weight)
            return weight
        return get_e2e(self.net,self.input)


class hadm(basic_net):
    # Hadmard Product
    def __init__(self,params,def_type=0,hadm_lr=1e-3):
        self.type = 'hadm'
        self.def_type = def_type
        self.net = self.init_para((params[0],params[-1]))
        self.data = self.init_data()
        self.opt = self.init_opt(hadm_lr=hadm_lr)

    def init_para(self,params):
        # Initial the parameter (Deep linear network)
        g = t.randn(params)*1e-4
        h = t.randn(params)*1e-4
        if cuda_if:
            g = g.cuda(cuda_num)
            h = h.cuda(cuda_num)
        g = Variable(g,requires_grad=True)
        h = Variable(h,requires_grad=True)
        return [g,h]

    def init_data(self):
        # Initial data
        if self.def_type == 0:
            return self.net[0]*self.net[1]
        else:
            return self.net[0]*self.net[0]-self.net[1]*self.net[1]


class dip(basic_net):
    # unet like neural network, which have DIP
    def __init__(self,para,img,lr=1e-3,input_mode='random',mask_in=None,opt_type='Adam'):
        self.type = 'dip'
        self.net = self.init_para(para)
        self.img = img
        if input_mode == 'random':
            self.input = t.rand(img.shape)*1e-1
        elif input_mode == 'masked':
            self.input = img*mask_in
        elif input_mode in ['knn','nnm','softimpute','simple','itesvd','mc','ii']:
            self.input = self.init_completion(img,mask_in,input_mode)
        else:
            raise('Wrong mode')
            
        self.input = t.unsqueeze(self.input,dim=0)
        self.input = t.unsqueeze(self.input,dim=0)
        self.input = self.input.cuda(cuda_num)
        self.data = self.init_data()
        self.opt = self.init_opt(lr=lr,opt_type=opt_type)
        
    def init_para(self,para):
        # Initial the parameter (Deep Image Prior)
        input_depth = 1
        pad = 'reflection'
        dtype = torch.cuda.FloatTensor
        net = get_net(input_depth, para, pad,
                      skip_n33d=64, 
                      skip_n33u=64, 
                      skip_n11=4, 
                      num_scales=5,
                      upsample_mode='bilinear',
                      n_channels=1).type(dtype)
        return net.cuda(cuda_num)

    def init_data(self):
        # Initial data
        #print(self.input.shape)
        pre_img = self.net(self.input)
        pre_img = t.squeeze(pre_img,dim=0)
        pre_img = t.squeeze(pre_img,dim=0)
        #print(pre_img.shape)
        return pre_img
    
    def init_completion(self,img,mask_in,init_mode):
        # Both the input img and mask_in are the tensor on cuda
        # We will translate them into numpy
        from fancyimpute import KNN
        X_incomplete = img.cpu().detach().numpy().copy()
        mask_in = mask_in.cpu().detach().numpy()
        X_incomplete[(1-mask_in).astype(bool)] = None
        if init_mode == 'knn':
            X_filled = KNN(k=3,verbose=False).fit_transform(X_incomplete)
        elif method_name == 'nnm':
            X_filled = NuclearNormMinimization().fit_transform(X_incomplete)
        elif method_name == 'softimpute':
            X_filled = SoftImpute(verbose=False).fit_transform(X_incomplete)
        elif method_name == 'simple':
            X_filled = SimpleFill().fit_transform(X_incomplete)
        elif method_name == 'itesvd':
            X_filled = IterativeSVD(20,verbose=False).fit_transform(X_incomplete)
        elif method_name == 'mc':
            X_filled = MatrixFactorization(verbose=False).fit_transform(X_incomplete)
        elif method_name == 'ii':
            X_filled = IterativeImputer(verbose=False).fit_transform(X_incomplete)
        else:
            raise('Wrong method_name.')
        return t.tensor(X_filled).cuda(cuda_num)

class nl_dmf(dmf):
    # Nonlinear deep matrix factorization
    def __init__(self,params):
        dmf.__init__(self,params)

    def init_data(self):
        # Initial data
        def get_e2e(model):
            #获取预测矩阵
            weight = None
            for fc in model.children():
                assert isinstance(fc, nn.Linear) and fc.bias is None
                if weight is None:
                    weight = fc.weight.t()
                else:
                    weight = fc(t.sigmoid(weight))
            return t.sigmoid(weight)
        return get_e2e(self.net)
    

class fp(basic_net):
    def __init__(self,params,img,lr=1e-3,std_b=1e-3):
        self.type = 'fp'
        self.net = self.init_para(params,std_b=std_b)
        self.img = img
        self.img2cor()
        #print(self.input.shape)
        self.data = self.init_data()
        self.opt = self.init_opt(lr)
        
    def img2cor(self):
        # 给定m*n灰度图像，返回mn*2
        img_numpy = self.img.cpu().detach().numpy()
        self.m,self.n = img_numpy.shape[0],img_numpy.shape[1]
        x = np.linspace(0,1,self.n)-0.5
        y = np.linspace(0,1,self.m)-0.5
        xx,yy = np.meshgrid(x,y)
        self.xyz = np.stack([xx,yy],axis=2).astype('float32')
        self.input = t.tensor(self.xyz).cuda(cuda_num).reshape(-1,2)

    def cor2img(self,img):
        # 给定形状为mn*1的网络输出，返回m*n的灰度图像
        return img.reshape(self.m,self.n)
    
    def init_para(self,params,std_b):
        model = bias_net(params,std_b).cuda(cuda_num)
        return model
    
    def init_data(self):
        # Initial data
        #print(self.img.shape)
        pre_img = self.net(self.input)
        return self.cor2img(pre_img)

class fc(basic_net):
    # fully_connected such as for AutoEncoderDecoder
    def __init__(self,params,img,lr=1e-3,std_b=1e-3):
        self.type = 'fc'
        self.m,self.n = img.shape[0],img.shape[1]
        params.insert(0,self.m*self.n//100)
        params.append(self.m*self.n)
        self.net = self.init_para(params,std_b=std_b)
        self.img = img
        self.input = t.rand(1,self.m*self.n//100)*1e-1
        self.input = self.input.cuda(cuda_num)
        #print(self.input.shape)
        self.data = self.init_data()
        self.opt = self.init_opt(lr)
        

    def cor2img(self,img):
        # 给定形状为mn*1的网络输出，返回m*n的灰度图像
        return img.reshape(self.m,self.n)
    
    def init_para(self,params,std_b):
        model = bias_net(params,std_b).cuda(cuda_num)
        return model
    
    def init_data(self):
        # Initial data
        #print(self.img.shape)
        pre_img = self.net(self.input)
        return self.cor2img(pre_img)

    
