import os
from re import S

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
        elif opt_type == 'RegAdam':
            optimizer = t.optim.Adam(self.net.parameters(),lr=lr, weight_decay=1e-6)
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
        if cuda_if:
            self.input = self.input.cuda(cuda_num)
        self.data = self.init_data()
        self.opt = self.init_opt(lr=lr,opt_type=opt_type)
        
    def init_para(self,para):
        # Initial the parameter (Deep Image Prior)
        input_depth = 1
        pad = 'reflection'
        dtype = torch.cuda.FloatTensor
        net = get_net(input_depth, 'skip', pad,
                      skip_n33d=64, 
                      skip_n33u=64, 
                      skip_n11=4, 
                      num_scales=5,
                      upsample_mode='bilinear',
                      n_channels=1).type(dtype)
        if cuda_if:
            return net.cuda(cuda_num)
        else:
            return net

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
        if cuda_if:
            return t.tensor(X_filled).cuda(cuda_num)
        else:
            return t.tensor(X_filled)

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


class inr(basic_net):
    def __init__(self,params,img,lr=1e-3,std_b=1e-3,act='relu',ynet=None,ysample={}):
        self.type = 'inr'
        params = [2,2000,1000,500,200,1]
        self.net = self.init_para(params,std_b=std_b,act=act)
        self.img = img
        self.img2cor(ynet,ysample)
        #print(self.input.shape)
        self.data = self.init_data()
        self.opt = self.init_opt(lr)
        
    def img2cor(self,ynet,ysample):
        # 给定m*n灰度图像，返回mn*2
        img_numpy = self.img.cpu().detach().numpy()
        self.m,self.n = img_numpy.shape[0],img_numpy.shape[1]
        if ynet == None:
            x = np.linspace(-1,1,self.n)
            y = np.linspace(-1,1,self.m)
            xx,yy = np.meshgrid(x,y)
            self.xyz = np.stack([xx,yy],axis=2).astype('float32')
            self.input = t.tensor(self.xyz).reshape(-1,2)
        else:
            m,n = 0,0
            for key in ysample.keys():
                if key == 'row':
                    n = ysample[key]
                    x = np.linspace(-1,1,n)
                    y = np.linspace(-1,1,self.m)
                    xx,yy = np.meshgrid(x,y)
                    xyz = np.stack([xx,yy],axis=2).astype('float32')
                    in_put = t.tensor(xyz).reshape(-1,2)
                    if cuda_if:
                        in_put = in_put.cuda(cuda_num)
                    feature_x_in = ynet(in_put).detach().reshape(self.m,n)
                    feature_x_in = t.repeat_interleave(feature_x_in.unsqueeze(1),repeats=self.n,dim=1)
                elif key == 'col':
                    m = ysample[key]
                    x = np.linspace(-1,1,self.n)
                    y = np.linspace(-1,1,m)
                    xx,yy = np.meshgrid(x,y)
                    xyz = np.stack([xx,yy],axis=2).astype('float32')
                    in_put = t.tensor(xyz).reshape(-1,2)
                    if cuda_if:
                        in_put = in_put.cuda(cuda_num)
                    feature_y_in = ynet(in_put).detach().reshape(m,self.n).T
                    feature_y_in = t.repeat_interleave(feature_y_in.unsqueeze(0),repeats=self.m,dim=0)
                elif key == 'patch':
                    x1,y1,x2,y2 = ysample[key]
                    pass
            self.input = t.cat([feature_x_in,feature_y_in],dim=2).reshape(-1,m+n)/(m+n)

        if cuda_if:
            self.input = self.input.cuda(cuda_num)
        if self.type=='fp' and self.rf_if:
            if self.train_sigma == False:
                self.init_B()
            else:
                self.init_sigma()


    def cor2img(self,img):
        # 给定形状为mn*1的网络输出，返回m*n的灰度图像
        return img.reshape(self.m,self.n)
    
    def init_para(self,params,std_b,act,std_w,bias_net_if=False):
        if bias_net_if:
            model = bias_net(params,std_b,act=act,std_w=std_w)
        else:
            if act == 'relu':
                nonlinear =  nn.ReLU()
            elif act == 'sigmoid':
                nonlinear =  nn.Sigmoid()
            elif act == 'tanh':
                nonlinear =  nn.Tanh()
            elif act == 'softmax':
                nonlinear =  nn.Softmax()
            elif act == 'threshold':
                nonlinear =  nn.Threshold()
            elif act == 'hardtanh':
                nonlinear =  nn.Hardtanh()
            elif act == 'elu':
                nonlinear =  nn.ELU()
            elif act == 'relu6':
                nonlinear =  nn.ReLU6()
            elif act == 'leaky_relu':
                nonlinear =  nn.LeakyReLU()
            elif act == 'prelu':
                nonlinear =  nn.PReLU()
            elif act == 'rrelu':
                nonlinear =  nn.RReLU()
            elif act == 'logsigmoid':
                nonlinear =  nn.LogSigmoid()
            elif act == 'hardshrink':
                nonlinear =  nn.Hardshrink()
            elif act == 'tanhshrink':
                nonlinear =  nn.Tanhshrink()
            elif act == 'softsign':
                nonlinear =  nn.Softsign()
            elif act == 'softplus':
                nonlinear =  nn.Softplus()
            elif act == 'softmin':
                nonlinear =  nn.Softmin()
            elif act == 'softmax':
                nonlinear =  nn.Softmax()
            elif act == 'log_softmax':
                nonlinear =  nn.LogSoftmax()
            elif act == 'softshrink':
                nonlinear =  nn.Softshrink()
            else:
                print('Wrong act name:',act)
            nn_list = []
            for enu,(n_i,n_o) in enumerate(zip(params[:-1],params[1:])):
                nn_list.append(nn.Linear(n_i,n_o))
                if enu < len(params)-2:
                    nn_list.append(nonlinear)
            model = nn.Sequential(*nn_list)
            print(model)
        if cuda_if:
            model = model.cuda(cuda_num)
        return model
    
    def init_data(self):
        # Initial data
        eye_2 = t.eye(2)
        if cuda_if:
            eye_2 = eye_2.cuda(cuda_num)
        if self.type=='fp' and self.rf_if:
            if isinstance(self.sigma,float) or isinstance(self.sigma,int):
                input_now = self.input@(self.sigma*eye_2)@(self.B)
            else:
                input_now = self.input@(self.sigma[0,0]*eye_2)@(self.B)
            pre_img = self.net(t.cat((t.cos(input_now),t.sin(input_now)),1))
            if self.type == 'mulbacon':
                self.multi_outputs = self.net.multi_outputs
            return self.cor2img(pre_img)
        else:
            return self.cor2img(self.net(self.input))

    def init_B(self):
        self.B = t.randn(self.input.shape[1],self.feature_dim)
        if cuda_if:
                self.B = self.B.cuda(cuda_num)
        if self.cv_if:
            self.B = t.nn.Parameter(self.B)
            self.opt_B = t.optim.Adam([self.B],lr=1.3e0)

    def init_sigma(self):
        if self.type=='fp' and self.rf_if:
            if cuda_if:
                    self.sigma = t.eye(2).cuda(cuda_num)
            if self.cv_if:
                self.sigma = t.nn.Parameter(self.sigma)
                self.opt_sigma = t.optim.Adam([self.sigma],lr=1e1)
            self.B = t.randn(self.input.shape[1],self.feature_dim)
            if cuda_if:
                    self.B = self.B.cuda(cuda_num)

    def update(self):
        self.opt.step()
        self.data = self.init_data()

    def update_B(self):
        self.opt_B.step()
        self.data = self.init_data()

    def update_sigma(self):
        self.opt_sigma.step()
        self.data = self.init_data()

class fp(inr):
    def __init__(self,params,img,lr=1e-3,std_b=1e-3,act='relu',std_w=1e-3,sigma=1,cv_if=False,bias_net_if=False,ynet=None,ysample={}):
        self.type = 'fp'
        if params[0] == 2:
            params = [2,2000,1000,500,200,1]
            self.rf_if = False
        else:
            self.feature_dim =params[0]
            params[0] = params[0]*2
            #params = [params[0]*2,2000,1000,500,200,1]
            self.rf_if = True
            self.cv_if = cv_if
            if isinstance(sigma,str):
                self.train_sigma = True
            else:
                self.train_sigma = False
                self.sigma = sigma
        if act == 'sin':
            hidden_size = img.shape[0]*img.shape[1]//1024
            params = [2,hidden_size,hidden_size,hidden_size,hidden_size,1]
        self.net = self.init_para(params,std_b=std_b,act=act,std_w=std_w,bias_net_if=bias_net_if)
        self.img = img
        self.img2cor(ynet,ysample)
        #print(self.input.shape)
        self.data = self.init_data()
        self.opt = self.init_opt(lr)




class fc(inr):
    # fully_connected such as for AutoEncoderDecoder
    def __init__(self,params,img,lr=1e-3,std_b=1e-3,act='relu'):
        self.type = 'fc'
        params = [2,2000,1000,500,200,1]
        self.m,self.n = img.shape[0],img.shape[1]
        params.insert(0,self.m*self.n//100)
        params.append(self.m*self.n)
        self.net = self.init_para(params,std_b=std_b,act=act)
        self.img = img
        self.input = t.rand(1,self.m*self.n//100)*1e-1
        if cuda_if:
            self.input = self.input.cuda(cuda_num)
        #print(self.input.shape)
        self.data = self.init_data()
        self.opt = self.init_opt(lr)

class mfn(inr):
    def __init__(self,params,img,lr=1e-3,type_name='fourier',ynet=None,ysample={}):
        self.type = type_name
        self.rf_if = False
        self.net = self.init_para(params)
        self.img = img
        self.img2cor(ynet,ysample)
        #print(self.input.shape)
        self.data = self.init_data()
        self.opt = self.init_opt(lr)

    def init_para(self,params):
        in_size = params[0]
        out_size = params[-1]
        hidden_size = params[1]
        n_layers = len(params)-2
        if self.type == 'fourier':
            model = FourierNet(in_size=in_size,hidden_size=hidden_size,out_size=out_size,n_layers=n_layers)
        elif self.type == 'garbor':
            model = GaborNet(in_size=in_size,hidden_size=hidden_size,out_size=out_size,n_layers=n_layers)
        if cuda_if:
            return model.cuda(cuda_num)
        else:
            return model

class fk(basic_net):
    def __init__(self,para,img,lr=1e-3,input_mode='masked',mask_in=None,opt_type='Adam'):
        self.type = 'fk_net'
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
        if cuda_if:
            self.input = self.input.cuda(cuda_num)
        self.net = self.init_para()
        self.data = self.init_data()
        self.opt = self.init_opt(lr=lr,opt_type=opt_type)
    
    def init_para(self):
        if cuda_if:
            model = fk_net(self.input).cuda(cuda_num)
        else:
            model = fk_net(self.input)
        return model

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
        if cuda_if:
            return t.tensor(X_filled).cuda(cuda_num)
        else:
            return t.tensor(X_filled)




class msn(basic_net):
    def __init__(self,params,img,lr=1e-3,n_layers=3,scale_factor=2,mainnet_name='fourier'):
        self.type = 'msn'
        self.net = self.init_para(params,n_layers=n_layers,scale_factor=scale_factor,mainnet_name=mainnet_name)
        self.img = img
        #print(self.input.shape)
        self.data = self.init_data()
        self.opt = self.init_opt(lr)
        
    def init_para(self,params=[256,256],n_layers=3,scale_factor=2,mainnet_name='fourier'):
        if cuda_if:
            model = MSNBase(params=params,n_layers=n_layers,scale_factor=scale_factor,mainnet_name=mainnet_name).cuda(cuda_num)
        else:
            model = MSNBase(params=params,n_layers=n_layers,scale_factor=scale_factor,mainnet_name=mainnet_name)
        return model
    
    def init_data(self):
        # Initial data
        return self.net()
        
class bacon(inr):
    def __init__(self,params,img,lr=1e-3,type_name='bacon',ynet=None,ysample={}):
        self.type = type_name
        self.img = img
        self.net = self.init_para(params)
        self.img2cor(ynet,ysample)
        #print(self.input.shape)
        self.data = self.init_data()
        self.opt = self.init_opt(lr)

    def init_para(self,params):
        hidden_features = params[0]
        out_features = 1
        hidden_layers = params[2]
        res = params[1]
        input_scales = [1/8, 1/8, 1/4, 1/4, 1/4]
        output_layers = [1, 2, 4]
        if self.type == 'mulbacon':
            model = MultiscaleBACON(2, hidden_features, out_size=out_features,
                  hidden_layers=hidden_layers,
                  bias=True,
                  frequency=(res, res),
                  quantization_interval=2*np.pi,
                  input_scales=input_scales,
                  output_layers=output_layers,
                  reuse_filters=False)
        elif self.type == 'bacon':
            model = BACON(2, hidden_features, out_size=out_features,
                  hidden_layers=hidden_layers,
                  bias=True,
                  frequency=(res, res),
                  quantization_interval=2*np.pi,
                  input_scales=input_scales,
                  output_layers=output_layers,
                  reuse_filters=False)
        if cuda_if:
            return model.cuda(cuda_num)
        else:
            return model

class dis_net(basic_net):
    def __init__(self,lr=1e-3):
        self.type = 'dis_net'
        self.net = self.init_para()
        self.opt = self.init_opt(lr)
        
    def init_para(self):
        if cuda_if:
            model = discriminator().cuda(cuda_num)
        else:
            model = discriminator()
        return model

class siren(inr):
    def __init__(self,params,img,lr=1e-3,opt_type='Adam',omega=30.,drop_out=[0,0,0,0,0],ynet=None,ysample={}):
        self.type = 'siren'
        hidden_size = img.shape[0]*img.shape[1]//1024
        in_features = params[0]
        out_features = params[-1]
        hidden_features = params[1]
        hidden_layers = len(params)-2
        self.net = self.init_para(in_features, hidden_features, hidden_layers, out_features,omega=omega,drop_out=drop_out)
        self.img = img
        self.img2cor(ynet,ysample)
        self.data = self.init_data()
        self.opt = self.init_opt(lr,opt_type=opt_type)

    def init_para(self,in_features, hidden_features, hidden_layers, out_features,omega,drop_out):
        model = SirenNet(
                        dim_in = in_features,                        # input dimension, ex. 2d coor
                        dim_hidden = hidden_features,                  # hidden dimension
                        dim_out = out_features,                       # output dimension, ex. rgb value
                        num_layers = hidden_layers,                    # number of layers
                        final_activation = nn.Sigmoid(),   # activation of final layer (nn.Identity() for direct output)
                        w0_initial = omega,                   # different signals may require different omega_0 in the first layer - this is a hyperparameter
                        drop_out = drop_out
                    )
        if cuda_if:
            model = model.cuda(cuda_num)
        return model


class attnet(basic_net):
    def __init__(self,lr=1e-3,x_train=None,y_train=None,x_test=None,dim_k=10,mask=None,performer_if=True):
        # x_train shape: batch_size * seq_len * input_dim
        # y_train shape: batch_size * seq_len * output_dim
        self.type = 'attnet'
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.mask = mask
        self.performer_if = performer_if
        self.dim_k = dim_k
        self.net = self.init_para()
        self.data = self.init_data()
        self.opt = self.init_opt(lr)
        
    def init_para(self):
        input_dim = self.x_train.shape[2]
        if self.performer_if:
            from performer_pytorch import SelfAttention
            model = SelfAttention(dim=self.x_train.shape[2],heads=1,causal=False)
        else:
            model = att_net(input_dim,self.dim_k)
        if cuda_if:
            model = model.cuda(cuda_num)
        return model
    
    def reshape_data(self,vec,mode='train'):
        vec = vec.reshape((vec.shape[1]))
        data = t.zeros(self.mask.shape).to(vec.device)
        if mode == 'train':
            data[self.mask==1] = vec
        elif mode == 'test':
            data[self.mask==0] = vec
        return data

    def init_data(self):
        # Initial data
        atten_train = self.net(self.x_train,self.x_train)
        output_train = t.bmm(atten_train,self.y_train)
        atten_test = self.net(self.x_test,self.x_train)
        output_test = t.bmm(atten_test,self.y_train)
        matrix_train = self.reshape_data(output_train,'train')
        matrix_test = self.reshape_data(output_test,'test')
        return matrix_train+matrix_test