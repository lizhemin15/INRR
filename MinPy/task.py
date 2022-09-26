import os
import sys
from unittest.mock import patch
import torch as t
from torch.utils.data import DataLoader
import numpy as np
import pickle as pkl
current_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(current_dir)
sys.path.append("..")

from config import settings
cuda_if = settings.cuda_if
cuda_num = settings.cuda_num

from toolbox.projection import svd_pro,mask_pro
from toolbox import dataloader,plot,pprint
import reg,demo,loss


class basic_task(object):
    def __init__(self):
        pass

    def init_mask(self,mask_mode='random',random_rate=0.5,mask_path=None,given_mask=None,patch_num=3):
        if mask_mode == 'random':
            #transformer = dataloader.data_transform(z=self.pic,return_type='tensor')
            mask_mask = t.rand(self.m,self.n)#transformer.get_drop_mask(rate=random_rate) #rate为丢失率
            mask_in = t.ones((self.m,self.n))
            mask_in[mask_mask<random_rate] = 0
            if cuda_if:
                mask_in = mask_in.cuda(cuda_num)
        elif mask_mode == 'patch':
            if cuda_if:
                mask_in = t.ones((self.m,self.n)).cuda(cuda_num)
            else:
                mask_in = t.ones((self.m,self.n))
            mask_in[70:100,150:190] = 0
            mask_in[200:230,200:230] = 0
        elif mask_mode == 'fixed':
            mask_in = dataloader.get_data(height=self.m,width=self.n,pic_name=mask_path)
            mask_in[mask_in<1] = 0
        elif mask_mode == 'patch_num':
            if cuda_if:
                mask_in = t.ones((self.m,self.n)).cuda(cuda_num)
            else:
                mask_in = t.ones((self.m,self.n))
            pixel_m = self.m//patch_num
            pixel_n = self.n//patch_num
            for i in range(self.m):
                for j in range(self.n):
                    if (i%pixel_m-pixel_m/2.0)*(j%pixel_n-pixel_n/2.0)<0:
                        mask_in[i,j] = 0
        elif mask_mode == 'down_sample':
            if cuda_if:
                mask_in = t.zeros((self.m,self.n)).cuda(cuda_num)
            else:
                mask_in = t.zeros((self.m,self.n))
            mask_in[::patch_num,::patch_num] = 1
        elif mask_mode == 'given':
            mask_in = given_mask
        if cuda_if:
            self.mask_in = mask_in.cuda(cuda_num)
        else:
            self.mask_in = mask_in
        #plot.gray_im(self.mask_in.cpu())

    def init_data(self,m=240,n=240,data_path=None,shuffle_mode='I'):
        ori_pic = dataloader.get_data(height=n,width=m,pic_name=data_path)
        def add_noise(pic,noise_dict):
            def get_gauss_noisy_image(img_np, sigma):
                """Adds Gaussian noise to an image.
                Args: 
                    img_np: image, np.array with values from 0 to 1
                    sigma: std of the noise
                """
                img_noisy_np = np.clip(img_np + np.random.normal(scale=sigma, size=img_np.shape), 0, 1).astype(np.float32)
                return t.from_numpy(img_noisy_np)

            def get_salt_noisy_image(img_np, SNR):
                """增加椒盐噪声
                Args:
                    snr （float）: Signal Noise Rate
                    p (float): 概率值，依概率执行该操作
                """
                #if img_np.shape
                h, w = img_np.shape
                img_new = img_np.copy()
                mask = np.random.choice((0, 1, 2), size=(h, w), p=[SNR, (1 - SNR) / 2., (1 - SNR) / 2.])
                img_new[mask == 1] = 1   # 盐噪声
                img_new[mask == 2] = 0      # 椒噪声
                return t.from_numpy(img_new)

            def get_poisson_noisy_image(img_np, lam):
                """增加泊松噪声
                """
                shape=img_np.shape
                lam=lam*np.ones((shape[0],1))
                img_noisy_np =np.clip(np.random.poisson(lam=lam*img_np, size=img_np.shape)/lam, 0, 1).astype(np.float32)
                return t.from_numpy(img_noisy_np)

            img_np = pic.cpu().detach().numpy()
            if noise_dict['type'] == 'gaussian':
                pic = get_gauss_noisy_image(img_np, noise_dict['sigma'])
            elif noise_dict['type'] == 'salt':
                pic = get_salt_noisy_image(img_np, noise_dict['SNR'])
            elif noise_dict['type'] == 'poisson':
                pic = get_poisson_noisy_image(img_np, noise_dict['lam'])
            else:
                print('Wrong type:',noise_dict['type'])
            return pic
        if self.task_type == 'denoising':
            pic = add_noise(ori_pic,self.noise_dict)
        else:
            pic = ori_pic
        if cuda_if:
            pic = pic.cuda(cuda_num)
            ori_pic = ori_pic.cuda(cuda_num)
        self.data_transform = dataloader.data_transform(pic)
        self.shuffle_list = self.data_transform.get_shuffle_list(shuffle_mode)
        self.pic = self.data_transform.shuffle(M=pic,shuffle_list=self.shuffle_list,mode='from')
        self.ori_pic = self.data_transform.shuffle(M=ori_pic,shuffle_list=self.shuffle_list,mode='from')
        if cuda_if:
            self.pic = self.pic.cuda(cuda_num)
            self.ori_pic = self.ori_pic.cuda(cuda_num)

class shuffle_task(basic_task):
    def __init__(self,m=240,n=240,data_path=None,mask_mode='random',random_rate=0.5,
                 mask_path=None,given_mask=None,para=[2,2000,1000,500,200,1],input_mode='masked',
                std_b=1e-1,reg_mode=None,model_name='dmf',pro_mode='mask',
                 opt_type='Adam',shuffle_mode='I',verbose=False,std_w=1e-3,
                 act='relu',patch_num=3,net_list=['dmf'],n_layers=3,scale_factor=2,model_load_path=None,
                 task_type='completion',noise_dict=None,sample_mode='random',att_para=None,
                 sigma=1,cv_if=False,lr=1e-3,bias_net_if=False,omega=30.,drop_out=[0,0,0,0,0],Lr=None,Lc=None):
        self.m,self.n = m,n
        self.task_type = task_type
        self.cv_if = cv_if
        self.noise_dict = noise_dict
        self.reg_mode = reg_mode
        self.model_name = model_name
        self.init_data(m=m,n=n,data_path=data_path,shuffle_mode=shuffle_mode)
        self.init_mask(mask_mode=mask_mode,random_rate=random_rate,mask_path=mask_path,given_mask=given_mask,patch_num=patch_num)
        if verbose:
            if data_path != None:
                plot.gray_im(self.pic.cpu()*self.mask_in.cpu())
            else:
                plot.red_im(self.pic.cpu()*self.mask_in.cpu())
        self.init_pro(pro_mode=pro_mode)
        self.init_reg(m,n,model_path=model_load_path,sample_mode=sample_mode,Lr=Lr,Lc=Lc)
        self.init_model(model_name=model_name,para=para,
                        input_mode=input_mode,std_b=std_b,
                        opt_type=opt_type,std_w=std_w,act=act,
                        net_list=net_list,n_layers=n_layers,
                        scale_factor=scale_factor,att_para=att_para,
                        sigma=sigma,lr=lr,bias_net_if=bias_net_if,omega=omega,drop_out=drop_out)
        
    
    def get_mask(self,sample_func=None,rate=0.5):
        if sample_func == None:
            self.mask_W = self.mask_in.clone()
            self.mask_B = self.mask_in.clone()
            mask_mask = t.rand(self.mask_in.shape)
            self.mask_W[mask_mask < rate] = 0
            self.mask_B[mask_mask >= rate] = 0
        else:
            pass


    def train(self,epoch=10000,verbose=True,imshow=True,print_epoch=100,
              imshow_epoch=1000,plot_mode='gray',stop_err=None,train_reg_gap=1,
             reg_start_epoch=0,eta=[None,None,None,None],model_save_path=None,
             model_save=False,sample_num=1000,fid_name=None,lr=1e-3,train_B=False,
             train_sigma=False,loss_save=False,loss_save_path=None):
        self.pro_list = []
        for ite in range(epoch):
            if ite>reg_start_epoch:
                if self.reg_mode == 'AIR':
                    eta = [None,1e-4,1e-4,None,None,None,None,None,None]
                elif self.reg_mode == 'TV':
                    eta = [1e-3,None,None,None,None,None,None,None,None]
                elif self.reg_mode == 'L2':
                    eta = [None,None,None,1e-3,None,None,None,None,None]
                elif self.reg_mode == 'NN':
                    eta = [None,None,None,None,1,None,None,None,None]
                elif self.reg_mode == 'CAIR':
                    eta = [None,None,None,None,None,1e-4,1e-4,None,None]
                elif self.reg_mode == 'eta':
                    eta = eta
                elif self.reg_mode == 'FLAP':
                    eta = [None,None,None,None,None,None,None,1,1]
                else:
                    eta = [None,None,None,None,None,None,None,None,None]
            else:
                eta = [None,None,None,None,None,None,None,None,None]
            if self.cv_if == False:
                mask_in = self.mask_in.clone()
            elif train_B==True or train_sigma==True:
                mask_in = self.mask_B.clone()
            else:
                mask_in = self.mask_W.clone()

            if ite%train_reg_gap == 0:
                self.model.train(self.pic,mu=1,eta=eta,mask_in=mask_in,train_reg_if=True,sample_num=sample_num,fid_name=fid_name,train_B=train_B,train_sigma=train_sigma)
            else:
                self.model.train(self.pic,mu=1,eta=eta,mask_in=mask_in,train_reg_if=False,sample_num=sample_num,fid_name=fid_name,train_B=train_B,train_sigma=train_sigma)

            if ite % print_epoch==0 and verbose == True:
                pprint.progress_bar(ite,epoch,self.model.loss_dict) # 格式化输出训练的loss，打印出训练进度条
            if ite % imshow_epoch==0 and imshow == True:
                model_data = self.model.net.data
                model_data = self.data_transform.shuffle(M=model_data,shuffle_list=self.shuffle_list,mode='to')
                matrix_data = model_data.cpu().detach().numpy()
                if self.model_name != 'mulbacon':
                    if plot_mode == 'gray':
                        plot.gray_im(matrix_data) # 显示训练的图像，可设置参数保存图像
                    else:
                        plot.red_im(matrix_data) # 显示训练的图像，可设置参数保存图像
                else:
                    for model_data in self.model.net.multi_outputs:
                        model_data = self.data_transform.shuffle(M=model_data,shuffle_list=self.shuffle_list,mode='to')
                        matrix_data = model_data.cpu().detach().numpy()
                        if plot_mode == 'gray':
                            plot.gray_im(matrix_data) # 显示训练的图像，可设置参数保存图像
                        else:
                            plot.red_im(matrix_data) # 显示训练的图像，可设置参数保存图像
                print('RMSE:',t.sqrt(t.mean((self.pic-self.model.net.data)**2)).detach().cpu().numpy())
                print_NMAE = t.sum(t.abs(self.pic-self.model.net.data)*(1-self.mask_in))/(t.max(self.pic)-t.min(self.pic))/t.sum(1-self.mask_in)
                print_NMAE = print_NMAE.detach().cpu().numpy()
                print('NMAE',print_NMAE)
                print('PSNR',loss.psnr(self.ori_pic,self.model.net.data))
            # 添加投影
            self.pro_list.append(self.my_pro.projection(self.model.net.data.cpu().detach().numpy()))
            if stop_err != None:
                if self.pro_list[-1][0]<stop_err:
                    break
        # 绘图
        if imshow == True:
            self.plot(len(self.model.loss_dict['psnr']))
        if model_save == True:
            t.save(self.model.net.net,model_save_path)
        if loss_save == True:
            self.save(self.model.loss_dict,loss_save_path)


        
    def init_pro(self,pro_mode='mask'):
        if pro_mode == 'svd':
            my_pro = svd_pro(self.pic.cpu().detach().numpy())
        elif pro_mode == 'mask':
            my_pro = mask_pro(self.pic.cpu().detach().numpy(),self.mask_in.cpu().detach().numpy())
        else:
            raise('Wrong projection mode')
        self.my_pro = my_pro
    
    def init_reg(self,m=240,n=240,model_path=None,sample_mode='random',sample_num=1000,Lr=None,Lc=None):
        reg_hc = reg.hc_reg(name='lap')
        reg_row = reg.auto_reg(n,'row')
        reg_col = reg.auto_reg(m,'col')
        reg_l2 = reg.hc_reg(name='l2')
        reg_crow = reg.cair_reg(mode='row')
        reg_ccol = reg.cair_reg(mode='col')
        reg_frow = reg.hc_reg(name='flap_row',Lr=Lr)
        reg_fcol = reg.hc_reg(name='flap_col',Lc=Lc)
        if self.reg_mode == 'NN':
            reg_nn = reg.hc_reg(name='nn',model_path=model_path,sample_mode=sample_mode,sample_num=sample_num)
        else:
            reg_nn = reg.hc_reg(name='lap')
        self.reg_list = [reg_hc,reg_row,reg_col,reg_l2,reg_nn,reg_crow,reg_ccol,reg_frow,reg_fcol]
    
    def init_model(self,model_name=None,para=[2,2000,1000,500,200,1],
                    input_mode='masked',std_b=1e-1,opt_type='Adam',
                    std_w=1e-3,act='relu',net_list=['dmf'],n_layers=3,
                    scale_factor=2,att_para=None,sigma=1,lr=1e-3,bias_net_if=False,omega=30.,drop_out=[0,0,0,0,0]):
        if model_name == 'dip':
            model = demo.dip(para=para,reg=self.reg_list,img=self.pic,input_mode=input_mode,mask_in=self.mask_in,opt_type=opt_type)
        elif model_name == 'fp':
            model = demo.fp(para=para,reg=self.reg_list,img=self.pic,std_b=std_b,act=act,std_w=std_w,sigma=sigma,cv_if=self.cv_if,net_lr=lr,bias_net_if=bias_net_if)
        elif model_name == 'dmf':
            model = demo.basic_dmf(para,self.reg_list,std_w)
        elif model_name == 'fc':
            model = demo.fc(para=para,reg=self.reg_list,img=self.pic,std_b=std_b)
        elif model_name == 'fourier' or model_name == 'garbor':
            model = demo.mfn(para=para,reg=self.reg_list,img=self.pic,std_b=std_b,type_name=model_name)
        elif model_name == 'fk':
            model = demo.fk(para=para,reg=self.reg_list,img=self.pic,input_mode=input_mode,mask_in=self.mask_in,opt_type=opt_type)
        elif model_name == 'multi_net':
            model = demo.multi_net(net_list=net_list,reg=self.reg_list,img=self.pic)
        elif model_name == 'msn':
            model = demo.msn(params=para,img=self.pic,reg=self.reg_list,n_layers=n_layers,scale_factor=scale_factor,mainnet_name='fourier')
        elif model_name == 'bacon' or model_name == 'mulbacon':
            model = demo.bacon(params=para,img=self.pic,reg=self.reg_list,type_name=model_name)
        elif model_name == 'siren':
            model = demo.siren(para=para,reg=self.reg_list,img=self.pic,opt_type=opt_type,omega=omega,drop_out=drop_out)
        elif model_name == 'attnet':
            dim_k = att_para['dim_k']
            x_train,x_test = att_para['feature_map'](self.ori_pic,self.mask_in,att_para['map_mode'])
            def get_data(img,mask):
                y_train = img[mask==1].reshape(1,-1,1)
                return y_train
            y_train = get_data(self.ori_pic,self.mask_in)
            model = demo.att(x_train=x_train,y_train=y_train,x_test=x_test,dim_k=dim_k,mask=self.mask_in,reg=self.reg_list)
        self.model = model
    
    def plot(self,epoch):
        line_dict = {}
        line_dict['x_plot']=np.arange(0,epoch,1)
        line_dict[self.reg_mode] = np.array(self.model.loss_dict['psnr'])
        plot.lines(line_dict,save_if=False,black_if=True,ylabel_name='PSNR')
    
    def save(self,data=None,path=None):
        with open(path,'wb') as f:
            pkl.dump(data,f)
   
class kernel_task(basic_task):
    def __init__(self,m=240,n=240,random_rate=0.5,mask_mode='random',
                data_path=None,kernel='gaussian',sigma=1,mask_path=None,
                patch_num=10,feature_type='coordinate',task_type='completion',
                impute_pic=None,weight=None,scale=1):
        self.m,self.n = m,n
        self.task_type = task_type
        self.init_data(m=m,n=n,data_path=data_path)
        self.init_mask(mask_mode=mask_mode,random_rate=random_rate,mask_path=mask_path,patch_num=patch_num)
        self.impute_pic = impute_pic
        self.weight = weight
        self.feature_type = feature_type
        if cuda_if:
            self.scale = scale.cuda(cuda_num)
        else:
            self.scale = scale
        self.init_xy()
        

    def init_xy(self):
        if isinstance(self.feature_type,list):
                    self.x_train,self.x_test,self.y_train = self.combination_data(self.feature_type)
        else:
            self.x_train,self.x_test,self.y_train = self.transformed_data(self.feature_type)
        self.x_train,self.x_test = self.x_train@self.scale,self.x_test@self.scale
        if self.weight != None:
            if cuda_if:
                self.x_train,self.x_test = self.x_train@self.weight.cuda(cuda_num),self.x_test@self.weight.cuda(cuda_num)
            else:
                self.x_train,self.x_test = self.x_train@self.weight,self.x_test@self.weight
    def combination_data(self,feature_list):
        for i,feature in enumerate(feature_list):
            x_train,x_test,y_train = self.transformed_data(feature)
            if i == 0:
                x_train_all = x_train
                x_test_all = x_test
            else:
                x_train_all = t.cat((x_train_all,x_train),axis=1)
                x_test_all = t.cat((x_test_all,x_test),axis=1)
        return x_train_all,x_test_all,y_train


    def transformed_data(self,feature_type):
        if feature_type == 'coordinate' or feature_type == 'random_feature':
            feature_dim = 2
            m,n = self.pic.shape[0],self.pic.shape[1]
            x = np.linspace(-1,1,n)
            y = np.linspace(-1,1,m)
            xx,yy = np.meshgrid(x,y)
            xyz = np.stack([xx,yy],axis=2).astype('float32')
            input = t.tensor(xyz).reshape(-1,2)
            mask_reshape = self.mask_in.reshape(-1)
            N_train = input[:,0][mask_reshape==1].shape[0]
            x_train = t.zeros((N_train,feature_dim))
            x_test = t.zeros((m*n-N_train,feature_dim))
            for i in range(input.shape[1]):
                x_train[:,i] = input[:,i][mask_reshape==1].reshape(-1)
                x_test[:,i] = input[:,i][mask_reshape==0].reshape(-1)
        if feature_type == 'random_feature':
            def random_feature(x,f_dim=1000,sigma=1):
                B = t.randn(x.shape[1],f_dim)*sigma
                x = t.cat((t.cos(x@B),t.sin(x@B)),1)
                return x
            x_train = random_feature(x_train)
            x_test = random_feature(x_test)
        if feature_type == 'patch':
            def img2patch(a):
                m,n = a.shape[0],a.shape[1]
                shift_m = int(np.sqrt(m)/2)*2+1
                shift_n = int(np.sqrt(n)/2)*2+1
                a = a.repeat((shift_m)*(shift_n),3,3)
                center_index = (shift_m)//2*shift_n+shift_n//2
                for i_m in range(shift_m):
                    for i_n in range(shift_n):
                        shift_real_m = i_m-shift_m//2
                        shift_real_n = i_n-shift_n//2
                        if shift_real_m<0:
                            if shift_real_n<0:
                                a[i_m*shift_n+i_n,:shift_real_m,:shift_real_n] = a[center_index,-shift_real_m:,-shift_real_n:]
                            elif shift_real_n==0:
                                a[i_m*shift_n+i_n,:shift_real_m,:] = a[center_index,-shift_real_m:,:]
                            elif shift_real_n>0:
                                a[i_m*shift_n+i_n,:shift_real_m,shift_real_n:] = a[center_index,-shift_real_m:,:-shift_real_n]
                        elif shift_real_m == 0:
                            if shift_real_n<0:
                                a[i_m*shift_n+i_n,:,:shift_real_n] = a[center_index,:,-shift_real_n:]
                            elif shift_real_n==0:
                                a[i_m*shift_n+i_n,:,:] = a[center_index,:,:]
                            elif shift_real_n>0:
                                a[i_m*shift_n+i_n,:,shift_real_n:] = a[center_index,:,:-shift_real_n]
                        else:
                            if shift_real_n<0:
                                a[i_m*shift_n+i_n,shift_real_m:,:shift_real_n] = a[center_index,:-shift_real_m,-shift_real_n:]
                            elif shift_real_n==0:
                                a[i_m*shift_n+i_n,shift_real_m:,:] = a[center_index,:-shift_real_m,:]
                            elif shift_real_n>0:
                                a[i_m*shift_n+i_n,shift_real_m:,shift_real_n:] = a[center_index,:-shift_real_m,:-shift_real_n]
                    return a[:,m:2*m,n:2*n]
            if self.impute_pic == None:
                pass
            else:
                stack_patch = img2patch(self.impute_pic)
                N_train = int(t.sum(self.mask_in).item())
                x_train = t.zeros((N_train,stack_patch.shape[0]))
                x_test = t.zeros((self.m*self.n-N_train,stack_patch.shape[0]))
                for i in range(stack_patch.shape[0]):
                    x_train[:,i] = stack_patch[i,:,:][self.mask_in==1].reshape(-1)
                    x_test[:,i] = stack_patch[i,:,:][self.mask_in==0].reshape(-1)

        if feature_type == 'exchangeable':
            if self.impute_pic == None:
                M_row = t.nn.functional.normalize(self.mask_in.to(t.float32),p=1,dim=1)
                M_col = t.nn.functional.normalize(self.mask_in.to(t.float32),p=1,dim=0)
                if cuda_if:
                    row_mean = t.sum(self.pic*M_row,axis=1).reshape(-1,1)@t.ones((1,self.n)).cuda(cuda_num)
                    col_mean = t.ones((self.m,1)).cuda(cuda_num)@t.sum(self.pic*M_col,axis=1).reshape(1,-1)
                else:
                    row_mean = t.sum(self.pic*M_row,axis=1).reshape(-1,1)@t.ones((1,self.n))
                    col_mean = t.ones((self.m,1))@t.sum(self.pic*M_col,axis=1).reshape(1,-1)
                N_train = int(t.sum(self.mask_in).item())
                x_train = t.zeros((N_train,2))
                x_test = t.zeros((self.m*self.n-N_train,2))
                x_train[:,0] = row_mean[self.mask_in==1].reshape(-1)
                x_train[:,1] = col_mean[self.mask_in==1].reshape(-1)
                x_test[:,0] = row_mean[self.mask_in==0].reshape(-1)
                x_test[:,1] = col_mean[self.mask_in==0].reshape(-1)
            else:
                stack_row = self.impute_pic.repeat(self.n,1,1).transpose(0,1)
                stack_col = self.impute_pic.repeat(self.m,1,1).transpose(1,2)
                stack_all = t.cat((stack_row,stack_col),2)
                N_train = int(t.sum(self.mask_in).item())
                x_train = t.zeros((N_train,stack_all.shape[2]))
                x_test = t.zeros((self.m*self.n-N_train,stack_all.shape[2]))
                for i in range(stack_all.shape[2]):
                    x_train[:,i] = stack_all[:,:,i][self.mask_in==1].reshape(-1)
                    x_test[:,i] = stack_all[:,:,i][self.mask_in==0].reshape(-1)

        if cuda_if:
            x_train = x_train.cuda(cuda_num)
            x_test = x_test.cuda(cuda_num)
        y_train = self.pic[self.mask_in==1].reshape((-1,1))
        if cuda_if:
            y_train = y_train.cuda(cuda_num)
        return x_train,x_test,y_train


    def cal_kernel(self,kernel='gaussian',sigma=1,x=None):
        def gaus_func(x,y,sigma):
            x2 = t.norm(x,dim=1)**2
            if cuda_if:
                x2 = x2.reshape(-1,1)@t.ones((1,y.shape[0])).cuda(cuda_num)
            else:
                x2 = x2.reshape(-1,1)@t.ones((1,y.shape[0]))
            y2 = t.norm(y,dim=1)**2
            if cuda_if:
                y2 = t.ones((x.shape[0],1)).cuda(cuda_num)@y2.reshape(1,-1)
            else:
                y2 = t.ones((x.shape[0],1))@y2.reshape(1,-1)
            xy = x@y.T
            result = (x2+y2-2*xy)/sigma**2
            #result = t.clamp(result,0,10)
            return t.exp(-result)
        

        if kernel == 'gaussian':
            kernel_func = gaus_func

        
        feature_dim_list = self.cal_feature_dim(self.feature_type)
        dim_all = 0
        k_all = t.ones((x.shape[0],self.x_train.shape[0]))
        if cuda_if:
            k_all = k_all.cuda(cuda_num)
        for feature_dim in feature_dim_list:
            k_all = k_all*(1-t.nn.functional.normalize(kernel_func(x[:,dim_all:dim_all+feature_dim],self.x_train[:,dim_all:dim_all+feature_dim],sigma),p=1,dim=1)) #N_test*N_train
            dim_all += feature_dim
        kernel_test = t.nn.functional.normalize(1-k_all,p=1,dim=1)
        if cuda_if:
            kernel_test = kernel_test.cuda(cuda_num)
        return kernel_test

    def cal_feature_dim(self,feature_list):
        dim_list = []
        for feature in feature_list:
            if feature == 'coordinate' or feature == 'exchangeable':
                dim_list.append(2)
            elif feature == 'patch':
                dim_list.append((int(np.sqrt(self.m)/2)*2+1)*(int(np.sqrt(self.n)/2)*2+1))
        return dim_list


    def rf(self,x_train,x_test,D=1000,sigma=1):
        B = t.randn(x_train.shape[1],D)*sigma
        if cuda_if:
            B = B.cuda(cuda_num)
        phi_X = t.cat((t.cos(x_train@B),t.sin(x_train@B)),1)/np.sqrt(D)
        phi_x = t.cat((t.cos(x_test@B),t.sin(x_test@B)),1)/np.sqrt(D)
        phi_XX = phi_X.T@phi_X
        return phi_x@t.pinverse(phi_XX)@phi_X.T@self.y_train

    def rf_sgd(self,x_train,x_test,D=1000,sigma=1,batch_size=256,iteration=100):
        # 1.将N个数据随机映射到 N*2D 维的 phi 矩阵
        # 2.dataloader
        # 3.定义w \in R^{2D*1} 
        # 4.构建损失函数，优化器
        # 5.迭代优化指定步数
        B = t.randn(x_train.shape[1],D)*sigma
        if cuda_if:
            B = B.cuda(cuda_num)
        phi_x = t.cat((t.cos(x_train@B),t.sin(x_train@B)),1)/np.sqrt(D)
        torch_data = dataloader.tensor_to_loader(phi_x,self.y_train)
        dataset = DataLoader(torch_data, batch_size=batch_size, shuffle=True, drop_last=False)
        w = t.rand(2*D,1)
        if cuda_if:
            w = w.cuda(cuda_num)
        w = t.nn.Parameter(w)
        optimizer = t.optim.Adam([w],lr=1e-3)
        for i in range(iteration):
            for _,data in enumerate(dataset):
                optimizer.zero_grad()
                x = data[0]
                y = data[1]
                loss = t.mean((x@w-y)**2)
                loss.backward()
                optimizer.step()
            if i%100 == 0:
                print(loss.item())
        return t.cat((t.cos(x_test@B),t.sin(x_test@B)),1)@w/np.sqrt(D)


    def predict(self,predict_mode='test',x_test=None,kernel='gaussian',sigma=1,batch_num_all=10):
        if predict_mode == 'test':
            x_test = self.x_test
        elif predict_mode == 'train':
            x_test = self.x_train
        elif predict_mode == 'all':
            if self.x_test.shape[0] != 0:
                x_test = t.cat((self.x_train,self.x_test),0)
            else:
                x_test = self.x_train
        if cuda_if:
            y_pre = t.ones((x_test.shape[0],1)).cuda(cuda_num)
        else:
            y_pre = t.ones((x_test.shape[0],1))

        if kernel == 'gaussian':
            batch_size = x_test.shape[0]//batch_num_all
            for batch_num in range(batch_num_all):
                k_now = self.cal_kernel(kernel=kernel,sigma=sigma,x=x_test[batch_num*batch_size:(batch_num+1)*batch_size])
                y_pre[batch_num*batch_size:(batch_num+1)*batch_size,:] = k_now@self.y_train
            if x_test.shape[0]%batch_num_all != 0:
                batch_num += 1
                k_now = self.cal_kernel(kernel=kernel,sigma=sigma,x=x_test[batch_num*batch_size:])
                y_pre[batch_num*batch_size:,:] = k_now@self.y_train
        elif kernel == 'RF':#random feature
            y_pre = self.rf(self.x_train,x_test,D=1000,sigma=1)
        elif kernel == 'RF_SGD':
            y_pre = self.rf_sgd(self.x_train,x_test,D=100,sigma=1,batch_size=256,iteration=1000)


        if predict_mode == 'all':
            img = t.zeros((self.m,self.n)).to(self.mask_in)
            img = img.to(t.float32)
            train_i = t.sum(self.mask_in).detach().cpu().numpy().astype('int')
            img[self.mask_in==1] = y_pre[:train_i].reshape(-1)
            img[self.mask_in==0] = y_pre[train_i:].reshape(-1)
            return img
        else:
            return y_pre

    
class train_kernel_task(basic_task):
    def __init__(self,m=240,n=240,random_rate=0.5,mask_mode='random',
                data_path=None,mask_path=None,weight_mode='all',
                patch_num=10,feature_type='coordinate',task_type='completion',lr=1e-1,scale=t.eye(2)):
        self.m,self.n = m,n
        self.scale = scale
        self.task_type = task_type
        self.random_rate = random_rate
        self.mask_mode = mask_mode
        self.data_path = data_path
        self.mask_path = mask_path
        self.weight_mode = weight_mode
        self.init_data(m=m,n=n,data_path=data_path)
        self.init_mask(mask_mode=mask_mode,random_rate=random_rate,mask_path=mask_path,patch_num=patch_num)
        self.init_weight(feature_type,lr)


    def cal_feature_dim(self,feature_list):
        dim = 0
        for feature in feature_list:
            if feature == 'coordinate' or feature == 'exchangeable':
                dim += 2
            elif feature == 'patch':
                dim += (int(np.sqrt(self.m)/2)*2+1)*(int(np.sqrt(self.n)/2)*2+1)
        return dim

    def init_weight(self,feature_list,lr=1e-3):
        self.feature_dim = self.cal_feature_dim(feature_list)
        self.w = t.eye(self.feature_dim)
        if cuda_if:
            self.w = self.w.cuda(cuda_num)
        self.w = t.nn.Parameter(self.w)
        self.optimizer = t.optim.Adam([self.w],lr=lr)

    def get_mse(self,feature_list,p,impute_pic=None,return_pic=False):
        if self.weight_mode == 'all':
            weight_in = self.w
        else:
            weight_in = self.w*t.eye(self.feature_dim)
        if cuda_if:
            weight_in = weight_in.cuda(cuda_num)
        task_now = kernel_task(m=self.m,n=self.n,random_rate=self.random_rate,mask_mode=self.mask_mode,
                        data_path=self.data_path,kernel='gaussian',sigma=1,mask_path=self.mask_path,
                        patch_num=10,feature_type=feature_list,impute_pic=impute_pic,weight=weight_in,scale=self.scale)
        task_now.mask_in = self.get_mask(self.mask_in,p)
        task_now.init_xy()
        y_pre = task_now.predict('all',sigma=1,kernel='gaussian')
        if return_pic:
            return y_pre
        else:
            return loss.mse(y_pre,self.pic,self.mask_in)

    def get_loss(self,feature_list,p,sample_num=10,impute_pic=None):
        loss_all = 0
        for i in range(sample_num):
            loss_all += self.get_mse(feature_list,p,impute_pic)
        return loss_all
    

    def get_mask(self,mask,p=0.5):
        mask_mask = t.rand(mask.shape)
        mask_new = mask.clone()
        mask_new[mask_mask>1-p] = 0
        return mask_new

    def train(self,feature_list,p=0.5,sample_num=10,impute_pic=None):
        self.optimizer.zero_grad()
        loss_all = self.get_loss(feature_list,p,sample_num,impute_pic)
        loss_all.backward()
        self.optimizer.step()
        return loss_all.item()
    

class sinr(basic_task):
    # 这个体系结构下包含两个部分:存储器和推理器
    # 存储器部分存储指定精度的网格数据
    # 推理器对于网格内任意一点，以该点所在位置和小网格信息，通过参数化的推理器后得到该点预测
    # 训练过程中，可以更新的参数包括：网格上的数据、推理器的参数化
    # 特别的，在使用 softmax 作为加权系数时，其\sigma为输入位置的函数
    # 构建损失函数的时候，可以对参数进行正则，例如我们期望\sigma尽可能小
    def __init__(self,grid_res=256,m=256,n=256,data_path=None,mask_mode='random',random_rate=0.5,
                 mask_path=None,given_mask=None,patch_num=4,batch_size=256):
        self.grid_res = grid_res
        self.m = m
        self.n = n
        self.batch_size = batch_size
        self.task_type = 'completion'
        self.init_B()
        self.init_data(m=m,n=n,data_path=data_path,shuffle_mode='I')
        self.init_mask(mask_mode=mask_mode,random_rate=random_rate,mask_path=mask_path,given_mask=given_mask,patch_num=patch_num)
        self.init_grid()
        self.init_inference()
        self.init_sets()

    def init_grid(self):
        self.grid = self.mask_in*self.ori_pic
        #t.zeros((self.grid_res,self.grid_res))
        if cuda_if:
            self.grid = self.grid.cuda(cuda_num)
        self.grid = t.nn.Parameter(self.grid)
        self.grid_opt = t.optim.Adam([self.grid],lr=1e-3)

    def init_inference(self):
        self.net = t.nn.Sequential(
                t.nn.Linear(400,256),
                t.nn.ReLU(),
                t.nn.Linear(256,256),
                t.nn.ReLU(),
                t.nn.Linear(256,1)
                )
        if cuda_if:
            self.net = self.net.cuda(cuda_num)
        self.net_opt = t.optim.Adam(self.net.parameters(),lr=1e-1)


    def init_sets(self):
        x = np.linspace(0,1,self.n)
        y = np.linspace(0,1,self.m)
        xx,yy = np.meshgrid(x,y)
        xx,yy = xx.T,yy.T
        xyz = np.stack([xx,yy],axis=2).astype('float32')
        xyz = t.tensor(xyz)
        #pic_coor = xyz.reshape(-1,2)
        self.train_x = xyz[self.mask_in==1]
        self.train_y = self.ori_pic[self.mask_in==1]
        self.pre_x = xyz.reshape(-1,2)
        self.pre_y = self.ori_pic.reshape(-1,1)
        if cuda_if:
            self.train_x,self.train_y,self.pre_x,self.pre_y = self.train_x.cuda(cuda_num),self.train_y.cuda(cuda_num),self.pre_x.cuda(cuda_num),self.pre_y.cuda(cuda_num)
        data = t.utils.data.TensorDataset(self.train_x,self.train_y)
        self.data_loader = t.utils.data.DataLoader(data,batch_size = self.batch_size,shuffle=True)
        


    def train(self):
        for _,batch_data in enumerate(self.data_loader):
            x = t.rand([self.batch_size,2])
            # x = batch_data[0]
            # y = batch_data[1]
            # if cuda_if:
            #     x,y = x.cuda(cuda_num),y.cuda(cuda_num)
            if cuda_if:
                x = x.cuda(cuda_num)
            loss = self.get_loss(x)
            self.net_opt.zero_grad()
            self.grid_opt.zero_grad()
            loss.backward(retain_graph=True)
            self.net_opt.step()
            self.grid_opt.step()

    def init_B(self):
        self.B = t.randn(2,100)*1e1
        self.Bw = t.randn(1,100)*1e1
        if cuda_if:
            self.B = self.B.cuda(cuda_num)
            self.Bw = self.Bw.cuda(cuda_num)

    def get_input(self,x):
        input_now = x@self.B
        rf_x= t.cat((t.cos(input_now),t.sin(input_now)),1)
        x = x * t.tensor([self.grid_res,self.grid_res],device=cuda_num).float()
        
        indices = x.long()
        weights = x - indices.float()
        x0 = indices[:,0].clamp(min=0,max=self.grid_res-1)
        y0 = indices[:,1].clamp(min=0,max=self.grid_res-1)
        x1 = (x0+1).clamp(max=self.grid_res-1)
        y1 = (y0+1).clamp(max=self.grid_res-1)
        # [[w0,w1],[w2,w3]]
        w0 = self.grid[x0,y0].reshape(-1,1) * (1.0 - weights[:,0:1]) * (1.0 - weights[:,1:2])
        w1 = self.grid[x0,y1].reshape(-1,1) * weights[:,0:1] * (1.0 - weights[:,1:2])
        w2 = self.grid[x1,y0].reshape(-1,1) * (1.0 - weights[:,0:1]) * weights[:,1:2]
        w3 = self.grid[x1,y1].reshape(-1,1) * weights[:,0:1] * weights[:,1:2]
        w = w0+w1+w2+w3
        rf_w= t.cat((t.cos(w@self.Bw),t.sin(w@self.Bw)),1)

        return  t.cat([rf_x,rf_w],axis=1),w0+w1+w2+w3
        return rf_x,w0+w1+w2+w3

    def get_loss(self,x):
        x_input,y = self.get_input(x)
        pre = self.net(x_input)
        return t.mean((pre-y)**2)

    def predict(self):
        x_input,y_grid = self.get_input(self.pre_x)
        pre_y = self.net(x_input)
        img = pre_y.reshape(self.m,self.n)
        plot.gray_im(y_grid.reshape(self.m,self.n).cpu().detach().numpy())
        plot.gray_im(img.cpu().detach().numpy())
        plot.gray_im(self.grid.cpu().detach().numpy())
        print('RMSE:',t.sqrt(t.mean((self.pic-img)**2)).detach().cpu().numpy())
        print_NMAE = t.sum(t.abs(self.pic-img)*(1-self.mask_in))/(t.max(self.pic)-t.min(self.pic))/t.sum(1-self.mask_in)
        print_NMAE = print_NMAE.detach().cpu().numpy()
        print('NMAE',print_NMAE)
        print('PSNR',loss.psnr(self.ori_pic,img))




