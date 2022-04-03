import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import misc
def gray_im(img,path='./test.jpg',save_if=False):
    show_pic = np.clip(img,0,1)
    plt.imshow(show_pic,'gray',vmin=0,vmax=1)
    plt.grid(0)
    plt.axis('off')
    plt.show()
    if save_if:
        misc.imsave(path, show_pic)

def red_im(img,path='./test.jpg',save_if=False):
    sns.set()
    show_pic = np.clip(img,0,1)
    sns.heatmap(show_pic,vmin=0,vmax=1)
    plt.grid(0)
    plt.axis('off')
    plt.show()
    if save_if:
        misc.imsave(path, show_pic)


def lines(line_dict,xlabel_name='epoch',ylabel_name='MSE',
          ylog_if=False,save_if=False,path='./lines.jpg',black_if=False):
    if black_if:
        sns.set()
    else:
        sns.set_style("whitegrid")  
    for name in line_dict.keys():
        if name != 'x_plot':
            plt.plot(line_dict['x_plot'],line_dict[name],label=name)
    plt.legend()
    plt.xlabel(xlabel_name)
    plt.ylabel(ylabel_name)
    if ylog_if:
        plt.yscale('log')
    if save_if:
        plt.savefig(path)
    plt.show()

def plot_pro(pro_mode,pro_dict,color_list=[],s=100):
    # Plot the projection
    # First plot the real point
    if pro_mode == 'mask':
        plt.scatter(0,0,marker='*',s=s,c='b')
        plt.xlabel('Obs_MSE')
        plt.ylabel('Unk_MSE')
    elif pro_mode == 'svd':
        plt.scatter(1,1,marker='*',s=s,c='b')
        plt.xlabel('Pro_Main')
        plt.ylabel('Pro_Sec')
        plt.xlim(-0.1,1.2)
        plt.ylim(-0.1,1.2)
    else:
        raise('Wrong pro_mode')
    # Define the single trajectory plot
    def plot_single_pro(pro_list,c,label):
        pro_arr = np.array(pro_list)
        #plt.scatter(pro_arr[:,0],pro_arr[:,1])
        plt.plot(pro_arr[:,0],pro_arr[:,1],c=c)
        plt.scatter(pro_arr[-1,0],pro_arr[-1,1],marker='d',s=100,c=c,label=label)
        plt.scatter(pro_arr[0,0],pro_arr[0,1],marker='o',s=100,c=c)
    # Start plot the trajectory
    for i,pro_name in enumerate(pro_dict.keys()):
        pro_list = pro_dict[pro_name]
        if i < len(color_list):
            plot_single_pro(pro_list,c=color_list[i],label=pro_name)
    plt.legend()
    plt.show()
    
    
def range_plot(x_plot=None,range_dict=None):
    # Input a range_dict, every elements in range_dict is a 'array' shaped (num,length)
    def singe_range_plot(fig,ax,x_plot,arr_now,color_name,label):
        arr_avg = arr_now.mean(axis=0)
        arr_max = arr_now.max(axis=0)
        arr_min = arr_now.min(axis=0)
        ax.plot(x_plot,arr_avg,sns.xkcd_rgb[color_name],label=label)
        ax.fill_between(x_plot,arr_min,arr_max,color=sns.xkcd_rgb[color_name],alpha=0.2)
        
    if x_plot == None:
        key_list = list(range_dict)
        x_plot = np.arange(range_dict[key_list[0]].shape[1])
    color_list = list(sns.xkcd_rgb)
    fig, ax = plt.subplots()
    for i,key in enumerate(range_dict.keys()):
        arr_now = range_dict[key]
        color_name = color_list[i]
        singe_range_plot(fig,ax,x_plot,arr_now,color_name,label=key)
        

def plot_tra(pro_list=None,color=None,label=None,end=False,mid_step=None,linewidth=2.5,pointsize=10):
    pro_arr = np.array(pro_list)
    if not end:
        plt.plot(pro_arr[:,0],pro_arr[:,1],color=color,label=label,linewidth=linewidth)
        plt.scatter(pro_arr[0,0],pro_arr[0,1],color=color,marker='o',s=pointsize)
        plt.scatter(pro_arr[-1,0],pro_arr[-1,1],color=color,marker='d',s=pointsize)
        if mid_step != None:
            plt.scatter(pro_arr[mid_step,0],pro_arr[mid_step,1],color=color,marker='^',s=pointsize)
            
    else:
        plt.scatter(pro_arr[-1,0],pro_arr[-1,1],color=color,marker='d',label=label,s=pointsize)
