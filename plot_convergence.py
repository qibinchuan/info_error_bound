# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse
import glob
import os
import sys

import matplotlib
matplotlib.use('Agg')

from matplotlib import pyplot as plt
plt.rc('font', family='serif')
plt.rc('font', size=12)

import numpy as np

init_names = [
    "uniform",
    "kaiming_uniform",
    "xavier_uniform",
    "normal",
    "kaiming_normal",
    "xavier_normal",
    "orthogonal"
]

def get_color_list():
    color_list=['#60966D', '#5B3660','#FFC839','#E90F44','#63ADEE','y','b', 'g', 'r', 'c', 'm', 'k', 'w']
    return color_list
    
def calc_personal_correlation(x, y):
    x_mean, y_mean = np.mean(x), np.mean(y)
    x_std, y_std = np.std(x), np.std(y)
    x_std *= y_std
    if len(x) <= 1:
        print("Not enough data points to calculate correlation")
        return 0
    covariance = np.sum((x - x_mean) * (y - y_mean)) / len(x)
    if x_std == 0 or y_std == 0: 
        print("x or y has no variance")
        return 0
    return covariance / x_std


def plot_diff_layers(net_name,beg,end):
    # 对比同一网络不同层数的收敛情况
    color_list=get_color_list()
    save_dir = "results/"
    snet_res_net_name = "SNet_res_1_lay"
    lnet_net_name = "LNet_nores_1_lay"
    cnet_net_name = "CNet_nores_1_lay"
    plt.figure(figsize=(10, 9))
    alpha = 1.0

    file_name = os.path.join(save_dir,snet_res_net_name+"*.txt")
    deeps = glob.glob(file_name)
    print("len:",len(deeps))    
    plt.subplot(3, 1, 1)
    for idx, deep in enumerate(deeps):
        print(deep)
        if idx >3:
            break
        data = np.genfromtxt(deep)
        if len(data):
            plt.title(r'Model b')
            #plt.plot(data[beg:end, 0], data[beg:end, 1] - data[beg:end, 3], color="b", alpha=alpha,label=r'$\log_2 F(\theta,q_{Y|x})$')#
            #plt.plot(data[beg:end, 0], data[beg:end, 4], color="r", alpha=alpha,label=r'$\log_2 R_{\ell}(f_\theta(x),q_{Y|x})$')#, 
            plt.plot(data[beg:end, 0], np.log2(np.power(2,data[beg:end, 4])-np.power(2,data[beg:end, 1] - data[beg:end, 3])), color_list[idx], alpha=alpha,label=r'$k={}$'.format(idx+1))#, 

            plt.margins(0.05)
            #if idx<2:
            #    plt.xticks([])
            #if idx!=0 and idx !=3:
            #    plt.yticks([])

            #plt.title(net_name)
            #if idx>1:
            #    plt.xlabel("batch number")
            #if idx==0:
            plt.legend()


    file_name = os.path.join(save_dir,lnet_net_name+"*.txt")
    deeps = glob.glob(file_name)
    print("len:",len(deeps))    
    plt.subplot(3, 1, 2)
    for idx, deep in enumerate(deeps):
        print(deep)
        if idx >3:
            break
        data = np.genfromtxt(deep)
        if len(data):
            plt.title(r'Model c')
            #plt.plot(data[beg:end, 0], data[beg:end, 1] - data[beg:end, 3], color="b", alpha=alpha,label=r'$\log_2 F(\theta,q_{Y|x})$')#
            #plt.plot(data[beg:end, 0], data[beg:end, 4], color="r", alpha=alpha,label=r'$\log_2 R_{\ell}(f_\theta(x),q_{Y|x})$')#, 
            plt.plot(data[beg:end, 0], np.log2(np.power(2,data[beg:end, 4])-np.power(2,data[beg:end, 1] - data[beg:end, 3])), color=color_list[idx], alpha=alpha,label=r'$k={}$'.format(idx+1))#, 

            plt.margins(0.05)
            #if idx<2:
            #    plt.xticks([])
            #if idx!=0 and idx !=3:
            #    plt.yticks([])

            #plt.title(net_name)
            #if idx>1:
            #    plt.xlabel("batch number")
            #if idx==0:
            plt.legend()

    file_name = os.path.join(save_dir,cnet_net_name+"*.txt")
    deeps = glob.glob(file_name)
    print("len:",len(deeps))    
    plt.subplot(3, 1, 3)
    for idx, deep in enumerate(deeps):
        print(deep)
        if idx >3:
            break
        data = np.genfromtxt(deep)
        if len(data):
            plt.title(r'Model d')
            #plt.plot(data[beg:end, 0], data[beg:end, 1] - data[beg:end, 3], color="b", alpha=alpha,label=r'$\log_2 F(\theta,q_{Y|x})$')#
            #plt.plot(data[beg:end, 0], data[beg:end, 4], color="r", alpha=alpha,label=r'$\log_2 R_{\ell}(f_\theta(x),q_{Y|x})$')#, 
            plt.plot(data[beg:end, 0], np.log2(np.power(2,data[beg:end, 4])-np.power(2,data[beg:end, 1] - data[beg:end, 3])), color=color_list[idx], alpha=alpha,label=r'$k={}$'.format(idx+1))#, 

            plt.margins(0.05)
            #if idx<2:
            #    plt.xticks([])
            #if idx!=0 and idx !=3:
            #    plt.yticks([])

            #plt.title(net_name)
            #if idx>1:
            #    plt.xlabel("batch number")
            #if idx==0:
            plt.legend()



    from matplotlib.ticker import MaxNLocator
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    #plt.show()

    plt.tight_layout(pad=0)
    pdf_name = os.path.join(save_dir,
                            '{}_plot.pdf'.format(net_name))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(pdf_name)


def plot_compare_one_layer_diffnet(beg,end):
    # 把不同模型在一层情况下的收敛对比

    color_list=get_color_list()
    save_dir = "results/"
    net_list = ["LNet_nores_1_lay1","CNet_nores_1_lay1","SNet_res_1_lay1","ResNet18_res_1_lay1"]  
    model_name_dic = {"ResNet18_res_1_lay1":"Model d","CNet_nores_1_lay1":"Model b","LNet_nores_1_lay1":"Model a","SNet_res_1_lay1":"Model c"}
    alpha = 1.0
    plt.figure(figsize=(10, 6))

    for fig_idx, net_name in enumerate(net_list):
        file_name = os.path.join(save_dir,net_name+".txt")
        deeps = glob.glob(file_name)
        # print("len:",len(deeps))
        for idx, deep in enumerate(deeps):
            print(deep)
            data = np.genfromtxt(deep)
            if len(data):
                plt.subplot(2, 2, fig_idx+1)
                plt.title(model_name_dic[net_name])
                plt.plot(data[beg:end, 0], data[beg:end, 1] + data[beg:end, 3]- data[beg:end, 2], color=color_list[0], alpha=alpha,label=r'$\log_2 F(\theta,q_{Y|x})$')#
                plt.plot(data[beg:end, 0], data[beg:end, 4], color=color_list[3], alpha=alpha,label=r'$\log_2 R_{\ell}(f_\theta(x),\bar q_{Y|x})$')#, 

                plt.margins(0.05)
                if fig_idx==0 or fig_idx==1:
                    plt.xticks([])
                if fig_idx==2 or fig_idx==3:
                    plt.xlabel("batch index")
            
                

    plt.legend()#frameon=False)
    from matplotlib.ticker import MaxNLocator
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    #plt.show()


    plt.tight_layout(pad=0)
    pdf_name = os.path.join(save_dir,
                            'convergence_indicator.pdf'.format(net_name))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(pdf_name)
    
def plot_compare_onenet_diff_layers(net_name,beg,end):
    # 对比同一网络不同层数的收敛情况
    color_list=get_color_list()
    save_dir = "results/"
    plt.figure(figsize=(10, 6))
    alpha = 1.0

    file_name = os.path.join(save_dir,net_name+"*.txt")
    deeps = glob.glob(file_name)
    print("len:",len(deeps))    
    plt.subplot(1, 1, 1)
    for idx, deep in enumerate(deeps):
        print(deep)
        if idx >4:
            #continue
            break
        data = np.genfromtxt(deep)
        if len(data):
            plt.title(r'Model b')
            #plt.plot(data[beg:end, 0], data[beg:end, 1] - data[beg:end, 3], color="b", alpha=alpha,label=r'$\log_2 F(\theta,q_{Y|x})$')#
            #plt.plot(data[beg:end, 0], data[beg:end, 4], color="r", alpha=alpha,label=r'$\log_2 R_{\ell}(f_\theta(x),q_{Y|x})$')#, 


            F = data[beg:end, 1] + data[beg:end, 3]- data[beg:end, 2]
            G= data[beg:end, 4]-F
            # np.log2(np.power(2,data[beg:end, 4])-np.power(2,F))
            mean_G = np.mean(G[300:end])
            mean_F = np.mean(F[300:end])
            mean_R = np.mean(data[300:end, 4])

            ref_G = np.full((len(data[beg:end, 4]), 1), mean_G)
            ref_F = np.full((len(data[beg:end, 4]), 1), mean_F)
            ref_R = np.full((len(data[beg:end, 4]), 1), mean_R)
           

            #plt.plot(data[beg:end, 0], G, color=color_list[idx], alpha=alpha,label=r'$k={}$'.
            #format(idx+1))#, 

            plt.plot(data[beg:end, 0], F, color=color_list[idx], alpha=alpha,label=r'$F:k={}$'.format(idx+1))#, 

            #plt.plot(data[beg:end, 0], data[beg:end, 4], color=color_list[idx], alpha=alpha,label=r'$G:k={}$'.format(idx+1))#, 
            #plt.plot(data[beg:end, 0], ref_F, color=color_list[idx], alpha=alpha,label=r'$G:k={}$'.format(idx+1))#, 
            


            plt.margins(0.05)
            #if idx<2:
            #    plt.xticks([])
            #if idx!=0 and idx !=3:
            #    plt.yticks([])

            #plt.title(net_name)
            #if idx>1:
            #    plt.xlabel("batch number")
            #if idx==0:
            plt.legend()



    from matplotlib.ticker import MaxNLocator
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    #plt.show()

    plt.tight_layout(pad=0)
    pdf_name = os.path.join(save_dir,
                            '{}_plot.pdf'.format(net_name))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(pdf_name)

def calc_corr():
    # 计算不同网络的R和相关性
    save_dir = "results/"
    net_list = ["ResNet18_res_1_lay1","CNet_nores_1_lay1","SNet_nores_2_lay1",
                "SNet_res_1_lay1","SNet_res_1_lay2","SNet_res_1_lay3",
                "SNet_res_1_lay4","SNet_res_1_lay5","SNet_res_1_lay6",
                "SNet_res_1_lay7","SNet_res_1_lay8"]  
    net_list = ["ResNet18_res_1_lay1","SNet_res_1_lay1","CNet_nores_1_lay1","LNet_nores_1_lay1"]  
    ret_dic = {}
    for net_name in net_list:
        file_name = os.path.join(save_dir,net_name+".txt")
        deeps = glob.glob(file_name)
        for idx, deep in enumerate(deeps):
            print(deep)
            data = np.genfromtxt(deep)
            F_v = data[beg:end, 1] +data[beg:end, 3]- data[beg:end, 2] # +max-min
            R_v = data[beg:end, 4]
            r = calc_personal_correlation(np.array(list(R_v)), np.array(list(F_v)))
            print(f'net_nam:{net_name}: ePearson Correlation: {r}')
            ret_dic[net_name]=r


    ret_file_name = os.path.join(save_dir,'all_coor.txt')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(os.path.join(ret_file_name), "w") as f:
        f.write("# {}\n".format(str(ret_file_name)))
        for key,value in ret_dic.items():
            f.write("{} {:.4f}\n".format(key,value))  


def plot_training_corr():
    # 计算不同网络随着训练的进行相关度的变化
    save_dir = "results/"
    step = 50
    net_list = ["LNet_nores_1_lay1","CNet_nores_1_lay1","SNet_res_1_lay1","ResNet18_res_1_lay1"]  
    model_name_dic = {"ResNet18_res_1_lay1":"Model d","CNet_nores_1_lay1":"Model b","LNet_nores_1_lay1":"Model a","SNet_res_1_lay1":"Model c"}

    ret_dic = {}
    for net_name in net_list:
        file_name = os.path.join(save_dir,net_name+".txt")
        deeps = glob.glob(file_name)

        for idx, deep in enumerate(deeps):
            print(deep)
            ret_dic[net_name]=[]
            data = np.genfromtxt(deep)
 
            for i in range(2,len(data[:,0])-step):
                cur_beg,cur_end = 0,0
                if i< step:
                    cur_beg,cur_end=0,i
                else:
                    cur_beg,cur_end=i-step,i
                F_v = data[cur_beg:cur_end, 1] +data[cur_beg:cur_end, 3]- data[cur_beg:cur_end, 2] # +max-min
                R_v = data[cur_beg:cur_end, 4]
                r = calc_personal_correlation(np.array(list(R_v)), np.array(list(F_v)))

                #print(f'net_nam:{net_name}: ePearson Correlation: {r}')
                ret_dic[net_name].append(r)

    color_list=get_color_list()
    plt.subplot(1, 1, 1)
    plt.title(r'The local Pearson correlation coefficient')


    for idx, (name,value) in enumerate(ret_dic.items()):
        
        value_mean = np.mean(value)
        print("name:{},mean:{}".format(name,value_mean))
        ref_mean = np.full((len(value),1),value_mean) 
        cur_beg = 0
        value = value[cur_beg:-1]
        plt.plot(range(beg+2*cur_beg,len(value)*2+2*cur_beg,2),value,color=color_list[idx], alpha=1.0,label=model_name_dic[name])#,
        #plt.plot(range(0,len(value)*2,2),ref_mean,color=color_list[idx], alpha=1.0,label=model_name_dic[name])#, 
    plt.xlabel("batch index")
    plt.margins(0.05)
    plt.legend()
    from matplotlib.ticker import MaxNLocator
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    #plt.show()
    plt.tight_layout(pad=0)
    pdf_name = os.path.join(save_dir,
                            'local_correlation.pdf')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(pdf_name)

def plot_training_corr_layers(net_type):
    # 计算同一网络，不同层数，随着训练的进行相关度的变化
    save_dir = "results/"
    step = 50
    net_list = []
    if net_type == "SNet_res":
        net_list = ["SNet_res_1_lay1","SNet_res_1_lay2","SNet_res_1_lay3","SNet_res_1_lay4","SNet_res_1_lay5","SNet_res_1_lay6"]#,"SNet_res_1_lay7","SNet_res_1_lay8"
    elif net_type == "SNet_no_res":
        net_list = ["SNet_nores_1_lay1","SNet_nores_1_lay2","SNet_nores_1_lay3","SNet_nores_1_lay4","SNet_nores_1_lay5","SNet_nores_1_lay6"]#,"SNet_nores_1_lay7","SNet_nores_1_lay8"]  
    elif net_type == "ResNet18_res":
        net_list = ["ResNet18_res_1_lay1","ResNet18_res_1_lay2","ResNet18_res_1_lay3","ResNet18_res_1_lay4","ResNet18_res_1_lay5","ResNet18_res_1_lay6"]
    elif net_type == "ResNet18_no_res":
        net_list = ["ResNet18_nores_1_lay1","ResNet18_nores_1_lay2","ResNet18_nores_1_lay3","ResNet18_nores_1_lay4","ResNet18_nores_1_lay5"]
    elif net_type == "CNet_nores":
        # 因为没有resblock 训练过程中梯度非常不稳定
        net_list = ["CNet_nores_1_lay1","CNet_nores_1_lay2","CNet_nores_1_lay3","CNet_nores_1_lay4","CNet_nores_1_lay5","CNet_nores_1_lay6"]#,,"CNet_nores_1_lay7","CNet_nores_1_lay8"]  
    else:
        print("wrong net name")
        return 0
    #net_list = ["LNet_nores_1_lay1","CNet_nores_1_lay1","SNet_res_1_lay1","ResNet18_res_1_lay1"]
    model_name_dic = {"ResNet18_res_1_lay1":"Model d","CNet_nores_1_lay1":"Model b","LNet_nores_1_lay1":"Model a","SNet_res_1_lay1":"Model c"}
    model_name_dic={}
    for name in net_list:
        model_name_dic[name]=name[-1]
    ret_dic = {}
    for net_name in net_list:
        file_name = os.path.join(save_dir,net_name+".txt")
        deeps = glob.glob(file_name)

        for idx, deep in enumerate(deeps):
            print(deep)
            ret_dic[net_name]=[]
            data = np.genfromtxt(deep)
 
            for i in range(10,len(data[:,0])-step):
                cur_beg,cur_end = 0,0
                if i< step:
                    cur_beg,cur_end=0,i
                else:
                    cur_beg,cur_end=i-step,i
                F_v = data[cur_beg:cur_end, 1] +data[cur_beg:cur_end, 3]- data[cur_beg:cur_end, 2] # +max-min
                R_v = data[cur_beg:cur_end, 4]
                r = calc_personal_correlation(np.array(list(R_v)), np.array(list(F_v)))

                #print(f'net_nam:{net_name}: ePearson Correlation: {r}')
                ret_dic[net_name].append(r)

    color_list=get_color_list()
    plt.subplot(1, 1, 1)
    if net_type=="SNet_res":
        plt.title(r'Model c')
    elif net_type=="ResNet18_res":
        plt.title(r'Model d')
    else:
        print("error!!")
        return 0

    for idx, (name,value) in enumerate(ret_dic.items()):
        
        value_mean = np.mean(value)
        print("name:{},mean:{}".format(name,value_mean))
        ref_mean = np.full((len(value),1),value_mean) 
        cur_beg = 0
        value = value[cur_beg:-1]
        plt.plot(range(beg+2*cur_beg,len(value)*2+2*cur_beg,2),value,color=color_list[idx], alpha=1.0,label=r'$k$={}'.format(model_name_dic[name]))#,
        #plt.plot(range(0,len(value)*2,2),ref_mean,color=color_list[idx], alpha=1.0,label=model_name_dic[name])#, 

    plt.xlabel("batch index")
    plt.margins(0.05)
    plt.legend()
    from matplotlib.ticker import MaxNLocator
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    #plt.show()
    plt.tight_layout(pad=0)
    pdf_name = os.path.join(save_dir,net_type+
                            'local_correlation_diff_layers.pdf')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(pdf_name)


if __name__ == "__main__":
    #net_name = "ResNet18_res_1_lay1" # lr=0.002
    #net_name = "LNet_nores_1_lay1" # 不用数值出现-50
    #net_name = "CNet_nores_1_lay1"
    #net_name = "SNet_res_1_lay1"  
    #net_name = "SNet_nores_1_lay1"  
    beg, end = 0,-1

    #calc_corr()
    #sys.exit()
    #plot_training_corr()

    #sys.exit
    #net_name = "SNet_res_1"
    #plot_compare_one_layer_diffnet(beg,end)
    #sys.exit()
    net_type = "SNet_res"
    #net_type = "ResNet18_res"
    #plot_compare_onenet_diff_layers(net_type,beg,end)
    plot_training_corr_layers(net_type)