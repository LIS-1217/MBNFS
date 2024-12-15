# -*- coding: utf-8 -*-
import os
import scipy.io as scio
import numpy as np
from getFileInfo import getEDFFile2Mat, getExceledfList
from Edf2MatCCY import Edf2Mat
from get_MI import get_MI_matrix
from brainNetwork import brainigraph
from bnFeature import cal_bnf
from Importance import removeNode,channelImportance,RFimportance
import pandas as pd
import datetime
import joblib
from collections import Counter
from train import load_data
from trainProcrc import train
import warnings
warnings.filterwarnings('ignore')


def train_model(edfFileDir):
    '''
    算法主程序，迭代训练得到模型 
    parameters:
        edfFileDir: 脑电数据集的路径
    return:
        model: 训练得到的模型所在的路径
    '''
    start = datetime.datetime.now()
    print(start)

    # edfFileDir = 'E:/研究生/脑网络/tuh_eeg_seizure/v1.5.0'
    # SaveRootDir = 'E:/研究生/脑网络/tuh_eeg_seizure_classData/'
    # epochdir = '../epoch/'   # 用了733个epoch，没有用所有的epoch
    MatSaveMI = '../MI/'     # 733个MI矩阵存放路径
    # get_MI_matrix(edfFileDir, SaveRootDir, epochdir, MatSaveMI)  # 生成733个MI矩阵（在MI文件夹中）
    MatSaveMI_t = '../MI_temp/'    # 迭代时，删除节点后的MI矩阵存放地址
    MatSaveBnf = '../bnfeature/'   # 网络特征保存路径
    
    t = 1          # 迭代次数
    M = 0          # 初始最小特征值
    minflis = []   # 最小值是特征时，特征的索引集合，迭代外
    Min = '0'      # 初始值，存放最小特征值，如'c2','s5','ftransitivity_local_undirected'
    MinL = []
    threshold = 0.05  # 设置最小值阈值
    chNum = 20        # 通道数
    subNum = 8        # 子带数
    nodelis = []      # 节点集合，初始化
    Result = {'t':[], 'C':[], 'acc':[],'pre':[],'sen':[], 'f1':[], 'kappa':[], 'c_r':[], 'del_im':[], 'im_val':[]}
    
    # 创键节点名
    for i in range(1, chNum+1):
        for j in range(1, subNum+1):
            noden = 'c' + str(i) + 's' + str(j)
            nodelis.append(noden) 
            
    # 开始迭代
    while(M < threshold):  
        start0 = datetime.datetime.now()
    
        # 得到所有epoch的对应脑网络的网络特征
        minflis = cal_bnf(MatSaveMI, MatSaveBnf, nodelis, minflis) 
        print('feature_total:',datetime.datetime.now()-start0)
        
        # 计算通道、子带和网络特征的重要性
        t0 = datetime.datetime.now()
        Imfeature = RFimportance(MatSaveBnf)    
        print('完成重要性计算',datetime.datetime.now() - t0)
        
        # 找到最小重要性及对应的节点
        t1 = datetime.datetime.now()
        # Min为最小节点名，M是最小重要性值 Ic,Is,If只是显示没用到  
        Min, M, Ic, Is, If = channelImportance(Imfeature) 
        MinL.append(Min)
        MinL.append(M)
        print('min:',Min)
        print('最小重要值：', M)
        print('最小重要性计算',datetime.datetime.now() - t1)
        
        # 删除重要性最小的节点
        t2 = datetime.datetime.now()
        if Min[0] == 'f':    # 最小值出现在特征重要性上
            minflis.append(Min[1:])  # 累积每次删除的特征名，防止下次迭代没删
        else:
            nodelis = removeNode(Min, nodelis, MatSaveMI)
            if os.path.exists(MatSaveMI_t):
                MatSaveMI = MatSaveMI_t
            else:
                os.mkdir(MatSaveMI_t)    
                MatSaveMI = MatSaveMI_t
            # MIfile_list = os.listdir(MatSaveMI)
        print('替换Matrix(MI)',datetime.datetime.now() - t2)
        
        # 加载数据
        t3 = datetime.datetime.now()
        X, y = load_data(MatSaveBnf)  # X是数据，y是标签
        
        # 训练
        result, model = train(X,y) 
        for j in result.keys():
            Result[j].append(result[j])
        Result['t'].append(t)
        Result['del_im'].append(Min)
        Result['im_val'].append(M)
        print('训练用时:', datetime.datetime.now()-t3)
        
        if M < threshold: # 留下最后一次迭代后的数据
            # 每次迭代清空一次网络特征，下次迭代重新生成
            feaFile = os.listdir(MatSaveBnf)  
            for m in feaFile:
                os.remove(MatSaveBnf + m)
        print('t:',t)
        t += 1
        print('一次迭代:',datetime.datetime.now()-start0)
    print('迭代总用时：',datetime.datetime.now()-start)
    return model
    


if __name__ == "__main__":

    start = datetime.datetime.now()
    print(start)      
    edfFileDir = 'E:/研究生/脑网络/tuh_eeg_seizure/v1.5.0'
    model = train_model(edfFileDir)

    print('总用时：',datetime.datetime.now() - start)

    # start = datetime.datetime.now()
    # print(start)    

    # # flag = 1 时, 训练癫痫分类模型
    # # flag = 1
    # # edfFileDir = 'E:/研究生/脑网络/tuh_eeg_seizure/v1.5.0'
    # # model = seizure_type_detection(edfFileDir, flag)
    
    # # flag = 2 时, 测试输入edf文件的癫痫发作类型 (这里测试和添加的数据都是以下这个edf文件，可以更改)
    # flag = 2
    # edfFileDir = 'E:/研究生/脑网络/tuh_eeg_seizure/v1.5.0/edf/train/01_tcp_ar/101/00010158/s003_2013_01_14/00010158_s003_t001.edf'    
    
    # outFilePath = '../'  # json文件存放位置，可更改。默认为上级目录
    # model = './model.pkl'  # 为flag = 1时训练得到的模型路径
    # X_test = seizure_type_detection(edfFileDir, flag, outFilePath, model)
  
    # print('总用时：',datetime.datetime.now() - start)


