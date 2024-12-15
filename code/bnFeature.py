# -*- coding: utf-8 -*-
from brainNetwork import brainigraph
import numpy as np
import pandas as pd
import os
import scipy.io as scio
from WPD import signal_wpd
from get_MI import butterFilter, adjacency_Matrix_MI, save_Adjacency_MatrixMI_t
import datetime
import warnings
warnings.filterwarnings('ignore')
import igraph as ig
from train import load_data_test


def bgFeature(G, matNum, label, de=None):
    '''
    parameters:
        G: 网络集，类型字典，键为网络名(字符型)，值为Graph
        matNum: 第 matNum 个 epoch 文件，即样本名
        label: 1-7，发作类型共7个类
        de: 删除的最小值特征，是列表，默认值为None
    return: 
        fG: 类型是字典，各个脑网络下所有的网络特征值    
        flabel: 脑网络提取的所有网络特征名
    '''
    fG = {}
    for i in G:
        
        fdic = {'label':[],'epoch':[], 'strength':[], 
                'transitivity_local_undirected':[],'harmonic_centrality':[],
                'betweenness':[],'closeness':[],'eigenvector_centrality':[],
                'evcent':[], 'pagerank':[], 'eccentricity':[]} 
        if de:
            for d in de:
                del fdic[d]
        
        vcount = G[i].vcount()   #统计节点数目
        p = list(map(abs, G[i].es['weight'])) 
        
        # label
        fdic['label'].extend([label for n in range(vcount)])
       
        # epoch
        fdic['epoch'].extend([matNum for n in range(vcount)])
        
        # 分别计算28个脑网络中的加权度、聚类系数、调和中心性、介数中心性、
        # 接近中心性、特征向量中心性、PageRank值和离心率。
        # 加权度
        if 'strength' in fdic:
            ad = G[i].strength(weights=G[i].es['weight']) # 计算所有节点的带权度值
            fdic['strength'].extend(ad)  

        # 聚类系数
        if 'transitivity_local_undirected' in fdic:
            clu = G[i].transitivity_local_undirected(weights=G[i].es['weight'])
            fdic['transitivity_local_undirected'].extend(clu) 

        # 调和中心性
        if 'harmonic_centrality' in fdic:
            hc = G[i].harmonic_centrality(weights=p)  # 参数必须为正值
            fdic['harmonic_centrality'].extend(hc) 
            
        # 介数中心性
        if 'betweenness' in fdic:
            be = G[i].betweenness(weights=p)  # 参数必须为正值
            fdic['betweenness'].extend(be) 
            
        # 接近中心性
        if 'closeness' in fdic:
            clo = G[i].closeness(weights=p)  # 参数必须为正值
            fdic['closeness'].extend(clo)  
            
        # 特征向量中心性
        if 'eigenvector_centrality' in fdic:
            ec = G[i].eigenvector_centrality(weights=G[i].es['weight'])
            fdic['eigenvector_centrality'].extend(ec)   
        
        # 特征向量中心性,特征向量中心性是网络邻接矩阵对应的最大特征值的特征向量。
        if 'evcent' in fdic:  
            ev = G[i].evcent(weights=G[i].es['weight']) # 计算图中节点的向量中心性
            fdic['evcent'].extend(ev) 
           
        # PageRank向量
        if 'pagerank' in fdic:  
            pa = G[i].pagerank(weights=G[i].es['weight']) # 计算节点的PageRank值
            fdic['pagerank'].extend(pa) 
           
        # 离心率
        if 'eccentricity' in fdic:  
            ecc = G[i].eccentricity() # 计算给定节点到图中其他节点的最短距离的最大值。
            fdic['eccentricity'].extend(ecc) 
            
        df = pd.DataFrame(fdic)  # 转换为DataFrame型
        flabel = np.array(list(df.columns))
        arr = np.array(df.values)    
        fG[i] = arr  # dict 28 , 每个网络(1,11)
        
    return fG, flabel
    


def cal_bnf(MatSaveMI, MatSaveBnf, nodelis, minflis):
    '''
    构建各个epoch的脑网络，每个epoch都有28个脑网络，每个脑网络都有9种网络特征，
    将所有epoch的对应脑网络的网络特征合并
    parameters:
        MatSaveMI: MI矩阵存放路径
        MatSaveBnf: 保存所有样本的各个网络特征值(.mat文件)的路径
        nodelis: 节点集
        minflis: 最小值是特征时，特征的索引集合
    return:
        minflis: 最小值是特征时，特征的索引集合
    '''
    
    if os.path.exists(MatSaveBnf):
        pass
    else:
        os.mkdir(MatSaveBnf)  
    
    MIfile_list = os.listdir(MatSaveMI)
    enum = 0
    fT = {}
    # 构建各个epoch的脑网络，每个epoch都有28个脑网络，每个脑网络都有9种网络特征。
    # 将所有epoch的对应脑网络的网络特征合并
    for m in MIfile_list:   # 遍历各个MI（.mat文件）
        matData = scio.loadmat(MatSaveMI + m)  
        Matrix = matData['data']  # MI矩阵
        label = matData['label']  # 类型标签
        d = m.find('_')           # m = '1112_adjacency_MatrixMI.mat'
        Cl = int(m[0:d])          # 第Cl个epoch，例如:1112
        G = brainigraph(nodelis, Matrix)   # 构建网络
        fG,flabel = bgFeature(G, Cl, label, minflis) # 每个.mat文件的所有脑网络的特征值
        
        # 存储网络特征
        if enum == 0:
            fT = fG
            enum += 1
        elif enum == len(MIfile_list)-1: 
            enum = 0
            for i in fG:
                fT[i] = np.concatenate((fT[i],fG[i]),axis=0)
                
                if os.path.exists(MatSaveBnf + str(i) + '.mat'):
                    bnfMat = scio.loadmat(MatSaveBnf + str(i) + '.mat')
                    bnfData = bnfMat['bnfdata']
                    data = np.concatenate((bnfData,fT[i]),axis=0)
                    fea = {'bnfdata':data,'bnflabel':flabel}
                    scio.savemat(MatSaveBnf + str(i) + '.mat', fea) 
                else:
                    fea = {'bnfdata':fT[i],'bnflabel':flabel}
                    scio.savemat(MatSaveBnf + str(i) + '.mat', fea) 
            fT = {}
            del fea  # 释放内存
        else:
            for i in fG:
                fT[i] = np.concatenate((fT[i],fG[i]),axis=0)
            enum += 1
            
        print(Cl,'done')
        
    return minflis




# 测试
if __name__ == "__main__":
    
    MatSaveMI = '../MI/'     # 733个MI矩阵存放路径
    # get_MI_matrix(edfFileDir, SaveRootDir, epochdir, MatSaveMI)  # 生成733个MI矩阵（在压缩包中的MI文件夹中）
    MatSaveMI_t = '../MI_temp/'    # 迭代时，删除节点后的MI矩阵存放地址
    MatSaveBnf = '../bnfeature/'   # 网络特征保存路径
    
    chNum = 20        # 通道数
    subNum = 8        # 子带数
    nodelis = []      # 节点集合，初始化
    minflis = []   # 最小值是特征时，特征的索引集合，迭代外
    # 创键节点名
    for i in range(1, chNum+1):
        for j in range(1, subNum+1):
            noden = 'c' + str(i) + 's' + str(j)
            nodelis.append(noden) 
            
    minflis = cal_bnf(MatSaveMI, MatSaveBnf, nodelis, minflis) 
    
#     t0 = datetime.datetime.now()
#     print(t0)
#     # for k in range(1,4):
#     # MatSavePCC = 'E:/研究生/脑网络/婴儿痉挛症数据/邻接矩阵PCC160/' # 0_adjacency_MatrixPCC.mat
#     # MatSaveMI = 'E:/研究生/脑网络/婴儿痉挛症数据/互信息矩阵MI160_3/互信息矩阵MI160_' + str(k) + '/' # 0_ictal_adjacency_MatrixMI.mat
#     MatSaveMI = 'E:/研究生/脑网络/tuh_eeg_seizure_classData/MI2/'  
#     matFile = os.listdir(MatSaveMI)
#     # MatSaveBnf = 'E:/研究生/脑网络/婴儿痉挛症数据/bnfeature_3/bnfeatureMI160_' + str(k) + '/' 
#     MatSaveBnf = 'E:/研究生/脑网络/tuh_eeg_seizure_classData/bnfeature2/'
#     chNum = 20      # 通道数
#     subNum = 8     # 子带数
#     nodelis = []    # 节点集合，初始化
#     for i in range(1, chNum+1):
#         for j in range(1, subNum+1):
#             noden = 'c' + str(i) + 's' + str(j)
#             nodelis.append(noden)       
    
#     # igraph库
#     enum = 0
#     fT = {}
#     for m in matFile:
#         matdata0 = scio.loadmat(MatSaveMI + m) # MatSaveMI + m
#         MI0 = matdata0['data']
#         label0 = matdata0['label']
#         d = m.find('_')   # 例如 m = '1112_adjacency_MatrixPCC.mat'
#         Cl = int(m[0:d])  # epoch
#         # Cl = 0
#         # for i in range(len(MI0)):  
#         #         MI0[i,:] = (MI0[i,:]>0).astype(np.int_)  # 二值化
#         G = brainigraph(nodelis, MI0)
#         fG,flabel = bgFeature(G, Cl, label0)
    
#         if enum == 0:
#             fT = fG.copy()
#             enum += 1
#         elif enum == len(matFile)-1:  # epoch数  
#             enum = 0
#             for i in fG:
#                 fT[i] = np.concatenate((fT[i],fG[i]),axis=0)
            
#                 if os.path.exists(MatSaveBnf + str(i) + '.mat'):
#                     bnfMat = scio.loadmat(MatSaveBnf + str(i) + '.mat')
#                     bnfData = bnfMat['bnfdata']
#                     data = np.concatenate((bnfData,fT[i]),axis=0)
#                     fea = {'bnfdata':data,'bnflabel':flabel}
#                     scio.savemat(MatSaveBnf + str(i) + '.mat', fea) 
#                 else:
#                     fea = {'bnfdata':fT[i],'bnflabel':flabel}
#                     scio.savemat(MatSaveBnf + str(i) + '.mat', fea) 
#             fT = {}
#         else:
#             for i in fG:
#                 fT[i] = np.concatenate((fT[i],fG[i]),axis=0)
#             enum += 1
        
#         print(Cl,'done')
    
#     print(datetime.datetime.now() - t0) 
   
   
