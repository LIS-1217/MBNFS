# -*- coding: utf-8 -*-
import scipy.io as scio
import os
import numpy as np
import igraph as ig


# 每5s一个epoch生成一个.mat文件，对每个.mat文件进行WPT生成子带
# 再将所有子带作为节点，每个节点之间的互信息作为边权重，来构建复杂网络
def brainigraph(nodelis, Matrix):
    '''
    nodelis : 节点集合，类型为列表
    Matrix : 一个epoch中所有节点的矩阵(PCC/MI)
    返回值 G : 一个epoch的网络集
    '''
    G0 = ig.Graph.Weighted_Adjacency(Matrix, mode='undirected') # 创建图形对象没有二值化
    G0.vs['name'] = nodelis
    
    cl = []
    sl = []
    # 找到通道名和子带名
    for i in nodelis:
        d = i.find('s')   
        c = i[0:d]   # 找到通道名，如'c5'
        s = i[d:]    # 找到子带名，如's3'
        cl.append(c)
        sl.append(s)
    cl = list(set(cl))  # 转换为集合再转换为列表，集合中不能存在重复的值
    sl = list(set(sl))
    
    G={}
    # 初始化
    for i in cl:  
        G[i] = 0
    for i in sl:  
        G[i] = 0
    
    # 构建通道网络    
    for i in cl:  # range(1,chNum+1)
        # n = 'c' + str(i)
        G[i] = G0.copy() # G[n] = G0.copy()
        to_delete_ids = [v.index for v in G0.vs if i in v['name']]  # if n in v['name']
        G[i].delete_vertices(to_delete_ids)  # G[n].delete_vertices(to_delete_ids)

    # 构建子带网络   
    for i in sl:  # range(1,subNum+1)
        # n = 's' + str(i)
        G[i] = G0.copy()
        to_delete_ids = [v.index for v in G0.vs if i in v['name']]
        G[i].delete_vertices(to_delete_ids)
        # adjacency = G.get_adjacency()    # 得到邻接矩阵

    return G


# # 测试
# if __name__ == "__main__":
#     chNum = 20      # 通道数
#     subNum = 8     # 子带数
#     nodelis = []    # 节点集合，初始化
#     for i in range(1, chNum+1):
#         for j in range(1, subNum+1):
#             noden = 'c' + str(i) + 's' + str(j)
#             nodelis.append(noden)       
#     # dellis = []  # 删除节点列表
#     # [dellis.append(i) for i in nodelis if 's5' in i]  # c2s
#     # [nodelis.remove(i) for i in dellis]
#     # # dellis = [] # 迭代时要清空
#     MatSavePCC = 'E:/研究生/脑网络/tuh_eeg_seizure_classData/MI1/1_adjacency_MatrixMI.mat'
#     matdata0 = scio.loadmat(MatSavePCC)
#     label0 = matdata0['label']
#     MI0 = matdata0['data']
#     # for i in range(len(PCC0)):  
#     #         PCC0[i,:] = (PCC0[i,:]>0).astype(np.int_)  # 二值化
#     G = brainigraph(nodelis, MI0)

         
        