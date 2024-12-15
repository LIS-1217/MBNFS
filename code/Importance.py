# -*- coding: utf-8 -*-
import numpy as np
from brainNetwork import brainigraph
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from bnFeature import bgFeature
import pandas as pd
import scipy.io as scio
import os
import datetime
import warnings
warnings.filterwarnings('ignore')


def removeNode(m, nodelis, MatSaveRMatrix):
    '''
    删除节点
    parameters:
        m: 所有通道、子带和特征重要性中最小的一个，字符型，如c2,s5,fAverage_neighbor_degree
        nodelis: 节点集合，类型列表
        MatSaveRMatrix: MI路径，第一次迭代时路径
    return:
        nodelis: 若最小值在通道或子带上则返回nodelis(节点集合)，否则返回特征名(最小)
    '''
    MatSaveMI_t = '../MI_temp/'    # 迭代时，删除节点后的MI矩阵存放地址
    if os.path.exists(MatSaveMI_t):
        pass
    else:
        os.mkdir(MatSaveMI_t) 
        
    matFile = os.listdir(MatSaveRMatrix)
    dellis = []  # 删除节点列表
    pos = []     # 删除节点的索引
    if m[0] == 'c':
        n = m +'s'   # c2s，多加个s防止出现'c2'，把'c20'也都去除了
    else:            # m[0] == 's'. 等于'f'的情况已经写过了
        n = m
        
    for p,i in enumerate(nodelis):
        if n in i:  
            dellis.append(i)
            pos.append(p)
    [nodelis.remove(i) for i in dellis]
    for m in matFile:
        matData = scio.loadmat(MatSaveRMatrix + m)  # 字典
        Matrix = matData['data']   # size(640,640) 
        label = matData['label']
        Matrix0 = np.delete(Matrix, pos, axis=0)    # 删除行
        Matrix1 = np.delete(Matrix0, pos, axis=1)   # 删除列
        Matrix = {'data':Matrix1, 'label':label} 
        scio.savemat(MatSaveMI_t + m, Matrix) 
        
    return nodelis



def channelImportance(Imfeature):
    '''
    求通道重要性、子带重要性和特征重要性的最小值位置
    parameters:
        Imfeature : 特征通过RF得到的网络重要性，为字典
    return:
        m是重要性最小值位置，类型是字符型，共2位。
        第一位表示在通道(c)、子带(s)或特征(f)中的哪个，第二位表示在第几个
        例如：c2。表示最小值是第3个通道
        Mi0为重要性最小值
    '''
    Ic = []  # 通道重要性  
    Is = []  # 子带重要性
    If = {}  # 特征重要性
    v = [i for i in Imfeature.values()] 
    for i in list(v[0].keys()):  # i为特征名
        If[i] = 0  # 初始化
        
    for i in Imfeature:  # 遍历所有网络
        for j in list(v[0].keys()):    # 特征名
            If[j] += Imfeature[i][j]   # 将所有网络的该特征相加
    
    # 去除通道各个网络的重要性，注意Ifc的shape是1 x chNum，计算通道重要性时要转置即Ifc.T
    Ifc = []
    chNum = 0
    chnlis = []
    for i in Imfeature:   # 如i为'c13.mat','s5.mat'...  # i为'c13','s5'....
        if 'c' in i:
            Ifc.append(sum(list(Imfeature[i].values())))
            chNum += 1
            # d = i.find('.')
            # chnlis.append(i[0:d])
            chnlis.append(i)
    Ifc = np.array(Ifc) 
    
    # 去除子带各个网络的重要性
    Ifs = []
    subNum = 0
    subnlis = []
    for i in Imfeature:   
        if 's' in i:
            Ifs.append(sum(list(Imfeature[i].values())))
            subNum += 1
            # d = i.find('.')
            # subnlis.append(i[0:d])
            subnlis.append(i)
    Ifs = np.array(Ifs) 
    
    Ec = np.ones((chNum, chNum), dtype=int)  # Ec是chNum*chNum的全1矩阵，主对角线上全为0
    for i in range(chNum):
        Ec[i][i] = 0
        
    Es = np.ones((subNum, subNum), dtype=int) # Es是subNum*subNum的全1矩阵，主对角线上全为0
    for i in range(subNum):
        Es[i][i] = 0
    
    # Ec*Ic = Ifc，求Ec的逆矩阵，即可求得Ic = Ec^(-1) * Ifc
    try:
        Ec1 = np.linalg.inv(np.array(Ec))      # 计算给定矩阵的逆矩阵
        # print(np.linalg.inv(np.array(Ec)))  
    except:
        print("Ec Matrix, Inverse not possible.")
    # Ic = Ec^(-1) * Ifc.T
    Ic = Ec1 @ Ifc.T
        
    # Es*Is = Ifs，求Es的逆矩阵，即可求得Is = Es^(-1) * Ifs
    try:
        Es1 = np.linalg.inv(np.array(Es))       # 计算给定矩阵的逆矩阵
        # print(np.linalg.inv(np.array(Es)))  
    except:
        print("Es Matrix, Inverse not possible.")
    # Is = Es^(-1) * Ifs.T
    Is = Es1 @ Ifs.T
    
    MIc = np.argmin(Ic)  # 最小值位置索引
    Icn = chnlis[MIc]    # 最小通道名 如'c13'
    MIs = np.argmin(Is)
    Isn = subnlis[MIs]   # 最小子带名 如's5'
    # 从小到大排列，是列表，每项为元组,MIf[0]=(0.0, 'Average_neighbor_degree')
    MIf = sorted(zip(If.values(), If.keys())) 
    
    TIc = np.sum(Ifc)   # 前chNum个网络的重要性之和
    TIs = np.sum(Ifs)   # chNum到(chNum+subNum)个网络的重要性之和
    
    Mi0 = min(Ic[MIc]/TIc, Is[MIs]/TIs, MIf[0][0])
    if Mi0 == Ic[MIc]/TIc:
        m = Icn    # 如'c13'
    elif Mi0 == Is[MIs]/TIs:
        m = Isn    # 如's5'
    else: 
        m = 'f' + MIf[0][1]
    
    return m, Mi0, Ic, Is, If
   
    
def RFimportance(MatSaveBnf):
    '''
    计算通道、子带和网络特征的重要性
    parameters:
        MatSaveBnf: 保存所有epoch的所有网络特征值(.mat文件)的路径 
    return: 
       Imfeature: 各个网络的特征重要性，是字典，键为网络名(字符型)，值为字典(键为特证名，值为特征重要性值)。
    '''
    
    matFile = os.listdir(MatSaveBnf)
    Imfeature = {}
    for m in matFile:
        bnfFile = scio.loadmat(MatSaveBnf + m) # 读取各个网络特征值(.mat文件)
        bnfData = bnfFile['bnfdata']
        bnfData = bnfData.astype(float)  # 将复数转换为实数
        bnfLabel = bnfFile['bnflabel']
        bnfLabel = np.array([i.strip() for i in bnfLabel]) # 去除空格
        f = pd.DataFrame(bnfData,columns=bnfLabel) # 各个网络的特征值表，类型是列表，每项是DataFrame型
        x = f.iloc[:, 2:].values  # 取除了label和epoch，剩下特征的值
        y = f.iloc[:, 1].values   # 取epoch的值
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 0)
        feat_labels = f.columns[2:]  # 特证名
        forest = RandomForestClassifier(n_estimators=500,random_state=0, n_jobs=1,max_depth=3) 
        forest.fit(x_train, y_train)
        # score = forest.score(x_test, y_test)
        importances = forest.feature_importances_
        indices = np.argsort(importances)[::-1] # 下标排序
        Imdic = {}
        for j in range(x_train.shape[1]):   # x_train.shape[1]=11
            Imdic[feat_labels[indices[j]]] = importances[indices[j]]
        
        m0 = m.find('.')
        m1 = m[0:m0]
        Imfeature[m1] = Imdic
    
    return Imfeature
    
   
# # 测试    
# if __name__ == "__main__":
    
#     start = datetime.datetime.now()
#     print(start)

#     MatSaveBnfMI = 'E:/研究生/脑网络/婴儿痉挛症数据/bnfeature/bnfeatureMI160_1/' 
#     MatSaveMI = 'E:/研究生/脑网络/婴儿痉挛症数据/互信息矩阵MI160/互信息矩阵MI160_1/'
#     MatSaveMI1 = 'E:/研究生/脑网络/婴儿痉挛症数据/MI/MI_1/'
    
#     MatSaveMatrix = MatSaveMI  # MatSaveMI / MatSavePCC  更改，记得清空 1
#     matFile = os.listdir(MatSaveMatrix)
#     MatSaveBnf = MatSaveBnfMI  # MatSaveBnfMI / MatSaveBnfPCC  更改，记得清空 2
#     t = 0          # 迭代次数
#     M = 0          # 初始最小特征值
#     minflis = []   # 最小值是特征时，特征的索引集合，迭代外
#     Min = '0'      # 初始值，存放最小特征值，如'c2','s5','ftransitivity_local_undirected'
#     MinL = []
#     threshold = 0.004  # 设置最小值阈值
#     chNum = 20         # 通道数
#     subNum = 8         # 子带数
#     nodelis = []       # 节点集合，初始化
#     for i in range(1, chNum+1):
#         for j in range(1, subNum+1):
#             noden = 'c' + str(i) + 's' + str(j)
#             nodelis.append(noden) 
            
#     while(M < threshold):  # t < tNum
#         start0 = datetime.datetime.now()
#         enum = 0
#         fT = {}
#         for m in matFile:    # 遍历.mat文件
#             matData = scio.loadmat(MatSaveMatrix + m)  # 字典
#             Matrix = matData['data']  # PCC或MI  类型Array
#             label = matData['label']
#             d = m.find('_')  # m = '1112_adjacency_MatrixPCC.mat'
#             Cl = int(m[0:d])  # epoch
#             # for i in range(len(Matrix)):  
#             #     Matrix[i,:] = (Matrix[i,:]>0).astype(np.int_)  # 二值化
#             G = brainigraph(nodelis, Matrix) # 构建网络
#             fG,flabel = bgFeature(G, Cl, MatSaveBnf, label, minflis) # 每个.mat文件的所有脑网络的特征值
              
#             # 存储网络特征为.mat文件
#             if enum == 0:
#                 fT = fG
#                 enum += 1
#             elif enum == len(matFile)-1:
#                 enum = 0
#                 for i in fG:
#                     fT[i] = np.concatenate((fT[i],fG[i]),axis=0)
                
#                     if os.path.exists(MatSaveBnf + str(i) + '.mat'):
#                         bnfMat = scio.loadmat(MatSaveBnf + str(i) + '.mat')
#                         bnfData = bnfMat['bnfdata']
#                         data = np.concatenate((bnfData,fT[i]),axis=0)
#                         fea = {'bnfdata':data,'bnflabel':flabel}
#                         scio.savemat(MatSaveBnf + str(i) + '.mat', fea) 
#                     else:
#                         fea = {'bnfdata':fT[i],'bnflabel':flabel}
#                         scio.savemat(MatSaveBnf + str(i) + '.mat', fea) 
#                 fT = {}
#                 del fea  # 释放内存
#             else:
#                 for i in fG:
#                     fT[i] = np.concatenate((fT[i],fG[i]),axis=0)
#                 enum += 1
#             # print(Cl,'done') 
#         print('feature_total:',datetime.datetime.now()-start0)
        
#         t0 = datetime.datetime.now()
#         Imfeature = RFimportance(MatSaveBnf) 
#         print('完成重要性计算',datetime.datetime.now() - t0)
        
#         t1 = datetime.datetime.now()
#         Min,M,Ic,Is,If = channelImportance(Imfeature) # Min为最小维度名，M是最小贡献度值 Ic,Is,If只是显示没用到
#         MinL.append(Min)
#         MinL.append(M)
#         print('min:',Min)
#         print('最小重要性计算',datetime.datetime.now() - t1)
        
#         t2 = datetime.datetime.now()
#         if Min[0] == 'f':    # 最小值出现在特征重要性上
#             minflis.append(Min[1:])  # 累积每次删除的特征名，防止下次迭代没删
#         else:
#             nodelis = removeNode(Min, nodelis, MatSaveMatrix)  # 可以用多进程
#             MatSaveMatrix = MatSaveMI1
#             matFile = os.listdir(MatSaveMatrix)
#         print('替换Matrix(MI)',datetime.datetime.now() - t2)
        
#         if M < threshold: # 留下最后一次迭代后的数据
#             # 每次迭代清空一次网络特征，下次迭代重新生成
#             feaFile = os.listdir(MatSaveBnf)  
#             for m in feaFile:
#                 os.remove(MatSaveBnf + m)
#         print('t:',t)
#         t += 1
#         print('一次迭代:',datetime.datetime.now()-start0)
#     print('迭代总用时：',datetime.datetime.now()-start)


# Imfeature = {'c10': {'closeness': 0.36465466281425707, 'harmonic_centrality': 0.27978681574514963, 'strength': 0.11149999684543782, 'betweenness': 0.08530309565861216, 'evcent': 0.062290227982659686, 'eigenvector_centrality': 0.060264352208021135, 'pagerank': 0.03620084874586258}, 'c11': {'closeness': 0.3676948945290289, 'harmonic_centrality': 0.27404046547073524, 'strength': 0.11546669774578082, 'betweenness': 0.08292535922962053, 'evcent': 0.06276367268578775, 'eigenvector_centrality': 0.061575651329991765, 'pagerank': 0.03553325900905497}, 'c12': {'closeness': 0.36547499477257517, 'harmonic_centrality': 0.27189802650376343, 'strength': 0.12439282737960383, 'betweenness': 0.08592671199396289, 'evcent': 0.060483801337155946, 'eigenvector_centrality': 0.05939317799696021, 'pagerank': 0.03243046001597857}, 'c13': {'closeness': 0.3617278530304047, 'harmonic_centrality': 0.27270667238480195, 'strength': 0.11235648422896848, 'betweenness': 0.08504160819513544, 'evcent': 0.06773990067695242, 'eigenvector_centrality': 0.0630140515669702, 'pagerank': 0.037413429916766876}, 'c14': {'closeness': 0.34359117369608516, 'harmonic_centrality': 0.2674767408240811, 'strength': 0.12026935830194302, 'betweenness': 0.09292270533908693, 'evcent': 0.06815303804437797, 'eigenvector_centrality': 0.0681515495079907, 'pagerank': 0.03943543428643515}, 'c15': {'closeness': 0.3706231898989735, 'harmonic_centrality': 0.28269213853685143, 'strength': 0.1056814250966036, 'betweenness': 0.08781840910618705, 'evcent': 0.061708586272461396, 'eigenvector_centrality': 0.059495971294553696, 'pagerank': 0.03198027979436928}, 'c16': {'closeness': 0.37261272572957405, 'harmonic_centrality': 0.2809461369541821, 'strength': 0.11170770426955444, 'betweenness': 0.08821737881262698, 'evcent': 0.05849107175815921, 'eigenvector_centrality': 0.05588287902471865, 'pagerank': 0.03214210345118451}, 'c17': {'closeness': 0.3605194486489394, 'harmonic_centrality': 0.2732735143772149, 'strength': 0.12601144034162687, 'betweenness': 0.09013156193684356, 'evcent': 0.061157768593052546, 'eigenvector_centrality': 0.05613799111128446, 'pagerank': 0.03276827499103819}, 'c18': {'closeness': 0.35575620151014214, 'harmonic_centrality': 0.258609519192033, 'strength': 0.11000034991481973, 'betweenness': 0.10694992750359163, 'evcent': 0.06593580488018232, 'eigenvector_centrality': 0.06389455457373767, 'pagerank': 0.03885364242549336}, 'c19': {'closeness': 0.37327408493710984, 'harmonic_centrality': 0.2805695087982237, 'strength': 0.10628551061805767, 'betweenness': 0.08773361016920989, 'evcent': 0.06216208217936143, 'eigenvector_centrality': 0.05649453587760753, 'pagerank': 0.033480667420430016}, 'c2': {'closeness': 0.37431774391555506, 'harmonic_centrality': 0.27821942260133564, 'strength': 0.11761047045262596, 'betweenness': 0.08311119962983751, 'eigenvector_centrality': 0.056920044726837496, 'evcent': 0.05614543345238756, 'pagerank': 0.03367568522142075}, 'c20': {'closeness': 0.36163798753424464, 'harmonic_centrality': 0.27625801518515675, 'strength': 0.11601389826854946, 'betweenness': 0.0908416306951691, 'evcent': 0.06202779910289084, 'eigenvector_centrality': 0.05783659738983791, 'pagerank': 0.03538407182415128}, 'c3': {'closeness': 0.3642441715018686, 'harmonic_centrality': 0.2733626618711998, 'strength': 0.1294768943738152, 'betweenness': 0.08406136870703491, 'evcent': 0.06033669920331187, 'eigenvector_centrality': 0.055943544895251524, 'pagerank': 0.03257465944751806}, 'c4': {'closeness': 0.36302089580575614, 'harmonic_centrality': 0.276076327177917, 'strength': 0.12350398772804955, 'betweenness': 0.08164857770498607, 'evcent': 0.06118677100341091, 'eigenvector_centrality': 0.05629486357841047, 'pagerank': 0.03826857700146987}, 'c5': {'closeness': 0.3784020879101236, 'harmonic_centrality': 0.2777937441792374, 'strength': 0.10845842239677123, 'betweenness': 0.08240302025483852, 'evcent': 0.059876838447750186, 'eigenvector_centrality': 0.05659286276032288, 'pagerank': 0.03647302405095614}, 'c6': {'closeness': 0.36360013742891856, 'harmonic_centrality': 0.27367218825150647, 'strength': 0.11874841396615186, 'betweenness': 0.08382327081838206, 'evcent': 0.06277460955231189, 'eigenvector_centrality': 0.058923514947765945, 'pagerank': 0.03845786503496316}, 'c7': {'closeness': 0.36416859542388424, 'harmonic_centrality': 0.2762485888458239, 'strength': 0.11133691982298227, 'betweenness': 0.08448719041910194, 'evcent': 0.06671204558134476, 'eigenvector_centrality': 0.062002777552404645, 'pagerank': 0.03504388235445818}, 'c8': {'closeness': 0.3660457733500981, 'harmonic_centrality': 0.27706239964637447, 'strength': 0.1150793627735306, 'betweenness': 0.08515679060208103, 'evcent': 0.06290296073245007, 'eigenvector_centrality': 0.05702533589723301, 'pagerank': 0.03672737699823291}, 'c9': {'closeness': 0.366702014874986, 'harmonic_centrality': 0.27790296038993284, 'strength': 0.11856649436675605, 'betweenness': 0.08696220183197959, 'evcent': 0.05833245955302165, 'eigenvector_centrality': 0.05651103553866335, 'pagerank': 0.035022833444660645}, 's1': {'closeness': 0.35607224046149055, 'harmonic_centrality': 0.26559821263677486, 'strength': 0.1383711639513758, 'betweenness': 0.07744249391620849, 'evcent': 0.06659623752504873, 'eigenvector_centrality': 0.06311464321983512, 'pagerank': 0.03280500828926657}, 's2': {'closeness': 0.38019787437193253, 'harmonic_centrality': 0.278295166671446, 'strength': 0.10605177143407124, 'betweenness': 0.0845434823128312, 'evcent': 0.05998000547864709, 'eigenvector_centrality': 0.05670139305985694, 'pagerank': 0.034230306671214884}, 's3': {'closeness': 0.35734701926946927, 'harmonic_centrality': 0.2728915675456829, 'strength': 0.1238817453371233, 'betweenness': 0.08249588237379231, 'evcent': 0.06270210817106439, 'eigenvector_centrality': 0.06012734732141493, 'pagerank': 0.040554329981453}, 's4': {'closeness': 0.3617011181579327, 'harmonic_centrality': 0.27933840407931176, 'strength': 0.12240846651029649, 'betweenness': 0.08192502383139379, 'evcent': 0.06221618836152225, 'eigenvector_centrality': 0.05843959232408548, 'pagerank': 0.033971206735457414}, 's5': {'closeness': 0.33957541409446984, 'harmonic_centrality': 0.25883011094054276, 'strength': 0.11179664594234325, 'betweenness': 0.09904375501214763, 'eigenvector_centrality': 0.07705289401631775, 'evcent': 0.07591020104418435, 'pagerank': 0.0377909789499944}, 's6': {'closeness': 0.3722476942634328, 'harmonic_centrality': 0.289606457220089, 'strength': 0.10815788828375233, 'betweenness': 0.08385597100070302, 'evcent': 0.058969459240706276, 'eigenvector_centrality': 0.05570182240137723, 'pagerank': 0.031460707589939324}, 's7': {'closeness': 0.36931895265018716, 'harmonic_centrality': 0.28390881595002, 'strength': 0.11178991459968819, 'betweenness': 0.0800740579104606, 'evcent': 0.060894469534141425, 'eigenvector_centrality': 0.05875009676874547, 'pagerank': 0.03526369258675706}, 's8': {'closeness': 0.3641872737072941, 'harmonic_centrality': 0.2759052356766321, 'strength': 0.11888524623452332, 'betweenness': 0.08512083030049958, 'eigenvector_centrality': 0.06329947254187372, 'evcent': 0.06233782778721905, 'pagerank': 0.030264113751958133}}
# Min, M, Ic, Is, If= channelImportance(Imfeature) 

 