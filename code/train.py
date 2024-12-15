# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc, r2_score  ###计算roc和auc
from sklearn.model_selection import train_test_split, GridSearchCV
from imblearn.combine import SMOTEENN, SMOTETomek   # 下采样与过采样结合
from imblearn.over_sampling import SMOTE
import xlwings as xw
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import seaborn as sn
import datetime
import os
import scipy.io as scio
import pandas as pd
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score,f1_score,cohen_kappa_score,classification_report,roc_curve
from sklearn.model_selection import KFold
from trainProcrc import train
import warnings
warnings.filterwarnings('ignore')



def normalization(data):
    """归一化数据"""
    # print(np.max(data, axis=0))
    # print(np.min(data, axis=0))
    _range = np.max(data, axis=0) - np.min(data, axis=0)
    return (data - np.min(data, axis=0)) / _range


def standardization(data):
    """标准化数据"""
    # print(np.mean(data, axis=0))
    # print(np.std(data, axis=0))
    return (data - np.mean(data, axis=0)) / np.std(data, axis=0)



def save_result(X_test, y_test, clf):
    """此函数利用混淆矩阵来计算测试集上的准确率，精确率，召回率和特异性，F1分数以及roc曲线的面积，并画出roc曲线图"""
    tn_test, fp_test, fn_test, tp_test = confusion_matrix(y_test, clf.predict(X_test)).ravel()
    tt_test = tn_test + fp_test + fn_test + tp_test
    p0 = (tp_test + tn_test)/tt_test
    pe = ((tp_test+fp_test)*(tp_test+fn_test)+(fn_test+tn_test)*(fp_test+tn_test))/tt_test/tt_test
    zhun = (tp_test + tn_test) / tt_test
    jing = tp_test / (tp_test + fp_test)
    zhao = tp_test / (tp_test + fn_test)
    teyi = tn_test / (fp_test + tn_test)
    F1 = 2 * jing * zhao / (jing + zhao)
    kappa = (p0-pe)/(1-pe)
    # print(tp_test, fp_test)
    # print(fn_test, tn_test)

    # Compute ROC curve and ROC area for each class
    y_score = clf.predict(X_test)
    fpr, tpr, threshold = roc_curve(y_test, y_score)  ###计算真正率和假正率
    roc_auc = auc(fpr, tpr)  ###计算auc的值
    result = [{'tp': np.float(tp_test),
               'fp': np.float(fp_test),
               'fn': np.float(fn_test),
               'tn': np.float(tn_test),
               'accuracy': zhun,
               'precision': jing,
               'recall/sensitivity/TPR': zhao,
               'specificity': teyi,
               'F1 score': F1,
               'ROC curve': roc_auc,
               'Kappa': kappa}]

    return result



def load_data(MatSaveBnf):
    '''
    MatSaveBnf : 保存所有epoch的所有网络特征值(.mat文件)的路径 , =====先初始化=====
    '''
    matbnf = os.listdir(MatSaveBnf)
    bnf = scio.loadmat(MatSaveBnf + matbnf[0])
    bnfData = bnf['bnfdata']
    bnfData = bnfData.astype(float)  # 将复数转换为实数
    epochN = bnfData[:,1]  # 第二列是epoch数
    epochN = list(set(epochN))
    Label = []
    T = 0     # 取epoch的label (是否发作或者发作类型)，只用在一个网络.mat文件中取
    # e = epochN[0]
    Data0 = {}
    for i in epochN:   # 初始化
        Data0[i] = []


    for i in matbnf:
        bnf = scio.loadmat(MatSaveBnf + i)
        bnfData = bnf['bnfdata']
        bnfData = bnfData.astype(float)  # 将复数转换为实数
        bnfLabel = bnf['bnflabel']
        f = pd.DataFrame(bnfData,columns=bnfLabel)
        for e in epochN:
            f1 = f[f.iloc[:,1] == e]    # 取f的'epoch'列值为e的行，f1类型为 Dataframe
            f2 = f1.iloc[:, 2:].values  # 取除了label和epoch，剩下特征的值，类型array
            f2 = f2.ravel(order = 'F')  # 展平数组为1维,类型array
            Data0[e].extend(f2)
            if T == 0:   
                L = f1.iloc[0,0]  # 取label (是否发作或者发作类型)的值，婴儿痉挛症:0或1，tuh:1~6
                Label.append(L)
        T = 1
        # print(i,'done')
    Label = np.array(Label)
    Data = []
    for i in Data0:
        Data.append(Data0[i])
    Data = np.array(Data)
    
    return Data,Label




# 测试
if __name__ == "__main__":
    
    start = datetime.datetime.now()
    print(start)
    Data = []
    Label = []
    excelSave = 'E:/研究生/脑网络/婴儿痉挛症数据/save_result_4_0.xlsx'
    Result = {'t':[],'ACC':[], 'PRE':[], 'SEN':[], 'F1':[], 'TP':[], 
              'FP':[], 'FN':[], 'TN':[], 'del_im':[], 'im_val':[]}  # 每次迭代完的指标结果(初始化)
    for i in range(5): #指标结果(初始化)
        Result['%d_TN'%i] = []
        Result['%d_FP'%i] = []
        Result['%d_FN'%i] = []
        Result['%d_TP'%i] = []
        Result['%d_ACC'%i] = []
        Result['%d_PRE'%i] = []
        Result['%d_SEN'%i] = []
        Result['%d_F1'%i] = []
        
    
    
    
       