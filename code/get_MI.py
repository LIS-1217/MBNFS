#coding:utf-8
import os
import numpy as np
import scipy
import scipy.io as scio
from WPD import signal_wpd
from calculate_MI import calc_MI
from scipy.signal import  butter, filtfilt
from getFileInfo import getEDFFile2Mat


def upper_triangular_to_symmetric(tri_Matrix):
    '''
    上三角矩阵转换为对称矩阵
    parameters:
        tri_Matrix: 上三角矩阵
    returns:
        tri_Matrix: 对称矩阵
    '''
    n = tri_Matrix.shape[0]
    for r in range(1, n):
        for c in range(r):
            tri_Matrix[r, c] = tri_Matrix[c, r]
    return tri_Matrix


def adjacency_Matrix_MI(subsignal_list):
    '''
    计算互信息矩阵
    parameters:
        subsignal_list: 需要计算互信息的子带信号列表
    returns:
        adjacency_Matrix: 互信息加权的邻接矩阵
    '''
    dim = len(subsignal_list)
    sub_Matrix = np.ones(dim)-np.eye(dim)
    sub_uptri = np.triu(sub_Matrix, k=0) # 提取上三角矩阵（两个子带信号的互信息只要计算一次，之后将上三角转为对称矩阵）
    for i in range(0,dim):
        for j in range(0,dim):
            if sub_uptri[i][j] != 0:
                # 4000个采样点的信号经过5级WPD后，子带信号采样点个数变为131个，这里互信息的计算是个估计值，结果与bins有关，可以适当修改
                sub_uptri[i][j] = sub_uptri[i][j]*calc_MI(subsignal_list[i], subsignal_list[j], bins=5)
    adjacency_Matrix = upper_triangular_to_symmetric(sub_uptri) # 将上三角矩阵转换为四方对称矩阵
    return adjacency_Matrix


def save_Adjacency_MatrixMI(epochdir, save_admatrix_dir):
    '''
    WPT处理得到子带信号，再计算并保存子带信号的互信息矩阵
    parameters:
        epochdir: epoch的.mat文件路径
        save_admatrix_dir: 保存互信息矩阵的路径
    returns:
    '''

    matFile0 = os.listdir(epochdir)
    for m in matFile0: 
        matdata0 = scio.loadmat(epochdir + m)
        epoch = matdata0['data']
        label0 = matdata0['label']  # 类型标签
        d = m.find('.')  # 例如: '1.mat'
        Cl = m[0:d]
        sub = []

        try:
            f = open(os.path.join(save_admatrix_dir +  Cl + '_adjacency_MatrixMI.mat'))
            f.close()
        except:
            for ch in range(20):
                sub_ch = signal_wpd(epoch[ch, :], maxlevel=4) # 4层
                sub.extend(sub_ch)
             
            sub_matrix = adjacency_Matrix_MI(sub)
            if os.path.exists(save_admatrix_dir):
                scipy.io.savemat(os.path.join(save_admatrix_dir + Cl + '_adjacency_MatrixMI.mat'),{'data':sub_matrix,'label':label0})
            else:
                os.mkdir(save_admatrix_dir)
                scipy.io.savemat(os.path.join(save_admatrix_dir + Cl + '_adjacency_MatrixMI.mat'),{'data':sub_matrix,'label':label0})


             
class butterFilter:
    '''
    巴特沃斯滤波器
    '''
    def butter_bandpass(self, lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='bandpass')
        return b, a

    def butter_bandpass_filter(self, data, lowcut, highcut, fs, order=5):
        b, a = self.butter_bandpass(lowcut, highcut, fs, order=order)
        y = filtfilt(b, a, data, axis=1)
        return y
    
    
def get_epoch_all(seizureFile, epochSave):
    '''
    切分发作片段为5s的小片段
    parameters:
        seizureFile: tuh数据集中所有发作片段的保存路径
        epochSave: 5s等长发作片段的保存路径
    return:
    '''
    matlist = os.listdir(seizureFile)
    Cl = 1 # epoch数
    for m in matlist:
        matData = scio.loadmat(seizureFile + m) 
        EpochLength = 5
        freq = matData['freq'][0][0]
        dataInit = matData['dataSeizure']
        seizureType = matData['seizureType'][0]
        # 先对各个Event降采样，统一为100Hz的信号
        if freq != 100:
            interval = freq // 100
            intervalNum = len(dataInit[0]) // interval * interval
            intervalList = np.linspace(0, intervalNum , len(dataInit[0]) // interval+1, dtype=int)
            dataInit = dataInit[:, intervalList[:-1]]
            freq = 100
        
        # 对脑电信号进行滤波，data的行:20
        data = butterFilter().butter_bandpass_filter(dataInit, lowcut=0.5, highcut=40, fs=freq, order=4)
        
        # 每一秒数据处理
        labelDict = {'ABSZ':1, 'CPSZ':2, 'FNSZ':3, 'GNSZ':4,'SPSZ':5,'TCSZ':6,'TNSZ':7}
        for i in range(np.size(data,1) // freq // EpochLength):  # np.size(data,1)是data的列数，循环每个片段
            if np.size(data,1) > freq * EpochLength:  # 至少有一个5s的片段
                D = {'data':data[:, i * freq * EpochLength : (i+1) * freq * EpochLength],'label':labelDict[seizureType]}
                if os.path.exists(epochSave):
                    scio.savemat(epochSave + str(Cl) + '.mat', D) 
                else:
                    os.mkdir(epochSave)
                    scio.savemat(epochSave + str(Cl) + '.mat', D) 
                Cl += 1


def get_MI_matrix(ExcelRootDir, SaveRootDir, epochdir, saveMIdir):
    '''
    parameters:
        ExcelRootDir: 脑电edf文件存放路径
        SaveRootDir: 发作片段保存根路径
        epochdir: 5s等长片段保存路径
        saveMIdir: MI矩阵保存路径
    return:
    '''
    # 得到tuh数据集中所有的发作片段，存在divided中
    getEDFFile2Mat(ExcelRootDir, SaveRootDir)  
    
    # # 得到所有的epoch（.mat文件）
    # seizureFile = SaveRootDir + 'divided/'
    # epochdir = '../epoch_all/'
    # get_epoch_all(seizureFile, epochdir)
    
    # 得到MI的（.mat文件）
    save_Adjacency_MatrixMI(epochdir, MatSaveMI)  # 生成733个MI矩阵
    

if __name__ == '__main__':

    # # 得到tuh数据集中所有的发作片段，存在divided中
    # ExcelRootDir = 'E:/研究生/脑网络/tuh_eeg_seizure/v1.5.0'
    # SaveRootDir = 'E:/研究生/脑网络/tuh_eeg_seizure_classData/'
    # getEDFFile2Mat(ExcelRootDir, SaveRootDir)  
    
    # # 处理tuh数据集，得到epoch的.mat文件
    # seizureFile = SaveRootDir + 'divided/'
    # epochSave = '../epoch_all/'
    # get_epoch_all(seizureFile, epochSave)
    
    # # 处理tuh数据集，得到MI的.mat文件
    # epochdir = '../epoch/'   # 用了733个epoch，没有用所有的epoch
    # MatSaveMI = '../MI/'     # 733个epoch生成733个MI矩阵
    # save_Adjacency_MatrixMI(epochdir, MatSaveMI)


    ExcelRootDir = 'E:/研究生/脑网络/tuh_eeg_seizure/v1.5.0'
    SaveRootDir = 'E:/研究生/脑网络/tuh_eeg_seizure_classData/'
    epochdir = '../epoch/'   # 用了733个epoch，没有用所有的epoch
    MatSaveMI = '../MI/'     # 733个MI矩阵存放路径
    get_MI_matrix(ExcelRootDir, SaveRootDir, epochdir, MatSaveMI)  # （只运行一次）生成733个MI矩阵
    
    
    
    
