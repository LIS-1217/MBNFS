#coding:utf-8
import networkx as nx
import numpy as np
import pywt
import scipy.io as scio
from matplotlib import pyplot as plt


# maxlevel 取4
def signal_wpd(singal, maxlevel):
    wp = pywt.WaveletPacket(singal, wavelet='db4', maxlevel=maxlevel, mode='symmetric')
    re = []  # 第n层所有节点的分解系数
    nodep = [node.path for node in wp.get_level(maxlevel,'freq')]  # 第 maxlevel 层小波包节点
    nodep1 = nodep[0:8] # 最后一层前8个节点
    # for i in [node.path for node in wp.get_level(maxlevel)]:
    for i in nodep1:
        re.append(wp[i].data)
    
    return re

