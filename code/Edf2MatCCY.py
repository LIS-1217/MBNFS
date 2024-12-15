# -*- coding: utf-8 -*-
import pandas as pd
import scipy.io as sio
import numpy as np
import os
from readedfox import readedforx
import time as t

try:
    import cPickle as pickle
except:
    import pickle

class Edf2Mat:
    """
    Transform edf file to mat, one second saved in one mat file.
    for trainning, use gen_train_mat();
    for test, use gen_test_mat()
    """
    def __init__(self, index="index_Noeye_region"):
        index_Witheye = ['FP1-F7', 'F7-T3', 'T3-T5', 'T5-O1', 'FP2-F8', 'F8-T4', 'T4-T6', 'T6-O2', 'FP1-F3',
                         'F3-C3', 'C3-P3', 'P3-O1', 'FP2-F4', 'F4-C4', 'C4-P4',  'P4-O2', 'FZ-CZ', 'CZ-PZ']
        index_Noeye = ['F7-T3', 'T3-T5', 'T5-O1', 'F8-T4', 'T4-T6', 'T6-O2', 'F3-C3',
                       'C3-P3', 'P3-O1', 'F4-C4', 'C4-P4', 'P4-O2', 'FZ-CZ', 'CZ-PZ']
        index_Noeye_region = ['F7-T3', 'T3-T5', 'T5-O1', 'F8-T4', 'T4-T6', 'T6-O2', 'F3-C3',
                              'C3-P3', 'P3-O1', 'F4-C4', 'C4-P4', 'P4-O2', 'FZ-CZ', 'CZ-PZ',
                              "F7-F3", "T3-C3", "T5-P3", "F4-F8", "C4-T4", "P4-T6"]

        self.channels_index_dic = {"index_Witheye": index_Witheye,
                              "index_Noeye": index_Noeye,
                              "index_Noeye_region": index_Noeye_region}
        self.channels_index = self.channels_index_dic[index]
        self.replace = {'T7': 'T3', 'T8': 'T4', 'P7': 'T5', 'P8': 'T6'}

    def std_format(self, channels_name_list, data_of_arrays):
        channels_num = len(channels_name_list)
        new = np.zeros((len(self.channels_index), data_of_arrays.shape[1]))
        for v in range(len(self.channels_index)):
            d = []
            splited = self.channels_index[v].split('-')
            for i in splited:
                for j in range(channels_num):
                    if i in str.upper(channels_name_list[j]):
                        d.append(j)
                        break
            new[v, :] = data_of_arrays[d[0], :] - data_of_arrays[d[1], :]
        return new

    def read_data_from_edf(self, full_path):
        """read data and frequency from edf"""
        header_tuple, data_of_arrays = readedforx(full_path)
        print(header_tuple.samplerate)
        channels_name_list = [i.strip() for i in header_tuple.channelname]
        freq = header_tuple.samplerate
        if isinstance(freq, list):
            freq = int(freq[0])

        data = self.std_format(channels_name_list, data_of_arrays)
        return data, freq

if __name__ == '__main__':
    Edf2Mat("index_Noeye_region").read_data_from_edf(r'E:\重症脑炎数据\已标样本\陈颖裕2018.12.6\chen1206.edf')