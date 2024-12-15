# -*- coding: utf-8 -*-
from collections import namedtuple
import numpy as np
import os
import scipy.io as scio

def fread(fid, nelements, dtype):
    if dtype is np.str:
        dt = np.uint8  # WARNING: assuming 8-bit ASCII for np.str!
    else:
        dt = dtype

    data_array = np.fromfile(fid, dt, nelements)
    data_array.shape = (nelements, 1)

    return data_array

def readedfx(filename):
    '''
    edfx解析，修改时间：20190627，ccy修改
    该解析波形存为的mat文件和brainstrom的edf完全一致
    适用场景：24位采集获取的数据，所有通道采样率相同
    解析方法：
        第一块数据为有符号位的8位，将其移动16位，例如255为-1，254为-2
        第二块数据为无符号位的8位，将其移动8位
        第三块数据为无符号位的8位，不需要移动
        再将他们拼接
    :param filename:
    :return:
    '''
    fp = open(filename,'rb')
    hdr=fp.read(256)
    Header = namedtuple('Header',['length','records','duration','channels','channelname','transducer','physdime','physmin','physmax','digimin','digimax','prefilt','samplerate'])
    length = int(hdr[184:192])
    records = int(hdr[236:244])

    duration = float(hdr[244:252])
    if duration <=0.005 and duration >=0.002:
        duration = 0.002
    else:
        duration = 1

    channels = int(hdr[252:556])
    channelname = [fp.read(16).decode("utf-8") for i in range(channels)]
    transducer = [fp.read(80).decode("utf-8") for i in range(channels)]
    physdime = [fp.read(8).decode("utf-8") for i in range(channels)]
    physmin = [float(fp.read(8).decode("utf-8")) for i in range(channels)]
    physmax = [float(fp.read(8).decode("utf-8")) for i in range(channels)]
    digimin = [float(fp.read(8).decode("utf-8")) for i in range(channels)]
    digimax = [float(fp.read(8).decode("utf-8")) for i in range(channels)]
    prefilt = [fp.read(80).decode("utf-8") for i in range(channels)]
    samplerate = [int(fp.read(8)) for i in range(channels)]

    header = Header(length, records, duration, channels, channelname, transducer, physdime, physmin, physmax, digimin, digimax, prefilt, samplerate)
    fp.seek(0,0)
    kk=len(fp.read())-header.length
    fp.seek(header.length, 0)
    dt = np.dtype('uint8')
    data=fread(fp,kk,dt)
    fp.close()
    data=data.ravel()

    data=data[0:len(data)//3*3]
    data=data.reshape([kk//3,3])

    # 第一个块的最高位为符号位，第二块和第三块进行拼接
    data = np.where(data[:, 0]<128, data[:,0] * 65536|data[:, 1] *256 | data[:, 2].astype('int32'),data[:,0].astype('int8') * 65536| data[:, 1] *256 | data[:, 2].astype('int32')) / 12

    dataAllChannel = np.reshape(data,(kk//3//channels//samplerate[0], channels,-1))
    # 变为通道 * 数据
    dataAllChannel = np.reshape(np.transpose(dataAllChannel,(1,0,2)),(np.size(dataAllChannel,1),-1))

    # 实际的采样率
    UseChannels = np.sum(np.where(np.array(samplerate) == samplerate[0],1,0))
    samplerate = int(samplerate[0] / duration)
    header = Header(length, records, duration, channels, channelname, transducer, physdime, physmin, physmax, digimin,
                    digimax, prefilt, samplerate)
    # 增益
    bitvalue = (np.array(physmax) - np.array(physmin)) / (np.array(digimax) - np.array(digimin))
    dataAllChannel = dataAllChannel.astype('float32')
    for i in range(len(dataAllChannel)):
        if bitvalue[i] >0:
            offset = -float(digimax[i]) + float(physmax[i]) / bitvalue[i]
            dataAllChannel[i,:] = (dataAllChannel[i,:] + offset) * bitvalue[i]
        else:
            dataAllChannel[i,:] = dataAllChannel[i,:]

        if 'mv' in physdime[i].lower():
            dataAllChannel[i,:] = dataAllChannel[i,:] * 1000
        elif 'uv' in physdime[i].lower():
            dataAllChannel[i,:] = dataAllChannel[i,:]
        else:
            dataAllChannel[i,:] = dataAllChannel[i,:]* 1000000
    #scio.savemat('./data.mat',{'data':dataAllChannel[:UseChannels]})
    return header,dataAllChannel[:UseChannels]


def readedf(filename):
    '''
    该解析波形存为的mat文件和brainstrom的edf完全一致
    20190618 CCY
    :param filename:
    :return:
    '''
    fp=open(filename,'rb')
    hdr=fp.read(256)
    Header=namedtuple('Header',['length','records','duration','channels','channelname','transducer','physdime','physmin','physmax','digimin','digimax','prefilt','samplerate'])
    length=int(hdr[184:192])
    records=int(hdr[236:244])
    duration = float(hdr[244:252])
    channels=int(hdr[252:556])
    channelname=[fp.read(16).decode("utf-8") for i in range(channels)]
    transducer=[fp.read(80) for i in range(channels)]
    physdime=[fp.read(8).decode("utf-8") for i in range(channels)]
    physmin=[float(fp.read(8).decode("utf-8")) for i in range(channels)]
    physmax=[float(fp.read(8).decode("utf-8")) for i in range(channels)]
    digimin=[float(fp.read(8).decode("utf-8")) for i in range(channels)]
    digimax=[float(fp.read(8).decode("utf-8")) for i in range(channels)]
    prefilt=[fp.read(80).decode("utf-8") for i in range(channels)]
    samplerate=[int(fp.read(8)) for i in range(channels)]
    header=Header(length,records,duration,channels,channelname,transducer,physdime,physmin,physmax,digimin,digimax,prefilt,samplerate)
    fp.seek(0,0)
    kk=len(fp.read())-header.length
    fp.seek(header.length, 0)
    dt = np.dtype('int16')
    data=fread(fp,kk//2,dt)
    fp.close()

    data = data.ravel()

    # 避免采样率不同的情况
    # 原始data数据排布，一个数据块，每个数据块根据每一个通道采样率进行排布
    dataAllChannel = np.zeros((records,len(samplerate),samplerate[0]))
    blockSize = np.sum(samplerate)
    for i in range(records):
        for j in range(len(samplerate)):
            dataAllChannel[i,j,0:samplerate[j]] = data[int(i * blockSize + np.sum(samplerate[:j])) : int(i * blockSize + np.sum(samplerate[:j+1]))]

    # 变为通道 * 数据
    dataAllChannel = np.reshape(np.transpose(dataAllChannel,(1,0,2)),(np.size(dataAllChannel,1),-1))

    # 实际的采样率
    UseChannels = np.sum(np.where(np.array(samplerate) == samplerate[0],1,0))
    samplerate = int(samplerate[0] // duration)
    header = Header(length, records, duration, channels, channelname, transducer, physdime, physmin, physmax, digimin,
                    digimax, prefilt, samplerate)

    # 增益
    bitvalue = (np.array(physmax) - np.array(physmin)) / (np.array(digimax) - np.array(digimin))
    dataAllChannel = dataAllChannel.astype('float32')
    for i in range(len(dataAllChannel)):
        if bitvalue[i] >0:
            offset = -float(digimax[i]) + float(physmax[i]) / bitvalue[i]
            dataAllChannel[i,:] = (dataAllChannel[i,:] + offset) * bitvalue[i]
        else:
            dataAllChannel[i,:] = dataAllChannel[i,:]

        if 'mv' in physdime[i].lower():
            dataAllChannel[i,:] = dataAllChannel[i,:] * 1000
        elif 'uv' in physdime[i].lower():
            dataAllChannel[i,:] = dataAllChannel[i,:]
        else:
            dataAllChannel[i,:] = dataAllChannel[i,:] * 1000000
    return header,dataAllChannel[:UseChannels]


def readedforx(filename):
    if os.path.split(filename)[1].endswith('edf') or os.path.split(filename)[1].endswith('EDF'):
        header,data=readedf(filename)
    if os.path.split(filename)[1].endswith('edfx') or os.path.split(filename)[1].endswith('EDFX'):
        header,data=readedfx(filename)
    #data = data.astype(np.float16)
    return header,data

if __name__ =="__main__":
    '''
    filepath = r'E:\DataEDF\spike_detection_data3\after_medicine\EEG_6113_bandpass_zuochadaolian.edf'
    header, data = readedforx(filepath)
    '''

    filepath = r'E:\重症脑炎数据\已标样本\陈颖裕2018.12.6\chen1206.edf'
    header, data = readedforx(filepath)
    print()

