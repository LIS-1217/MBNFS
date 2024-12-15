# coding = utf-8
import pandas as pd
import os
import shutil
import scipy.io as scio

from Edf2MatCCY import Edf2Mat

def getExcel(fileDir):
    '''
    get excel seizure data
    :param fileDir:
    :return:
    '''
    dfTrain = pd.read_excel(fileDir,sheet_name='train')
    dfTrain = dfTrain.drop([0])

    # rename column name
    dfTrain.columns = ['Index','FileNum','Patient','Session','FileId','EEGType','EEGSubType','LTMorRoutine',
                       'NormalOrAbmormal','NumSeizure_File','NumSeizure_Session','TesFileDir','SeizureTimeStart','SeizureTimeEnd','SeizureType','Unnamed',
                       'SummaryEEGSubType1','SummarySessions1','SummaryFreq1','SummaryCum1','Unnamed1',
                       'SummaryEEGSubType2','SummarySessions2','SummaryFreq2','SummaryCum2','Unnamed2',
                       'SummaryEEGSubType3','SummarySessions3','SummaryFreq3','SummaryCum3'
                       ]

    dfTrainSeizure = dfTrain[dfTrain['SeizureTimeStart'].notnull()]
    return dfTrainSeizure


def getExceledfList():
    ExcelrootDir = '../tuh_eeg_seizure/v1.5.0/'
    excelDir = ExcelrootDir + '_DOCS/seizures_v32r.xlsx'
    dfTrainSeizure = getExcel(excelDir)
    
    edfFileList = {}
    try:
        for i in range(len(dfTrainSeizure.values)):
            tempDir = dfTrainSeizure.iloc[i]['TesFileDir'][2:]
            tseFileDir = ExcelrootDir + tempDir
            edfFileDir = tseFileDir.replace('.tse','.edf')
            index = edfFileDir.rfind('/')
            edfFileDir_1 = edfFileDir[index+1:]
            seizureTimeStart = dfTrainSeizure.iloc[i]['SeizureTimeStart']
            seizureTimeEnd = dfTrainSeizure.iloc[i]['SeizureTimeEnd']
            seizureType = dfTrainSeizure.iloc[i]['SeizureType']
            edfFileList[edfFileDir_1] = [seizureType, seizureTimeStart, seizureTimeEnd]
    except:
        pass
        
    return edfFileList


def getEDF(edfFileDir, seizureTimeStart, seizureTimeEnd):
    # header, data = readedforx(edfFileDir)
    data, freq = Edf2Mat("index_Noeye_region").read_data_from_edf(edfFileDir)
    dataSeizure = data[:,int(seizureTimeStart * freq):int(seizureTimeEnd * freq)]
    return freq, dataSeizure


def getEDFFile2Mat(ExcelrootDir, MatSaveRootDir):
    excelDir = os.path.join(ExcelrootDir,'_DOCS\seizures_v32r.xlsx')

    dfTrainSeizure = getExcel(excelDir)
    for i in range(len(dfTrainSeizure.values)):
        # get seizure epoch
        try:
            tempDir = dfTrainSeizure.iloc[i]['TesFileDir'][1:].replace('/','\\')
            tseFileDir = ExcelrootDir + tempDir
            seizureTimeStart = dfTrainSeizure.iloc[i]['SeizureTimeStart']
            seizureTimeEnd = dfTrainSeizure.iloc[i]['SeizureTimeEnd']
            seizureType = dfTrainSeizure.iloc[i]['SeizureType']
            # start analysis data, start get label info
            edfFileDir = tseFileDir.replace('.tse','.edf').replace('\\','/')
            # 只对01_tcp_ar进行处理
            if '01_tcp_ar' in edfFileDir:
                if os.path.exists(edfFileDir):
                    freq, dataSeizure = getEDF(edfFileDir, seizureTimeStart, seizureTimeEnd)
                    if os.path.exists(MatSaveRootDir + '/divided'):
                        scio.savemat(MatSaveRootDir + 'divided/' + str(i) + '.mat',{'dataSeizure':dataSeizure,'freq':freq,'seizureType':seizureType,'edfFileDir':edfFileDir})
                    else:
                        #创建该目录并保存mat文件
                        os.mkdir(MatSaveRootDir + '/divided')
                        scio.savemat(MatSaveRootDir +'divided/' +str(i) + '.mat',{'dataSeizure':dataSeizure,'freq':freq,'seizureType':seizureType,'edfFileDir':edfFileDir})
                else:
                    pass
        except:
            print(dfTrainSeizure.values[i])


#if __name__ == "__main__":
    #rootDir = r'E:\DataEDF\tuh_eeg_ccy\tuh_eeg_seizure\v1.5.0'
    #edfRootDir = r'G:\tuhData\tuh_eeg\v1.1.0'
#    ExcelRootDir = r'D:/癫痫分类数据TUH数据集/tuh_eeg_seizure/v1.5.0'
#    MatSaveRootDir = 'D:/癫痫分类/癫痫分类数据TUH数据集/tuh_eeg_seizure_classData/'
#    getEDFFile2Mat(ExcelRootDir, MatSaveRootDir)