# -*- coding: utf-8 -*-
import numpy as np
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score,f1_score,cohen_kappa_score,classification_report,precision_recall_fscore_support
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import joblib
import json
from train import load_data


def calResult(y_test, procrcPreLabel):
    '''
    计算各个指标并保存
    parameters:
        y_test: 真实标签
        procrcPreLabel: 预测标签
    return:
        C, acc, pre, sen, f1, kappa, c_r: 
        分别为混淆矩阵、准确性、特异性、灵敏性、F1评分、kappa评分和分类结果
    '''
    print(accuracy_score(y_test, procrcPreLabel))
    print(cohen_kappa_score(y_test, procrcPreLabel))
    print(confusion_matrix(y_test, procrcPreLabel))         # 混淆矩阵
    print(classification_report(y_test, procrcPreLabel, digits=4))
    C = confusion_matrix(y_test, procrcPreLabel)
    acc = accuracy_score(y_test, procrcPreLabel)
    pre,sen,f1,_ = precision_recall_fscore_support(y_test, procrcPreLabel)
    kappa = cohen_kappa_score(y_test, procrcPreLabel)
    c_r = classification_report(y_test, procrcPreLabel, digits=4)
    
    result = {
            "accuracy_score" : str(acc),
            "precision_score" : str(np.mean(pre)),
            "recall_score" : str(np.mean(sen)),
            "f1_score" : str(np.mean(f1)),
            "kappa_score" : str(kappa)
            #"confusion_matrix" : str(C),
            # "classification_report" : c_r
        }

    # 将数据转换为JSON字符串并写入文件
    with open("result.json", "w") as file:
        json_str = json.dumps(result)
        file.write(json_str)
    file.close()
    
    return C, acc, pre, sen, f1, kappa, c_r

    
def train(featureData, featureLabel):
    '''
    训练模型并保存
    parameters:
        featureData: 数据
        featureLabel: 标签
    return:
        result: 该次迭代中的各个指标，类型：字典
        model: 模型路径
    '''
    featureData[np.isnan(featureData)] = 0
    # 分为六类
    y1 = np.where(featureLabel == 1)[0]
    y2 = np.where(featureLabel == 2)[0]
    y3 = np.where(featureLabel == 3)[0]
    y4 = np.where(featureLabel == 4)[0]
    y5 = np.where(featureLabel == 5)[0]
    y6 = np.where(featureLabel == 6)[0]
    y7 = np.where(featureLabel == 7)[0]
    featureLabel[y2] = 1
    featureLabel[y3] = 2
    featureLabel[y4] = 3
    featureLabel[y5] = 4
    featureLabel[y6] = 5
    featureLabel[y7] = 6
    
    featureData2 = featureData
    featureLabel2 = featureLabel
    # # 添加标准化、归一化
    # std = StandardScaler()
    # featureData2 = std.fit_transform(featureData2)
    # joblib.dump(std, 'std.pkl')  # 保存std模型
    
    # 加入LDA进行降维
    lda = LinearDiscriminantAnalysis()
    lda.fit(featureData2, featureLabel2)
    featureData2 = lda.transform(featureData2)
    joblib.dump(lda, 'lda.pkl')  # 保存lda模型

    outer_cv = KFold(n_splits=5, random_state=1, shuffle=True)  # K折交叉验证，5折
    RFaccList = []
    RFkappaList = []
    rfPreLabel = []
    realLabel = []
    for train_index, test_index in outer_cv.split(featureLabel2):
       
        y_train = featureLabel2[train_index]
        X_train_data = featureData2[train_index, :]
        y_test = featureLabel2[test_index]
        X_test_data = featureData2[test_index, :]

        realLabel.extend(y_test)
        X_train = X_train_data
        X_test = X_test_data
        rnd_clf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=1, oob_score=True)
        rnd_clf.fit(X_train, y_train)
        outputRF = rnd_clf.predict(X_test)
        rfPreLabel.extend(list(outputRF))
        RFaccList.append(accuracy_score(y_test, outputRF))
        RFkappaList.append(cohen_kappa_score(y_test, outputRF))
    
    joblib.dump(rnd_clf, 'model.pkl')  # 保存模型
    result = {'C':[], 'acc':[],'pre':[], 'sen':[],'f1':[], 'kappa':[], 'c_r':[]}
    print('start rf')
    C, acc, pre, sen, f1, kappa, c_r = calResult(realLabel, rfPreLabel)
    result['C'].append(C)
    result['acc'].append(acc)
    result['pre'].append(pre)
    result['sen'].append(sen)
    result['f1'].append(f1)
    result['kappa'].append(kappa) # 一个数值
    result['c_r'].append(c_r)  # 类型:str
    model = './model.pkl'
    
    return result, model

    
# 测试
if __name__ == "__main__":
    
    MatSaveBnf = '../bnfeature/'   # 网络特征保存路径
    
    # 加载数据
    X, y = load_data(MatSaveBnf)  # X是数据，y是标签
    
    # 训练
    result, model = train(X,y) 
    
    
    
    
