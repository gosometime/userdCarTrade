#coding:utf-8
#导入warnings包，利用过滤器来实现忽略警告语句。
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno

from dataAnalysis_EDA import dataAnalysis



if __name__ == '__main__':

    trainDataPath = 'data/used_car_train_20200313.csv'
    testDataPath = 'data/used_car_testB_20200421.csv'

    ## 1) 载入训练集和测试集；
    print(trainDataPath)
    Train_data = pd.read_csv(trainDataPath, sep=' ')
    Test_data = pd.read_csv(testDataPath, sep=' ')

    da = dataAnalysis(Train_data,Test_data)
    #da.printDfInfo()

    # nan可视化
    plt.figure()
    missing = Train_data.isnull().sum()
    missing = missing[missing > 0]
    missing.sort_values(inplace=True)
    #dataFrame plot ---> 如何画图出来？
    plt.show(missing.plot.bar())
    # 可视化看下缺省值
    msno.matrix(Train_data.sample(250))
    plt.show()