from baseparams import baseParams
import matplotlib.pyplot as plt
import missingno as msno
import seaborn as sns
import numpy as np
import scipy.stats as st # 分布
class analysisFuns(baseParams):

    def __init__(self):
        super().__init__()

    def pinrtDfInfo(self, df, b_train=True):
        if b_train:
            dfName = 'trainData'
        else:
            dfName = 'testData'
        # 1) 简略观察数据(head()+tail()+shape)
        print('******* shape ******')
        print('shape of ' + dfName + ': ', df.shape)
        print()
        print('*******  head and tail  *******')
        headTail = df.head().append(df.tail())
        print('----- ' + 'head and tail of ' + dfName + ': ')
        print(headTail)
        # 2) 通过describe()来熟悉数据的相关统计量
        print()
        print('******* describe *******')
        print('----- ' + 'descirbe ' + dfName + ': ')
        print(df.describe())
        # 3) 通过info()来熟悉数据类型
        print()
        print('******* info *******')
        print('----- ' + 'info ' + dfName + ': ')
        print(df.info())

    def missingDataInfo(self,df,b_train=True):
        ## 1) 查看每列的存在nan情况
        if b_train:
            dfName = 'trainData'
        else:
            dfName = 'testData'
        df_sumNull = df.isnull().sum()
        print('----- sum null of '+ dfName + ': ')
        print(df_sumNull)
        # 柱状图
        plt.figure()
        df_sumNull = df_sumNull[df_sumNull > 0]
        df_sumNull.sort_values(inplace=True)
        df_sumNull.plot.bar()  # dataFrame plot ---> 如何画图出来？
        plt.title('sum null of '+ dfName)
        plt.show()
        # 可视化看下缺省值-训练集
        msno.matrix(df.sample(250))
        msno.bar(df.sample(1000))

    # input: df['col']
    def colDataDistribution(self,dfSeries):
        colName = dfSeries.columns  # ?
        print(dfSeries[colName])  # print()
        print(colName+'valu_count(): ')
        dfSeries[colName].value_counts()

        ## 1) 总体分布概况（无界约翰逊分布等）
        y = dfSeries[colName]
        plt.figure(1)
        plt.title('Johnson SU')
        sns.distplot(y, kde=False, fit=st.johnsonsu)
        plt.figure(2)
        plt.title('Normal')
        sns.distplot(y, kde=False, fit=st.norm)
        plt.figure(3)
        plt.title('Log Normal')
        sns.distplot(y, kde=False, fit=st.lognorm)
        # 价格不服从正态分布，所以在进行回归之前，它必须进行转换。虽然对数变换做得很好，但最佳拟合是无界约翰逊分布
        ## 2) 查看skewness and kurtosis
        sns.distplot(dfSeries[colName])
        print("Skewness: %f" % dfSeries[colName].skew())
        print("Kurtosis: %f" % dfSeries[colName].kurt())

        dfSeries.skew(), dfSeries.kurt()  # print
        sns.distplot(dfSeries.skew(), color='blue', axlabel='Skewness')
        sns.distplot(dfSeries.kurt(), color='orange', axlabel='Kurtness')

        ## 3) 查看预测值的具体频数
        plt.hist(dfSeries[colName], orientation='vertical', histtype='bar', color='red')
        plt.show()
        # 查看频数, 大于20000得值极少，其实这里也可以把这些当作特殊得值（异常值）直接用填充或者删掉，再前面进行
        # log变换 z之后的分布较均匀，可以进行log变换进行预测，这也是预测问题常用的trick
        plt.hist(np.log(dfSeries[colName]), orientation='vertical', histtype='bar', color='red')
        plt.title('after log transfer')
        plt.show()

