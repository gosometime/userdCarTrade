from baseparams import baseParams


class dataAnalysis(baseParams):
    def __init__(self,Train_data,Test_data):
        super().__init__()
        self.Train_data = Train_data
        self.Test_data = Test_data

    def printDfInfo(self):
        ## 2) 简略观察数据(head()+shape)
        print('******* shape ******')
        print('train data shape:',self.Train_data.shape)
        print('test data shape:',self.Test_data.shape)

        print()
        print('*******  head and tail  *******')
        trainData_head_tail = self.Train_data.head().append(self.Train_data.tail())
        print('----- train data head and tail ----')
        print(trainData_head_tail)
        print()
        print('----- test data head and tail ----')
        testData_head_tail = self.Test_data.head().append(self.Test_data.tail())
        print(testData_head_tail)
        ## 1) 通过describe()来熟悉数据的相关统计量
        print()
        print('******* describe *******')
        print('Train data descirbe:')
        print(self.Train_data.describe())
        print()
        print('Test data descirbe:')
        print(self.Test_data.describe())
        ## 2) 通过info()来熟悉数据类型
        print('******* data info *******')
        print()
        print('self.Train_data.info:')
        print(self.Train_data.info())
        print()
        print('self.Test_data info:')
        print(self.Test_data.info())

