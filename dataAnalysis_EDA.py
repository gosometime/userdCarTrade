from baseparams import baseParams
import matplotlib.pyplot as plt
import missingno as msno
from analysisFunctions import analysisFuns

class dataAnalysis(analysisFuns):
    def __init__(self,Train_data,Test_data):
        super().__init__()
        self.Train_data = Train_data
        self.Test_data = Test_data
        self.colList = Train_data.columns

        # col name list
        self.numeric_features = None
        self.categorical_features = None
    #
    def printInfo(self):
        self.pinrtDfInfo(self.Train_data,b_train=True)
        self.pinrtDfInfo(self.Test_data,b_train=False)


    def missingDataStat(self):
        print()
        print('*' * 7 + ' nan info of each col ' + '*' * 7)
        self.missingDataInfo(self.Train_data,b_train=True)
        self.missingDataInfo(self.Test_data,b_train=False)

    def labelDistribution(self):
        self.colDataDistribution(self.Train_data[self.colList[-1]])

    def getFeatureList(self,numFeatList,catFeatList):
        self.numeric_features = numFeatList
        self.categorical_features = catFeatList



