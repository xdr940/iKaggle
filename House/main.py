#invite people for the Kaggle party
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
from sklearn import preprocessing
normlize = True
from path import Path
warnings.filterwarnings('ignore')
#%matplotlib inline
from sklearn.preprocessing import LabelEncoder
root = Path('/home/roit/datasets/kaggle/House')

df_train = pd.read_csv(root/'train.csv')#dataframe
#df_test = pd.read_csv(root/'train.csv')

#total = df_train.isnull().sum().sort_values(ascending=False)#每个特征空置数量，numpy[nums_featrues,]
#percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)#每个特征空值比例
#missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
#missing_data.head(20)
#有些特征,统计量缺失率太高,如下由高到低的排名

def strings_cnt(series):
    '''
        统计categoric variables (string)
    :param series:pandas.Series
    :return: pandas.DataFrame
    '''
    dic = {}
    total = len(series)
    for i in range(total):
        try:
            dic[series[i]] += 1

        except:
            dic[series[i]] = 1

    lis = sorted(dic.items(), key=lambda dic: dic[1], reverse=True)


    names = [it[0] for it in lis]
    cnt = [it[1] for it in lis]
    percent = [it[1] / total for it in lis]

    dic = {'names':names, 'cnt':cnt,'percent':percent}

    ret_df = DataFrame(dic)
    return ret_df


# 对catergorical(string) 变量的预处理 独热编码




print('Shape all_data: {}'.format(df_train.shape))

def main():
    cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond',
            'ExterQual', 'ExterCond', 'HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1',
            'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
            'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond',
            'YrSold', 'MoSold')
    for c in cols:
        lbl = LabelEncoder()
        lbl.fit(list(df_train[c].values))
        df_train[c] = lbl.transform(list(df_train[c].values))
        print(df_train[c].iloc[:10])


if __name__ =="__main__":
    main()