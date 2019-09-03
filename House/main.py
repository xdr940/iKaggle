#invite people for the Kaggle party
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
from sklearn import preprocessing
normlize = True
warnings.filterwarnings('ignore')
#%matplotlib inline
df_train = pd.read_csv('./data/train.csv')
df_test = pd.read_csv('./data/train.csv')

sale_price=df_train['SalePrice']


if normlize:
    sale_price_values = sale_price.values#series 2 numpy, shape=(1460,)
    reshapes = sale_price.values.reshape(-1, 1)# shape (1460,1)
    sale_price_scaled = preprocessing.StandardScaler().fit_transform(reshapes)#
    s = pd.Series(reshapes.squeeze(1))
    sns.distplot(s)

