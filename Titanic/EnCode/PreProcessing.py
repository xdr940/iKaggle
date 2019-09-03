
import numpy as np
import pandas as pd
from pandas import  read_csv
def preprocess(trainPath,testPath):
	df_train = pd.read_csv(trainPath,  index_col= False)
	df_test = pd.read_csv(testPath,  index_col= False)

	df_train = df_train[['Survived','Pclass','Sex','Age','Fare']]
	df_test = df_test[['Pclass','Sex','Age','Fare']]

	df_train['Sex']=df_train['Sex'].replace(['male','female'],[1,0])#turn male to 1，female to 0
	df_test['Sex']=df_test['Sex'].replace(['male','female'],[1,0])#把male换成1，female换成0

	df_train=df_train.fillna(0)#turn null to 0
	df_test=df_test.fillna(0)


	#  z-score nomorlization
	samples=pd.concat([df_train,df_test])#return a df
	print(samples.shape)
	mean_age=samples['Age'].mean()
	mean_fare=samples['Fare'].mean()

	std_age = samples['Age'].std()
	std_fare=samples['Fare'].std()

	df_train['Age']=(df_train['Age']-mean_age)/std_age
	df_train['Fare']=(df_train['Fare']-mean_fare)/std_fare

	df_test['Age']=(df_test['Age']-mean_age)/std_age
	df_test['Fare']=(df_test['Fare']-mean_fare)/std_fare
	#nomorlize 01
	'''
		maxAge=max(df_test['Age'].max(),df_train['Age'].max())
		minAge=min(df_test['Age'].min(),df_train['Age'].min())

		maxFare = max(df_test['Fare'].max(),df_train['Fare'].max())
		minFare = min(df_test['Fare'].min(),df_train['Fare'].min())

		df_train['Age']/=(maxAge-minAge)
		df_test['Age']/=(maxAge-minAge)

		df_train['Fare']/=(maxFare-minFare)
		df_test['Fare']/=(maxFare-minFare)
	'''
	# start test processing

	X_train=df_train[['Pclass','Sex','Age','Fare']].values
	Y_train=df_train['Survived'].values

	X_test=df_test[['Pclass','Sex','Age','Fare']].values


	return X_train,Y_train,X_test

#def to_class_vec(dims,Y):
