
from PreProcessing import preprocess
from net import  Inet
import os
import pickle
from keras import utils
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

'''load hyperparameters'''
BATCH_SIZE = 64
NUM_CLASESS = 2
EPOCHS =2000
NUM_FETRUES=4#nums of featrue
INPUT_SHAPE=(1,NUM_FETRUES)
VALIDATION_SPLIT=0.2
#data preprocessing

trainPath='train.csv'
testPath='test.csv'
X_train,Y_train,X_test=preprocess(trainPath,testPath)
'''from shape=(n,1) into shape=(n,num_classes)'''
Y_train = utils.to_categorical(Y_train, NUM_CLASESS)

print('X_train:',X_train.shape)
print('Y_train:',Y_train.shape)
print('X_test:',X_test.shape)



model = Inet.buildInet(INPUT_SHAPE,NUM_CLASESS)


modelfile = 'modelweight.model' #save weight
file_path_history = 'historyfile.bin'#save history for matplot

if os.path.exists(modelfile):#if there is .model file
print('load .model file')
model.load_weights(modelfile)
else:
print('trainning')
history = model.fit(X_train, Y_train,
batch_size=BATCH_SIZE,
epochs=EPOCHS,
verbose=1,
validation_split=VALIDATION_SPLIT)
print('\n')
model.save(modelfile)
historyfile = open(file_path_history, 'wb')
pickle.dump(history, historyfile)
historyfile.close()



'''test and output'''
Y_test=model.predict(X_test)#return ndarray

Y_test=np.argmax(Y_test,axis=1)#change binary matrix to category matrix
print('shape of Y_TEST:',Y_test.shape)
temp=np.arange(Y_train.shape[0]+1,Y_train.shape[0]+Y_test.shape[0]+1)
Y_test=np.vstack([temp,Y_test])

print(Y_train.shape)
Y_test=np.transpose(Y_test)
print(Y_test.shape)

df_out=pd.DataFrame(Y_test)

df_out.to_csv(path_or_buf='out.csv',index=None,header=['PassengerId','Survived'])

'''plot'''
if os.path.exists(file_path_history):#if there is history file
historyfile=open(file_path_history,'rb')
#historyfile.read()
historyfile.seek(0)
history = pickle.load(historyfile)

fig = plt.figure(1, figsize=(10, 5))

plt.subplot(1,2,1)
plt.plot(history.history["acc"])
plt.plot(history.history["val_acc"])
plt.title("Model accuracy")
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.legend(["train", "test"], loc="upper left")
# summarize history for loss
plt.subplot(1,2,2)
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("Model loss")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend(["train", "test"], loc="upper left")
plt.show()