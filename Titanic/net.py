from keras.models import Sequential
from keras.layers.core import Activation,Flatten,Dense,Dropout
from keras import optimizers


class Inet:
	def buildInet(input_shape,num_classes):

		model = Sequential()

		model.add(Dense(units=10, activation='sigmoid', input_dim=input_shape[1]))  # 第一层10个神经元，用sigmoid激活函数，输入维度3
		model.add(Dropout(0.25))
		model.add(Dense(units=10, activation='relu', input_dim=10))  # 第2层10个神经元，用sigmoid激活函数，输入维度10
		model.add(Dropout(0.25))
		model.add(Dense(units=5, activation='relu', input_dim=10))  # 第3层5个神经元，用sigmoid激活函数，输入维度10
		model.add(Dropout(0.25))

		model.add(Dense(num_classes))# 第4层2个神经元，用softmax激活函数，输入维度5
		model.add(Activation('softmax'))

		#model.compile(loss='binary_crossentropy', optimizer=optimizers.rmsprop(lr=0.001, decay=1e-6), metrics=['accuracy'])
		model.compile(loss='binary_crossentropy', optimizer=optimizers.SGD(lr=0.01, clipvalue=0.5), metrics=['accuracy'])#随机梯度下降作为优化器

		model.summary()
		return model


