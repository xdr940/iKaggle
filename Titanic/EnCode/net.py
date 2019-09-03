from keras.models import Sequential
from keras.layers.core import Activation,Flatten,Dense,Dropout
from keras import optimizers


class Inet:
	def buildInet(input_shape,num_classes):

		model = Sequential()

		model.add(Dense(units=10, activation='sigmoid', input_dim=input_shape[1]))  # input of first layer is 4 and output is 10,using sigmod as activation
		model.add(Dropout(0.25))
		model.add(Dense(units=10, activation='relu', input_dim=10))  # input of second layer is 10 and output is 2,using sigmod as activation
		model.add(Dropout(0.25))
		model.add(Dense(units=5, activation='relu', input_dim=10))  # input of third layer is 10 and output is 5,using sigmod as activation
		model.add(Dropout(0.25))

		model.add(Dense(num_classes))# input of fouth layer is 5 and output is 2,using softmax as activation
		model.add(Activation('softmax'))

		#model.compile(loss='binary_crossentropy', optimizer=optimizers.rmsprop(lr=0.001, decay=1e-6), metrics=['accuracy'])
		model.compile(loss='binary_crossentropy', optimizer=optimizers.SGD(lr=0.01, clipvalue=0.5), metrics=['accuracy'])#sgd as optimizer

		model.summary()
		return model


