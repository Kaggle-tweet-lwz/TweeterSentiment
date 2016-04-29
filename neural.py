from __future__ import print_function
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils

batch_size = 128
nb_classes = 10
nb_epoch = 20

df = np.load("temp5551.npz")
y = df['y']
X = df["X"][()].toarray()

index = []
i = []
for k in range(len(y)):
	if y[k]!=-1:
		index.append(k)
	i.append(k)

train = index
valid = range(50000)
# np.random.shuffle(index)
# train = index[:int(len(y) * .6)]
# valid = index[int(len(y) * .6):]

X_train, y_train = X[train], y[train]
X_test, y_test = X[valid], y[valid]

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

model = Sequential()
model.add(Dense(512, input_shape=(5004,)))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.2))
# model.add(Dense(256))
# model.add(Activation('relu')) #added layer with less accuracy in training
# model.add(Dropout(0.2))
model.add(Dense(128))
model.add(Activation('sigmoid'))
model.add(Dropout(0.1))
model.add(Dense(10))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

model.fit(X_train, Y_train, nb_epoch=nb_epoch, batch_size=batch_size)
classes = model.predict_classes(X_test, batch_size=32)
proba = model.predict_proba(X_test, batch_size=32)

# history = model.fit(X_train, Y_train,
#                     batch_size=batch_size, nb_epoch=nb_epoch,
#                     verbose=1, validation_data=(X_test, Y_test))
# score = model.evaluate(X_test, Y_test, verbose=0)
# print('Test score:', score[0])
# print('Test accuracy:', score[1])
