#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 16 00:41:35 2018

@author: chayan
"""

from pandas import read_csv
from pandas import concat
from matplotlib import pyplot
dataset1 = read_csv('train_data.csv', header=0, index_col=0)

dataset2=read_csv('test_data.csv', header=0, index_col=0)

dataset3=concat([dataset1,dataset2])
                                                                                                                                                                                                                                                                                                                                                                                                        

values = dataset3.values
groups = [0, 1, 2, 3, 5, 6, 7]
i = 1
pyplot.figure()
for group in groups:
	pyplot.subplot(len(groups), 1, i)
	pyplot.plot(values[:, group],'k')
	pyplot.title(dataset3.columns[group], y=0.5, loc='right')
	i += 1
pyplot.show()


# Lets normalize all features, and remove the weather variables for the hour to be predicted.
import pandas as pd 
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
def s_to_super(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = pd.DataFrame(data)
	cols, names = list(), list()
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	agg = pd.concat(cols, axis=1)
	agg.columns = names
	if dropnan:
            agg.dropna(inplace=True)
            return agg
encoder = preprocessing.LabelEncoder()
values[:,4] = encoder.fit_transform(values[:,4])
values = values.astype('float32')
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
reframed = s_to_super(scaled, 1, 1)

values=reframed.values


values= values[: , 0 : 9]

class_=values[:,8]
ad=[]

for i in class_:
   if (i>=0.0 and i<=0.1):
       a=0
       ad.append(a)
   elif (i>0.1 and i<=0.2):
       a=1
       ad.append(a)
   elif (i>0.2 and i<=0.3):
       a=2
       ad.append(a)
   elif (i>0.3 and i<=0.4):
       a=3
       ad.append(a)
   elif (i>0.4 and i<=0.5):
       a=4
       ad.append(a)
   elif (i>0.5 and i<=0.6):
       a=5
       ad.append(a)
   elif (i>0.6 and i<=0.7):
       a=6
       ad.append(a)
   elif (i>0.7 and i<=0.8):
       a=7
       ad.append(a)
   elif (i>0.8 and i<=0.9):
       a=8
       ad.append(a)
   else:
       a=9
       ad.append(a)

ad=np.asarray(ad)
ad=ad.reshape(ad.shape[0],1)
#ad=ad[1:,:]
values=values[:,:-1]
values=np.concatenate((values,ad ), axis=1)

train_hours=30000

train=values[: train_hours, : ]
test=values[train_hours: , :]

train_X, train_Y = train[:, :-1], train[:, -1]
test_X, test_Y = test[:, :-1],test[:,-1]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))


####MODEL
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense, Dropout


model = Sequential()
# 50 neurons in first hidden layer
model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dropout(0.4))
model.add(Dense(1,kernel_initializer='normal', activation='sigmoid' ))
model.compile(loss='mae', optimizer='adam' , metrics=['accuracy'])
history = model.fit(train_X, train_Y, epochs=100, batch_size=24, validation_data=(test_X, test_Y),verbose=2, shuffle=False)
score=model.evaluate(test_X,test_Y)
print('Error= ',score[0])
print('Accuracy= ',score[1])
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")
pyplot.plot(history.history['loss'], 'b', label='training history')
pyplot.plot(history.history['val_loss'],  'r',label='testing history')
pyplot.title("Train and Test Loss for the LSTM")
pyplot.legend()
pyplot.show()

