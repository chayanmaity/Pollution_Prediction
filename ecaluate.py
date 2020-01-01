#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 16 01:29:36 2018

@author: chayan
"""
import pandas as pd
import numpy as np
from keras.models import model_from_json

dataset = pd.read_csv('unseen_data.csv', header=0, index_col=0)
values=dataset.values

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
        
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
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

X, Y = values[:, :-1], values[:, -1]
X=X.reshape(X.shape[0], 1,X.shape[1])




json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")
 
# evaluate loaded model on test data
loaded_model.compile(loss='mae', optimizer='adam' , metrics=['accuracy'])
score = loaded_model.evaluate(X, Y, verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))