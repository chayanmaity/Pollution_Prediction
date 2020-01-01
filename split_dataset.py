#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 16 00:27:32 2018

@author: chayan
"""

from pandas import read_csv
from matplotlib import pyplot
dataset = read_csv('pollution.csv', index_col=0)
train=dataset.iloc[:30000,:]
test=dataset.iloc[30000:40000,:]
unseen=dataset.iloc[40000:,:]

train.to_csv('train_data.csv')
test.to_csv('test_data.csv')
unseen.to_csv('unseen_data.csv')