# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 15:15:50 2018

@author: Administrator
"""

import numpy as np
import pandas as pd
import pickle as pkl

def load_chengdu_data(dataset):
    chengdu_adj = pd.read_csv(r'data/chengdu_adj.csv',header=None)
    adj = np.mat(chengdu_adj)
    chengdu_tf = pd.read_csv(r'data/chengdu_speed.csv')
    return chengdu_tf, adj

def load_XiAn_data(dataset):
    Xian_adj = pd.read_csv(r'data/Xian_adj.csv',header=None)
    adj = np.mat(Xian_adj)
    xian_tf = pd.read_csv(r'data/Xian_speed.csv')
    return xian_tf, adj


def preprocess_data(data, time_len, rate, seq_len, pre_len):
    train_size = int(time_len * rate)
    train_data = data[0:train_size]
    test_data = data[train_size:time_len]
    
    trainX, trainY, testX, testY = [], [], [], []
    for i in range(len(train_data) - seq_len - pre_len):
        a = train_data[i: i + seq_len + pre_len]
        trainX.append(a[0 : seq_len])
        trainY.append(a[seq_len : seq_len + pre_len])
    for i in range(len(test_data) - seq_len -pre_len):
        b = test_data[i: i + seq_len + pre_len]
        testX.append(b[0 : seq_len])
        testY.append(b[seq_len : seq_len + pre_len])
      
    trainX1 = np.array(trainX)
    trainY1 = np.array(trainY)
    testX1 = np.array(testX)
    testY1 = np.array(testY)
    return trainX1, trainY1, testX1, testY1
    
