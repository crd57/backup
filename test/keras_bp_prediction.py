# -*- coding:utf-8 _*-  
""" 
@author:crd
@file: keras_bp_prediction.py 
@time: 2018/03/17 
"""
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from keras.models import Sequential
from keras.layers import Dense, Activation
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def data_preparation(ye, filename='qihaihu.csv'):
    data = pd.read_csv(filename)
    data_array = np.array(data)
    data_normalization = normalization(data_array)
    n = ye  # 序列长度
    x_re = []
    for i_1 in range(len(data) - n):
        x_re.append([])
        for k_1 in range(n):
            x_re[i_1].append(data_normalization[i_1 + k_1])
    x_1 = np.array(x_re)
    x_1 = x_1.reshape(x_1.shape[0], n)
    y_1 = data_normalization[n:]
    return data_array, x_1, y_1, data_normalization


def normalization(data1):
    max = np.max(data1)
    min = np.min(data1)
    ccc = data1 - min
    cc = max - min
    data_normalizations = ccc / cc
    return data_normalizations


def renormalization(x2, data2):
    max = np.max(data2)
    min = np.min(data2)
    x_renormalization = x2 * (max - min) + min
    return x_renormalization


years = 11
model = Sequential()
model.add(Dense(25, input_dim=years))  # 序列长度
model.add(Activation('relu'))
# model.add(Dense(25, activation='relu'))
# model.add(Dense(50, activation='sigmoid'))
model.add(Dense(50, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='mae')
data_arr, x, y, data_normalization = data_preparation(years)
model.fit(x, y, epochs=5000)
x_p = data_normalization[len(data_arr) - years:]  # 序列长度
x_p = np.transpose(x_p)
y_p = model.predict(x_p)
Y = list(renormalization(y_p, data_arr))
x_p = np.transpose(x_p)
testing_instance_matrix = []
p_labels = []
for i in range(1, 5):
    testing_instance_matrix = x_p
    p_labels = y_p
    for k in range(len(testing_instance_matrix) - 1):
        testing_instance_matrix[k] = testing_instance_matrix[k + 1]
    testing_instance_matrix[-1] = p_labels[-1]
    testing_instance_matrix = np.transpose(testing_instance_matrix)
    y_p = model.predict(testing_instance_matrix)
    Y.append(renormalization(y_p, data_arr))
Y = np.float64(Y)
dd = pd.read_csv('qi.csv')
d = np.array(dd)
plt.plot(Y, "x-", label="预测")
plt.plot(d, "+-", label="真值")
plt.grid(True)
plt.show()
mse = np.sum(np.abs(Y - d)) / (len(Y))
