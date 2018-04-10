# -*- coding:utf-8 _*-
""" 
@author:crd
@file: liner.py 
@time: 2018/02/03 
"""
from __future__ import print_function

import pandas as pd
import time
import warnings
import numpy as np
import time
import matplotlib.pyplot as plt
from numpy import newaxis
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential




def load_data(filename, seq_len=3, n=0.9):
    data = pd.read_csv(filename)
    data = np.float64(data).reshape([len(data)])
    data = np.array(data)
    sequence_length = seq_len + 1
    result = []
    for i in range(len(data) - sequence_length):
        result.append(data[i:i + sequence_length])
    result = np.array(result)
    row = round(n * result.shape[0])
    train = result[:row, :]
    x_train = train[:, :-1]
    y_train = train[:, -1]
    x_test = result[row:, :-1]
    y_test = result[row:, -1]
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    return [x_train, y_train, x_test, y_test]


def build_model(layers):
    model = Sequential()
    model.add(LSTM(input_dim=layers[0], output_dim=layers[1], return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(layers[2], return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(output_dim=layers[3]))
    model.add(Activation("linear"))
    start = time.time()
    model.compile(loss="mse", optimizer="rmsprop")
    print("Compilation Time: ", time.time() - start)
    return model


def predict_point_by_point(model, data):
    predicted = model.predict(data)
    print('predicted shape:', np.array(predicted).shape)
    predicted = np.reshape(predicted, (predicted.size,))
    return predicted


"""
def plot_results_multiple(predicted_data, true_data, prediction_len):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    # Pad the list of predictions to shift it in the graph to it's correct start
    for i, data in enumerate(predicted_data):
        padding = [None for p in xrange(i * prediction_len)]
        plt.plot(padding + data, label='Prediction')
        plt.legend()
    plt.show()
    plt.savefig('plot_results_multiple.png')

"""
import liner

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    global_start_time = time.time()
    epochs = 1
    seq_len = 3

    print('> Loading data...')

    [x_train, y_train, x_test, y_test] = liner.load_data('qihaihu.csv')
    model = liner.build_model([1, 3, 100, 1])
    model.fit(x_train, y_train, nb_epoch=epochs, validation_split=0.05)
    point_by_point_predictions = liner.predict_point_by_point(model, x_test)
    print('point_by_point_predictions shape:', np.array(point_by_point_predictions).shape)
    print('Training duration (s) : ', time.time() - global_start_time)
