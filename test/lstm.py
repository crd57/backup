# -*- coding:utf-8 _*-  
""" 
@author:crd
@file: LSTM.py 
@time: 2018/03/09 
"""
from __future__ import print_function

import time
import pandas as pd
import warnings
import numpy as np
import time
import matplotlib.pyplot as plt
from numpy import newaxis
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential

warnings.filterwarnings("ignore")


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


def normalise_windows(window_data):
    normalised_data = []
    for window in window_data:  # window shape (sequence_length L ,)  即(51L,)
        normalised_window = [((float(p) / float(window[0])) - 1) for p in window]
        normalised_data.append(normalised_window)
    return normalised_data


def build_model(layers):  # layers [1,50,100,1]
    model = Sequential()

    model.add(LSTM(input_dim=layers[0], output_dim=layers[1], return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(layers[2], return_sequences=False))
    model.add(Dropout(0.2))

    model.add(Dense(output_dim=layers[3]))
    model.add(Activation("linear"))

    start = time.time()
    model.compile(loss="mse", optimizer="rmsprop")
    print("Compilation Time : ", time.time() - start)
    return model


# 直接全部预测
def predict_point_by_point(model, data):
    predicted = model.predict(data)
    print('predicted shape:', np.array(predicted).shape)  # (412L,1L)
    predicted = np.reshape(predicted, (predicted.size,))
    return predicted


# 滚动预测
def predict_sequence_full(model, data, window_size):  # data X_test
    curr_frame = data[0]  # (50L,1L)
    predicted = []
    for i in range(len(data)):
        # x = np.array([[[1],[2],[3]], [[4],[5],[6]]])  x.shape (2, 3, 1) x[0,0] = array([1])  x[:,np.newaxis,:,:].shape  (2, 1, 3, 1)
        predicted.append(
            model.predict(curr_frame[newaxis, :, :])[0, 0])  # np.array(curr_frame[newaxis,:,:]).shape (1L,50L,1L)
        curr_frame = curr_frame[1:]
        curr_frame = np.insert(curr_frame, [window_size - 1], predicted[-1],
                               axis=0)  # numpy.insert(arr, obj, values, axis=None)
    return predicted


def predict_sequences_multiple(model, data, window_size, prediction_len):  # window_size = seq_len
    prediction_seqs = []
    for i in range(np.int64(len(data) / prediction_len)):
        curr_frame = data[i * prediction_len]
        predicted = []
        for j in range(prediction_len):
            predicted.append(model.predict(curr_frame[newaxis, :, :])[0, 0])
            curr_frame = curr_frame[1:]
            curr_frame = np.insert(curr_frame, [window_size - 1], predicted[-1], axis=0)
        prediction_seqs.append(predicted)
    return prediction_seqs


def plot_results(predicted_data, true_data, filename):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    plt.plot(predicted_data, label='Prediction')
    plt.legend()
    plt.show()
    plt.savefig(filename + '.png')


def plot_results_multiple(predicted_data, true_data, prediction_len):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    # Pad the list of predictions to shift it in the graph to it's correct start
    for i, data in enumerate(predicted_data):
        padding = [None for p in range(i * prediction_len)]
        plt.plot(padding + data, label='Prediction')
        plt.legend()
    plt.show()
    plt.savefig('plot_results_multiple.png')


import lstm

if __name__ == '__main__':
    global_start_time = time.time()
    epochs = 1
    seq_len = 50

    print('> Loading data... ')

    X_train, y_train, X_test, y_test = lstm.load_data('sp500.csv', seq_len, True)

    print('X_train shape:', X_train.shape)  # (3709L, 50L, 1L)
    print('y_train shape:', y_train.shape)  # (3709L,)
    print('X_test shape:', X_test.shape)  # (412L, 50L, 1L)
    print('y_test shape:', y_test.shape)  # (412L,)

    print('> Data Loaded. Compiling...')

    model = lstm.build_model([1, 50, 100, 1])

    model.fit(X_train, y_train, batch_size=512, nb_epoch=epochs, validation_split=0.05)

    multiple_predictions = lstm.predict_sequences_multiple(model, X_test, seq_len, prediction_len=50)
    print('multiple_predictions shape:', np.array(multiple_predictions).shape)  # (8L,50L)

    #full_predictions = lstm.predict_sequence_full(model, X_test, seq_len)
    #print('full_predictions shape:', np.array(full_predictions).shape)  # (412L,)

    point_by_point_predictions = lstm.predict_point_by_point(model, X_test)
    print('point_by_point_predictions shape:', np.array(point_by_point_predictions).shape)  # (412L)

    print('Training duration (s) : ', time.time() - global_start_time)

    plot_results_multiple(multiple_predictions, y_test, 50)
    #plot_results(full_predictions, y_test, 'full_predictions')
    plot_results(point_by_point_predictions, y_test, 'point_by_point_predictions')
