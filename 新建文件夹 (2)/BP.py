# -*- coding:utf-8 _*-  
""" 
@author:crd
@file: BP.py 
@time: 2018/03/16 
"""
import numpy as np
import pandas as pd


def tanh(x):
    return np.tanh(x)


def tanh_deriv(x):  # tanh导数
    return 1.0 - np.tanh(x) * np.tanh(x)


def logistic(x):
    return 1 / (1 + np.exp(-x))


def logistic_derivative(x):  # logistic导数
    return logistic(x) * (1 - logistic(x))


class NeuralNetwork:
    def __init__(self, layers, activation='tanh'):
        """

        :param layers: A list containing the number of units in each layer.
        Should be at least two values
        :param activation: The activation function to be used. Can be
        "logistic" or "tanh"
        """
        if activation == 'logistic':
            self.activation = logistic
            self.activation_deriv = logistic_derivative
        elif activation == 'tanh':
            self.activation = tanh
            self.activation_deriv = tanh_deriv

        self.weights = []
        for i in range(1, len(layers) - 1):
            self.weights.append((2 * np.random.random((layers[i - 1] + 1, layers[i] + 1)) - 1) * 0.25)
            self.weights.append((2 * np.random.random((layers[i] + 1, layers[i + 1])) - 1) * 0.25)

    def fit(self, X, y, learning_rate=0.2, epochs=10000):
        """

        :param X: 训练数据
        :param y: 样本标签
        :param learning_rate: 学习速率
        :param epochs: 训练次数
        :return: 权重系数
        """
        # 添加1
        X = np.atleast_2d(X)
        temp = np.ones([X.shape[0], X.shape[1] + 1])
        temp[:, 0:-1] = X
        X = temp
        y = np.array(y)

        for k in range(epochs):
            i = np.random.randint(X.shape[0])
            a = [X[i]]

            for l in range(len(self.weights)):
                a.append(self.activation(np.dot(a[l], self.weights[l])))
            error = y[i] - a[-1]
            deltas = [error * self.activation_deriv(a[-1])]

            for l in range(len(a) - 2, 0, -1):
                deltas.append(deltas[-1].dot(self.weights[l].T) * self.activation_deriv(a[l]))
            deltas.reverse()

            for i in range(len(self.weights)):
                layer = np.atleast_2d(a[i])
                delta = np.atleast_2d(deltas[i])
                self.weights[i] += learning_rate * layer.T.dot(delta)
        np.save('weights', self.weights)

    def predict(self, x):
        x = np.array(x)
        temp = np.ones([x.shape[0] + 1, 1])
        temp[0:-1] = x
        a = temp
        a = a.transpose()
        for l in range(0, len(self.weights)):
            a = self.activation(np.dot(a, self.weights[l]))
        return a


def normalization(data):
    max = data.max()
    min = data.min()
    mean = data.mean()
    data_normalization = (data - mean) / (max - min)
    return data_normalization


def renormalization(x, data):
    max = data.max()
    min = data.min()
    mean = data.mean()
    x_renormalization = x * (max - min) + mean
    return x_renormalization


def data_preparation(filename='qihaihu.csv'):
    data = pd.read_csv(filename)
    data = np.array(data)
    data_normalization = normalization(data)
    n = 4
    x_re = []
    for i in range(len(data) - n):
        x_re.append([])
        for k in range(n):
            x_re[i].append(data_normalization[i + k])
    x = np.array(x_re)
    x = x.reshape(x.shape[0], n)
    y = data_normalization[n:]
    return data, x, y



data, x, y = data_preparation()
layers = list([4, 5, 1])
ANN = NeuralNetwork(layers)
ANN.fit(x, y)

x_p = normalization(data)[len(data) - 4:]
y_p = ANN.predict(x_p)
Y = renormalization(y_p, data)

testing_instance_matrix = x_p
p_labels = y_p
for k in range(len(testing_instance_matrix) - 1):
    testing_instance_matrix[k] = testing_instance_matrix[k + 1]
testing_instance_matrix[-1] = p_labels[-1]
np.append(p_labels, ANN.predict(testing_instance_matrix))


