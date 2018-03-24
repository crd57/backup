# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:     CNN
   Author:        crd
   date:          2018/3/23
-------------------------------------------------
"""
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_digits
from sklearn.preprocessing import OneHotEncoder


def data_pre(X_data, Y_data):
    X = MinMaxScaler().fit_transform(X_data)  # 每一列最大值最小值标准化
    Y = OneHotEncoder().fit_transform(Y_data).todense()  # one-hot编码
    X = X.reshape(-1, 8, 8, 1)
    return X, Y


def generatebatch_MBGD(X, Y, n_examples, batch_size=8):
    for batch_i in range(n_examples // batch_size):   # //整数除法
        start = batch_i * batch_size
        end = start + batch_size
        batch_xs = X[start:end]
        batch_ys = Y[start:end]
        yield batch_xs, batch_ys


def generatebatch_RBGD(X, Y, n_examples, n):
    for i in range(n):
        batch_i = np.random.randint(0, n_examples)
        batch_xs = X[batch_i]
        batch_ys = Y[batch_i]
        yield batch_xs, batch_ys


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_3x3(x):
    return tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
    #  [batch, height, width, channel] 池化窗口大小
    #  [batch, height, width, channel] 窗口滑动步长


digits = load_digits()
X_data = digits.data.astype(np.float32)
Y_data = digits.target.astype(np.float32).reshape(-1, 1)
X, Y = data_pre(X_data, Y_data)
tf.reset_default_graph()
tf_X = tf.placeholder(tf.float32, [None, 8, 8, 1])
tf_Y = tf.placeholder(tf.float32, [None, 10])

W_conv1 = weight_variable([3, 3, 1, 10])  # 前两维patch的大小，输入通道数，输出的通道数。
b_conv1 = bias_variable([10])
h_conv1 = tf.nn.relu(conv2d(tf_X, W_conv1) + b_conv1)

h_pool1 = max_pool_3x3(h_conv1)

W_conv2 = weight_variable([3, 3, 10, 5])
b_conv2 = bias_variable([5])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

h_pool2 = max_pool_3x3(h_conv2)

h_pool2_flat = tf.reshape(h_pool2, [-1, 2 * 2 * 5])
W_fc = tf.Variable(tf.random_normal([2 * 2 * 5, 50]))
b_fc = tf.Variable(tf.random_normal([50]))
h_fc = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc) + b_fc)

W_out = tf.Variable(tf.random_normal([50, 10]))
b_out = tf.Variable(tf.random_normal([10]))
out_layer = tf.nn.softmax(tf.matmul(h_fc, W_out) + b_out)

loss = -tf.reduce_mean(tf_Y * tf.log(tf.clip_by_value(out_layer, 1e-11, 1.0)))  # 损失函数
# clip_by_value = 输入一个张量A，把A中的每一个元素的值都压缩在min和max之间。小于min的让它等于min，大于max的元素的值等于max。
# reduce_mean 求平均值  reduce_max 求最大值
train_step = tf.train.AdamOptimizer(1e-3).minimize(loss)  # Adam 算法优化器

y_pred = tf.arg_max(out_layer, 1)  # 返回最大的那个数值所在的下标
bool_pred = tf.equal(tf.arg_max(tf_Y, 1), y_pred)  # 是对比这两个矩阵或者向量的相等的元素，如果是相等的那就返回True
accuracy = tf.reduce_mean(tf.cast(bool_pred, tf.float32))  # cast 数据格式转换
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(1000):
        for batch_xs, batch_ys in generatebatch_MBGD(X, Y, Y.shape[0]):
            sess.run(train_step, feed_dict={tf_X: batch_xs, tf_Y: batch_ys})
            if epoch % 100 == 0:
                res = sess.run(accuracy, feed_dict={tf_X: X, tf_Y: Y})
                print(epoch, res)
    res_ypred = y_pred.eval(feed_dict={tf_X: X, tf_Y: Y}).flatten()
    print(res_ypred)





