# -*- coding:utf-8 _*-  
""" 
@author:crd
@file: BP_predict.py 
@time: 2018/03/17 
"""
import BP
data, x, y = BP.data_preparation()
layers = list([4, 5, 6, 7, 1])
ANN = BP.NeuralNetwork(layers)
ANN.fit(x, y)
x_p = data[len(data) - 4:]
x_p.reshape(4)
y_p = ANN.predict(x_p)
Y = BP.renormalization(y_p, data)