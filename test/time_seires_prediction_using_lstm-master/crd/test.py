# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 10:59:06 2018

@author: wgj2018
"""

from keras.layers import SimpleRNN,Activation,Dense
model.add(SimpleRNN(
        #使用tensorflow作为backend，我们必须把batch_size设置为None
        #否则，model.evaluate()将会报错
        bath_input_shape=(None,TIME_STEP,INPUT_SIZE),
        output_dim=CELL_SIZE,
        unroll=True,
        ))
model.add(Dense(OUTOUT_SIZE))
model.add(Activation('linear'))
model.compile(optimizer='sgd',loss='mse')
