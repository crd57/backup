# -*- coding:utf-8 _*-  
""" 
@author:crd
@file: readcsv.py 
@time: 2018/02/05 
"""

import pandas as pd
import numpy as np

def create_interval_dataset(dataset, look_back):
    """
    :param dataset: 输入时间序列
    :param look_back: 每一个训练集的长度
    :return: 将数组值转化为数据集矩阵
    """

    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        dataX.append(dataset[i:i+look_back])
        dataY.append(dataset[i+look_back])
    return np.asarray(dataX), np.asarray(dataY)

