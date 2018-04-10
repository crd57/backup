# -*- coding:utf-8 _*-  
""" 
@author:crd
@file: weather.py 
@time: 2018/04/09 
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

filename = 'month.csv'
data = pd.read_csv(filename)
data_DelRainError = data.drop(data.index[abs(data['V13011']) == 32766])  # 删除没有降水量的行
data_station1 = data_DelRainError[data_DelRainError['V01000'] == 52754]  # 刚察
data_station2 = data_DelRainError[data_DelRainError['V01000'] == 52842]  # 共和
data_station3 = data_DelRainError[data_DelRainError['V01000'] == 52856]  # 茶卡
# plt.plot(data_station1['V04001'], data_station1['V13011'])
# plt.plot(data_station2['V04001'], data_station2['V13011'])
# plt.plot(data_station3['V04001'], data_station3['V13011'])
# plt.show()
Sattion = []
Year = []
data_year = []
for sattion in [52754, 52842, 52856]:
    data_station = data_DelRainError[data_DelRainError['V01000'] == sattion]
    for year in range(1978, 2017):
        Sattion.append(sattion)
        Year.append(year)
        c = data_station[data_station['V04001'] == year]
        data_year.append((c['V12001'].sum()/12)/10)
data = {'Sattion': Sattion, 'Year': Year, 'data_year': data_year}
frame = pd.DataFrame(data)
s1 = frame[frame['Sattion'] == 52856]
s2 = frame[frame['Sattion'] == 52754]
s3 = frame[frame['Sattion'] == 52842]

