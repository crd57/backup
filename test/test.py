# -*- coding:utf-8 _*-  
""" 
@author:crd
@file: test.py 
@time: 2018/03/03 
"""

import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima_model import ARIMA
import statsmodels.api as sm
from sklearn import preprocessing


def test_stationarity(timeseries):
    # 决定起伏统计
    rolmean = pd.rolling_mean(timeseries, window=3)  # 对size个数据进行移动平均
    rol_weighted_mean = pd.ewma(timeseries, span=3)  # 对size个数据进行加权移动平均
    rolstd = pd.rolling_std(timeseries, window=3)  # 偏离原始值多少
    # 画出起伏统计
    plt.plot(timeseries, color='blue', label='Original')
    plt.plot(rolmean, color='red', label='Rolling Mean')
    plt.plot(rol_weighted_mean, color='green', label='weighted Mean')
    plt.plot(rolstd, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    # 进行df测试
    print('Result of Dickry-Fuller test')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical value(%s)' % key] = value
    print(dfoutput)


data = pd.read_csv('qihaihu.CSV')
data = np.float64(data).reshape([25])
data_scale = preprocessing.scale(data)
data_series = pd.Series(data)
data_series.index = pd.Index(sm.tsa.datetools.dates_from_range('1987', '2011'))
plt.plot(data_series)
plt.show()
# test_stationarity(data_series)

date_diff1 = data_series.diff(1)
date_diff1.dropna(inplace=True)
test_stationarity(date_diff1)
date_diff2 = data_series.diff(2)
# test_stationarity(date_diff2)
data_series.plot()
date_diff1.plot()
date_diff2.plot()
plt.show()
plt.show()
