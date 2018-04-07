# -*- coding: utf-8 -*-
"""
# @Time    : 18-4-2 上午10:08
# @Author  : Crd
# @Email   : crd57@outlook.com
# @File    : logistic.py
# @Software: PyCharm
"""
from __future__ import division  # 导入未来支持的用法——精确除法
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import precision_recall_curve, roc_curve, auc


data = pd.read_csv('data1.txt', sep=',', skiprows=[2], names=['score1', 'score2', 'result'])
x = data.loc[:, ['score1', 'score2']]
y = data.result
for i in range(10):
    p = 0
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    model = svm.SVC()
    model.fit(x_train, y_train)
    predict_y = model.predict(x_test)
    p += (np.mean(predict_y == y_test))

"""
pos_data = data[data.result == 1].loc[:, ['score1', 'score2']]
neg_data = data[data.result == 0].loc[:, ['score1', 'score2']]
h = 0.02
x_min, x_max = x.loc[:, ['score1']].min(), x.loc[:, ['score1']].max()
y_min, y_max = x.loc[:, ['score2']].min(), x.loc[:, ['score2']].max()
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])

Z = Z.reshape(xx.shape)
plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)
plt.scatter(x=pos_data.score1, y=pos_data.score2, color='black', marker='o')
plt.scatter(x=neg_data.score1, y=neg_data.score2, color='red', marker='*')

plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.show()

answer = model.predict_proba(x_test)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_test, answer)
report = answer > 0.5
print(classification_report(y_test, report, target_names = ['neg', 'pos']))
print("average precision:", p/100)
"""