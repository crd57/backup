# -*- coding: utf-8 -*-
"""
# @Time    : 18-4-7 下午7:45
# @Author  : Crd
# @Email   : crd57@outlook.com
# @File    : array_disorder.py
# @Software: PyCharm
"""
from arrays import Array


class grid(object):
    def __init__(self, rows, columns, fillValue=None):
        self._data = Array(rows)
        for i in range(rows):
            self._data[i] = Array(columns[i], fillValue)
    
    def getHeight(self):
        return len(self._data)

    def getWidth(self, columns):
        return len(self._data[columns])


c = grid(3, [3, 6, 8], 2)
