# -*- coding: utf-8 -*-
"""
# @Time    : 18-4-7 下午7:55
# @Author  : Crd
# @Email   : crd57@outlook.com
# @File    : array_3D.py
# @Software: PyCharm
"""
from arrays import Array


class Grid(object):
    def __init__(self, rows, columns, depth, fillValues=None):
        self._data = Array(rows)
        for i in range(rows):
            self._data[i] = Array(columns)
            for j in range(columns):
                self._data[i][j] = Array(depth, fillValues)


A = Grid(3, 3, 2, 5)
A[2]