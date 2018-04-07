# -*- coding: utf-8 -*-
"""
# @Time    : 18-4-7 下午5:28
# @Author  : Crd
# @Email   : crd57@outlook.com
# @File    : array_2D.py
# @Software: PyCharm
"""
from arrays import Array


class Grid(object):
    def __init__(self, rows, heights, fillnumbers=None):
        self._data = Array(rows)
        for i in range(rows):
            self._data[i] = Array(heights, fillnumbers)

    def getHeight(self):
        return len(self._data)

    def getRow(self):
        return len(self._data[0])

    def __getitem__(self, index):
        return self._data[index]

    def find_negative(self):
        result = str(self.getHeight()) + ' ' + str(self.getRow())
        for row in range(self.getHeight()):
            for col in range(self.getRow()):
                if self._data[row][col] < 0:
                    result = str(row)+' '+str(col)
                    break
        return result

    def __str__(self):
        result = ""
        for row in range(self.getHeight()):
            for col in range(self.getRow()):
                result += self._data[row][col]+''
            result += "\n"
        return result

c = Grid(3, 3, 0)
# c[1][1] = -1
print(c.find_negative())