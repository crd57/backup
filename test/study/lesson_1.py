# -*- coding:utf-8 _*-  
""" 
@author:crd
@file: lesson_1.py 
@time: 2018/03/18 
"""

import numpy as np

"""
class sphere:
    def __init__(self, R):
        self.r = R

    def diameter(self):
        Diameter = 2 * self.r
        return Diameter

    def circumference(self):
        C = self.r * 2 * np.pi
        return C

    def Surface_area(self):
        S = self.r * self.r * 4 * np.pi
        return S

    def volume(self):
        V = self.r ** 3 * np.pi * 4 / 3
        return V

def pay(money, nomal, plus):
    Pay_plus = plus * 1.5 * money
    Pay = money * nomal + Pay_plus
    return Pay


a = pay(10000, 8, 16)
"""
Pi = 0
for i in range(1,1000):
    Pi += (-1) ** (i+1) * 1/(2 * i - 1)
pi = 4 * Pi