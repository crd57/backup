# -*- coding:utf-8 _*-  
""" 
@author:crd
@file: die.py 
@time: 2018/02/05 
"""
from random import randint


class Die:

    """表示一个骰子类"""
    def __init__(self, num_side=6):
        """默认骰子有六个面"""
        self.num_sides = num_side

    def roll(self):
        """返回一个骰子的随机值"""
        return randint(1, self.num_sides)


