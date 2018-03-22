# -*- coding:utf-8 _*-  
""" 
@author:crd
@file: emd.py 
@time: 2018/02/05 
"""
import scipy.signal as signal
import pandas as pd


class MyEmd():
    def __init__(self, filepath="data.csv"):
        self.data = pd.read_csv(filepath)

    def getspline(self):
        N=len(self.data)
        [p,l]=signal.find_peaks_cwt(self.data)