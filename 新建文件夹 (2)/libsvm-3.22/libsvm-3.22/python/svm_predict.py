# -*- coding:utf-8 _*-  
""" 
@author:crd
@file: svm_predict.py 
@time: 2018/03/12 
"""

from svm import svm_problem
from svm import svm_parameter
from svmutil import svm_train
from svmutil import svm_predict
import pandas as pd
import numpy as np


class pedict:

    def load_data(filename, seq_len=3):
        """

        :param filename :input file's name
        :param seq_len:input the length of sequence
        :return:label and train data
        """
        data = pd.read_csv(filename)
        data = np.float64(data).reshape([len(data)])
        prob_y = list(data[seq_len:])
        prob_x = []
        for i in range(len(prob_y)):
            xi = {}
            for k in range(seq_len):
                xi[k + 1] = data[i + k]
            prob_x += [xi]
        return [prob_y, prob_x, data]

    def bestSVC(y, x, cmin=-8, cmax=8, gmin=-8, gmax=8, pmin=-8, pmax=8, v=5, cstep=0.8, gstep=0.8, pstep=0.8, t=2):
        """

        :param y: label
        :param x: train data
        :param cmin:The minimum value of c
        :param cmax:The maximum value of c
        :param gmin:The minimum value of g
        :param gmax:The maximum value of g
        :param pmin:The minimum value of p
        :param pmax:The maximum value of p
        :param v:Number of folds
        :param cstep:c steps
        :param gstep:g steps
        :param pstep:p steps
        :param t:set type of kernel function
        :return:The best c value,The best g value,The best p value
        """
        cnums = np.arange(cmin, cmax, cstep)
        gnums = np.arange(gmin, gmax, gstep)
        pnums = np.arange(pmin, pmax, pstep)
        data_list = np.meshgrid(cnums, gnums, pnums)
        [m, n, q] = data_list[0].shape
        bestc = 0
        bestg = 0
        bestp = 0
        MSE = float('inf')
        basenum = 2
        eps = 10 ** (-4)
        prob = svm_problem(y, x)
        for i in range(1, m):
            for j in range(1, n):
                for r in range(1, q):
                    param = svm_parameter(
                        '-v ' + str(v) + ' -c ' + str(basenum ** (data_list[0][i][j][r])) + ' -g ' + str(
                            basenum ** (data_list[1][i][j][r])) + ' -p ' + str(
                            basenum ** (data_list[2][i][j][r])) + ' -t ' + str(t) + ' -s 3')
                    m = svm_train(prob, param)
                    if (m < MSE):
                        MSE = m
                        bestc = basenum ** (data_list[0][i][j][r])
                        bestg = basenum ** (data_list[1][i][j][r])
                        bestp = basenum ** (data_list[2][i][j][r])
                    if ((abs(m - MSE) <= eps) & (bestc > (basenum ** (data_list[0][i][j][r])))):
                        MSE = m
                        bestc = basenum ** (data_list[0][i][j][r])
                        bestg = basenum ** (data_list[1][i][j][r])
                        bestp = basenum ** (data_list[2][i][j][r])

        return [bestc, bestg, bestp]


def svmprdict(y, x, data, bestc=0.2, bestg=0.2, bestp=0.2, years=5):
    """

    :param y: train label
    :param x: train data
    :param data: sequence data
    :param bestc: The best c value
    :param bestg: The best g value
    :param bestp: The best p value
    :param years: The year to be forecasted
    :return: Predictive value
    """
    prob = svm_problem(y, x)
    param = svm_parameter('-c ' + str(bestc) + ' -g ' + str(bestg) + ' -p ' + str(bestp))
    model = svm_train(prob, param)
    testing_instance_matrix = []
    data_p = data[(len(data) - len(x[0])):]
    xi = {}
    for k in range(len(data_p)):
        xi[k + 1] = data_p[k]
    testing_instance_matrix += [xi]
    p_labels, p_acc, p_vals = svm_predict(([0] * len(testing_instance_matrix)), testing_instance_matrix, model)
    predict_label = []
    predict_label += [p_labels]
    for i in range(1, years, 1):
        for k in range(len(testing_instance_matrix[0]) - 1):
            testing_instance_matrix[0][k + 1] = testing_instance_matrix[0][k + 2]
        testing_instance_matrix[0][len(testing_instance_matrix[0])] = p_labels[0]
        p_labels, p_acc, p_vals = svm_predict(([0] * len(testing_instance_matrix)), testing_instance_matrix, model)
