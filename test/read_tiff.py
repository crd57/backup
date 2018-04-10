# -*- coding:utf-8 _*-  
""" 
@author:crd
@file: read_tiff.py 
@time: 2018/03/13 
"""
import gdal
import pandas as pd
import numpy as np


def readTif(fileName):
    """

    :param fileName: file name
    :return: im_data = data set
             im_bands = The numbers of bands
             im_geotrans = Affine matrix information
             im_pro = Projection information
    """
    dataset = gdal.Open(fileName)
    if dataset == None:
        print(fileName + "文件无法打开")
        return
    im_width = dataset.RasterXSize  # 栅格矩阵的列数
    im_height = dataset.RasterYSize  # 栅格矩阵的行数
    im_bands = dataset.RasterCount  # 波段数
    im_data = dataset.ReadAsArray(0, 0, im_width, im_height)  # 获取数据
    im_geotrans = dataset.GetGeoTransform()  # 获取仿射矩阵信息
    im_proj = dataset.GetProjection()  # 获取投影信息
    return [im_data,im_bands, im_geotrans, im_proj]


def writeTiff(im_data, im_width, im_height, im_bands, im_geotrans, im_proj, path):
    """

    :param im_data: data set
    :param im_width: The numbers of rows
    :param im_height: The numbers of lines
    :param im_bands: The numbers of bands
    :param im_geotrans: Affine matrix information
    :param im_proj: Projection information
    :param path: path
    :return: dataset
    """
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32

    if len(im_data.shape) == 3:
        im_bands, im_height, im_width = im_data.shape
    else:
        im_bands, (im_height, im_width) = 1, im_data.shape
        # 创建文件
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(path, im_width, im_height, im_bands, datatype)
    if dataset is not None:
        dataset.SetGeoTransform(im_geotrans)  # 写入仿射变换参数
        dataset.SetProjection(im_proj)  # 写入投影
    for i in range(im_bands):
        dataset.GetRasterBand(i + 1).WriteArray(im_data[i])
    del dataset


def red_rio(filename, Class=1):
    data = pd.read_csv(filename)

    prob_y = np.zeros(len(data))
    prob_y[:] = Class
    prob_y = np.int64(prob_y)
    prob_y = list(prob_y)


# [im_data,im_bands, im_geotrans, im_proj] = readTif('qinghaihu.tif')

# data3dd=np.dstack((im_1987,im_2004))