#%%
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from data_trans import read_dat
import pandas as pd
import numpy as np
import elkai#LKH-3算法包
import random
import copy
from typing import List, Tuple, Dict, Optional
#%%
class data_reader(object):
    def __init__(self, file_path):
        self.file_path = file_path

        self.customers_nums = 10# 客户数量
        self.depot_nums = 5# 仓库数量
        self.vehicles_nums = 3# 车辆数量
        self.scenario_nums = 20# 场景数量
        self.vehicles_capacity = 70# 车辆容量
        self.coord_depots = [(6, 7),(19, 44),(37, 23),(35, 6),(5, 8)]# 仓库坐标
        self.coord_customers = [(26, 20),(20, 33),(15, 46),(20, 26),(17, 19),(15, 12),(5, 30),(13, 40),(38, 5),(9, 40)]# 客户坐标
        self.demands_customer = [16,18,15,11,18,16,15,15,15,16]# 客户需求
        self.ttmatrices = np.array([])  # 包含所有场景的随机旅行时间矩阵
        self.customer_ttmatrices = np.array([])  # 仅包含客户的所有场景的随机旅行时间矩阵

    def read_ttmatrices(self):
        '''
        入参:
        file_path (str): 文件路径

        出参:
        包含所有场景的随机旅行时间矩阵（三维数组——场景，时间矩阵）
        '''
        file_path = self.file_path
        self.ttmatrices = np.array(list(read_dat(file_path).values()))
        return 
    
    def read_cus_ttmatrices(self):
        """
        去掉三维数组中每个矩阵的前n行和前n列。

        入参:
        matrix_3d (numpy.ndarray)

        出参:
        numpy.ndarray: 去掉前n行和前n列后的三维数组 （场景，时间矩阵）
        """
        # 确保 n 不超过矩阵的行数或列数
        if self.depot_nums >= self.ttmatrices.shape[1] or self.depot_nums >= self.ttmatrices.shape[2]:
            raise ValueError("n 不能大于矩阵的行数或列数")

        # 去掉每个矩阵的前n行和前n列
        self.customer_ttmatrices = self.ttmatrices[:, self.depot_nums:, self.depot_nums:]
        return 
# %%
