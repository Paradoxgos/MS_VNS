#%%
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import pandas as pd
import numpy as np

#%%
def read_dat(file_path):
    '''
    入参:
    file_path (str): dat文件路径
    
    出参:
    matrices_dict (dict): 包含所有矩阵的字典，键为scenario_序号，值为array矩阵
    '''
    # 初始化存储所有矩阵的字典
    matrices_dict = {}
    # 初始化存储当前矩阵的临时列表
    current_matrix = []
    # 定义矩阵维度为 25x25
    matrix_size = 15
    # 情景序号计数器
    scenario_counter = 1
    # 以只读模式打开文件并读取所有行    
    with open(file_path, 'r') as file:
        lines = file.readlines()
    # 初始化行索引    
    i = 0
    # 逐行处理文件内容
    while i < len(lines):
        # 移除每行的首尾空白字符
        line = lines[i].strip()
        # 跳过空行或以负号开头的行（如文件末尾的 -111）        
        if not line or line.startswith('-'):
            i += 1
            continue
        # 尝试处理矩阵行数据            
        try:
            # 将行按空格分割并转换为浮点数列表
            row = [float(x) for x in line.split()]
            # 如果行包含 25 个元素，添加到当前矩阵            
            if len(row) == matrix_size:
                current_matrix.append(row)
            # 如果收集了 25 行，完成一个 25x25 矩阵                
            if len(current_matrix) == matrix_size:
                # 将当前矩阵存入字典，键为情景序号
                matrices_dict[f"scenario_{scenario_counter}"] = np.array(current_matrix)
                scenario_counter += 1
                # 清空当前矩阵，准备读取下一个矩阵
                current_matrix = []                
        except ValueError:
            # 跳过无法转换为浮点数的行（如头部数据）
            pass  
        i += 1
    return matrices_dict
# %%
