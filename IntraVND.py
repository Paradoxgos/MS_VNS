#%%
import numpy as np
import random
from typing import List, Tuple, Dict, Optional
import copy
#%%
class IntraVND:#没有考虑原文中惩罚量a
    def __init__(self, travel_time_matrix):
        self.travel_time_matrix = travel_time_matrix
    
    def calculate_total_latency(self, path: List[int]) -> float:
        """
        计算路径的总延迟时间
        延迟 = 客户的到达时间（从仓库出发开始累计）
        """
        if len(path) < 2:
            return 0
        
        total_latency = 0
        # current_time = 0
        # depot = path[0]  # 第一个节点是仓库
        
        for i in range(len(path) - 1):
            # depot到第一个客户的所造成的latency
            latency_d = (len(path)-1-i)*self.travel_time_matrix[path[i]][path[i+1]]
            total_latency += latency_d
            
            # # 如果到达的是客户节点，累加延迟
            # if path[i+1] != depot:
            #     latency_c += current_time  # 客户的延迟 = 到达时间
            #     current_time += self.service_times[path[i+1]]
        
        return total_latency
    
    def insertion_neighborhood(self, path: List[int]) -> Tuple[List[int], float]:
        """
        Insertion邻域：移除一个客户并在另一个位置重新插入，基于最佳改进原则，即探索领域内的所有解
        """
        best_path = path.copy()
        best_cost = self.calculate_total_latency(path)
        
        # 只对客户节点进行操作（跳过仓库节点）
        for i in range(1, len(path) - 1):  # 跳过起始和结束的仓库
            customer = path[i]
            
            # 创建移除客户i后的路径
            temp_path = path[:i] + path[i+1:]
            
            # 尝试在所有可能位置插入客户
            for j in range(1, len(temp_path)):
                new_path = temp_path[:j] + [customer] + temp_path[j:]
                new_cost = self.calculate_total_latency(new_path)
                
                if new_cost < best_cost:
                    best_path = new_path
                    best_cost = new_cost
        
        return best_path, best_cost
    
    def swap_neighborhood(self, path: List[int]) -> Tuple[List[int], float]:
        """
        Swap邻域：交换路径中两个客户的位置
        """
        best_path = path.copy()
        best_cost = self.calculate_total_latency(path)
        
        # 只交换客户节点（跳过仓库节点）
        for i in range(1, len(path) ):
            for j in range(i + 1, len(path) ):
                # 创建新路径，交换位置i和j的客户
                new_path = path.copy()
                new_path[i], new_path[j] = new_path[j], new_path[i]
                
                new_cost = self.calculate_total_latency(new_path)
                
                if new_cost < best_cost:
                    best_path = new_path
                    best_cost = new_cost
        
        return best_path, best_cost
    
    def two_opt_neighborhood(self, path: List[int]) -> Tuple[List[int], float]:
        """
        2-Opt邻域：反转路径中两个点之间的子序列
        """
        best_path = path.copy()
        best_cost = self.calculate_total_latency(path)
        
        for i in range(1, len(path) - 1):  # 跳过仓库节点
            for j in range(i + 1, len(path) ):
                # 创建2-opt交换后的新路径
                # 反转从i到j的子序列
                new_path = path[:i] + path[i:j+1][::-1] + path[j+1:]
                
                new_cost = self.calculate_total_latency(new_path)
                
                if new_cost < best_cost:
                    best_path = new_path
                    best_cost = new_cost
        
        return best_path, best_cost
    
    def optimize_route(self, initial_path: List[int]) -> Tuple[List[int], float]:
        """
        使用VND（Variable Neighborhood Descent）优化路径
        """
        current_path = initial_path.copy()
        current_cost = self.calculate_total_latency(current_path)
        
        # 定义邻域操作的顺序
        neighborhoods = [
            self.insertion_neighborhood,
            self.swap_neighborhood,
            self.two_opt_neighborhood
        ]
        
        improved = True
        iteration = 0
        max_iterations = 1000  
        
        while improved and iteration < max_iterations:
            improved = False
            iteration += 1
            
            for neighborhood in neighborhoods:
                new_path, new_cost = neighborhood(current_path)
                
                if new_cost < current_cost:
                    current_path = new_path
                    current_cost = new_cost
                    improved = True
                    break  # 找到改进就重新开始
        
        return current_path, current_cost
# %%
