#%%
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import coptpy as cp
from coptpy import COPT
import numpy as np
import elkai
from typing import List, Tuple, Dict, Optional, Any
import copy
from data_read import data_reader
import random
from IntraVND import IntraVND
from frist_phase import FristPhase

class SecondPhase:

    def update(s_0_z, travel_time_matrix, scenario_id):
        """
        更新配置的延迟成本
        入参:
        s_0_z: 解决方案字典，包含open_depots和depot_paths
        travel_time_matrix: 旅行时间矩阵

        出参:
        total_latency: 总延迟成本
        """
        total_latency = 0.0
        for depot_id in s_0_z["open_depots"]:
            depot_routes = s_0_z["depot_paths"][depot_id]
            depot_latency = 0.0
            for route in depot_routes:
                route_latency = IntraVND(travel_time_matrix).calculate_total_latency(route)
                depot_latency += route_latency
            total_latency += depot_latency
        s_0_z[f"latency_z_{scenario_id}"] = total_latency
        return

    def VND(s_c, s_bf_z, travel_time_matrix, scenario_id):#基于整个配置的VND，IntraVND是基于单条路径的
        """
        变邻域下降（VND）算法，对配置中的每个d路径进行局部优化。
        入参：
        s_c: 当前解
        s_bf_z: 最优解
        travel_time_matrix: 旅行时间矩阵
        scenario_id: 场景ID
        出参：
        优化后的解
        """
        s_c_cur = copy.deepcopy(s_c)
        latency_total = 0
        for depot_id in s_c["open_depots"]:
            depot_routes = s_c["depot_paths"][depot_id]
            optimized_routes = []
            for route in depot_routes:
                route_cur, latency_cur = IntraVND(travel_time_matrix).optimize_route(route)
                latency_total += latency_cur
                optimized_routes.append(route_cur)
            s_c_cur["depot_paths"][depot_id] = optimized_routes
        s_c_cur[f"latency_z_{scenario_id}"] = latency_total
        return s_c_cur

    def shake(s_c, n, SK, travel_time_matrix):
        """
        对当前解进行扰动（shake），以跳出局部最优。
        入参:
            s_c: 当前解（字典，包含open_depots和depot_paths）
            n: 邻域结构编号（1: 插入, 2: 交换, 3: 2-opt）
            SK: 扰动次数
            travel_time_matrix: 旅行时间矩阵
        出参:
            扰动后的新解
        """
        s_c_new = copy.deepcopy(s_c)
        intra = IntraVND(travel_time_matrix)
        neighborhood_funcs = [
            intra.insertion_neighborhood,
            intra.swap_neighborhood,
            intra.two_opt_neighborhood
        ]
        neighborhood_func = neighborhood_funcs[n-1]
        for _ in range(SK):
            depot_id = random.choice(s_c_new["open_depots"])
            routes = s_c_new["depot_paths"][depot_id]
            if not routes:
                continue
            route_idx = random.randrange(len(routes))
            route = routes[route_idx]
            if len(route) > 2:
                new_route, _ = neighborhood_func(route)
                s_c_new["depot_paths"][depot_id][route_idx] = new_route
        return s_c_new


    def RouteRelocation(s_c, travel_time_matrix):
        """
        路径重定位操作：将一个depot中的一条路径重新分配到另一个depot，并对该路径进行局部优化。
        入参:
            s_c: 当前解（字典，包含open_depots和depot_paths）
            travel_time_matrix: 旅行时间矩阵
        出参:
            新解
        """
        s_c_new = copy.deepcopy(s_c)
        open_depots = s_c_new["open_depots"]
        candidate_depots = [d for d in open_depots if len(s_c_new["depot_paths"][d]) > 1]
        if not candidate_depots:
            return s_c_new
        depot_from = random.choice(candidate_depots)
        routes_from = s_c_new["depot_paths"][depot_from]
        route_idx = random.randrange(len(routes_from))
        route = routes_from.pop(route_idx)
        depot_to_candidates = [d for d in open_depots if d != depot_from]
        if not depot_to_candidates:
            routes_from.insert(route_idx, route)
            return s_c_new
        depot_to = random.choice(depot_to_candidates)
        route_opt, _ = IntraVND(travel_time_matrix).optimize_route(route)
        s_c_new["depot_paths"][depot_to].append(route_opt)
        return s_c_new

    def f(solution, scenario_id):
        """
        只返回当前场景的延迟
        """
        key = f"latency_z_{scenario_id}"
        return solution.get(key, float('inf'))

    def VNS(s_0_z, travel_time_matrix, Neigh=3, iter_max=30, iter_RR=0.7, SK=15, tryVNS=10, scenario_id=1):
        """
        VNS算法主流程
        参数：
            s_0_z: 初始解
            travel_time_matrix: 旅行时间矩阵
            Neigh: 邻域数量
            iter_max: 内层VNS最大迭代次数
            iter_RR: 路径重定位阈值
            SK: shake扰动次数
            tryVNS: 外层VNS最大尝试次数
        返回：
            s_bf_z: 最优解
        """
        s_c = copy.deepcopy(s_0_z)
        s_bf_z = copy.deepcopy(s_0_z)
        s_vnd = SecondPhase.VND(s_c, s_bf_z, travel_time_matrix, scenario_id)
        s_aux = copy.deepcopy(s_vnd)
        for t in range(tryVNS):
            iter = 0
            s_c = copy.deepcopy(s_vnd)
            s_bf_z = copy.deepcopy(s_vnd)
            while iter < iter_max:
                n = 1
                while n <= Neigh:
                    s_c_prime = SecondPhase.shake(s_c, n, SK, travel_time_matrix)
                    s_c_double_prime = SecondPhase.VND(s_c_prime, s_bf_z, travel_time_matrix, scenario_id)
                    n += 1
                    if SecondPhase.f(s_c_double_prime, scenario_id) < SecondPhase.f(s_c, scenario_id):
                        s_c = copy.deepcopy(s_c_double_prime)
                        n = 1
                if iter < iter_RR:
                    s_c = SecondPhase.RouteRelocation(s_c, travel_time_matrix)
                else:
                    s_c = copy.deepcopy(s_bf_z)
                iter += 1
            if SecondPhase.f(s_bf_z, scenario_id) < SecondPhase.f(s_aux, scenario_id):
                s_aux = copy.deepcopy(s_bf_z)
        s_bf_z = copy.deepcopy(s_aux)
        return s_bf_z
