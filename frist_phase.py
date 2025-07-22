#%%
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import coptpy as cp
from coptpy import COPT#COPT求解器
import numpy as np
import elkai
from typing import List, Tuple, Dict, Optional, Any
import copy
from data_read import data_reader
import random
from IntraVND import IntraVND
#%%
class FristPhase:
    def __init__(self, scenario_nums, customer_ttmatrices, demands_customer, vehicles_nums, vehicles_capacity, ttmatrices, p: int = 3, NC: int = 3, MP: int = 10):
        """
        scenario_nums: 场景数量
        customer_ttmatrices: 客户间时间矩阵
        demands_customer: 客户需求量
        vehicles_nums: 车辆数量
        vehicles_capacity: 车辆容量
        ttmatrices: 仓库-客户时间矩阵
        p: 最大可开启的仓库数量
        NC: 每次扰动的仓库数量
        MP: 最多输出的配置数量
        """
        self.p = p
        self.NC = NC
        self.MP = MP
        self.depot = [0, 1, 2, 3, 4]
        self.Costs: Dict[str, Dict[Tuple[int, int], float]] = {}
        self.Routes: Dict[str, Dict[Tuple[int, int], List[int]]] = {}
        self.cl: Dict[str, List[List[int]]] = {}
        self.scenario_nums = scenario_nums
        self.customer_ttmatrices = customer_ttmatrices
        self.demands_customer = demands_customer
        self.vehicles_nums = vehicles_nums
        self.vehicles_capacity = vehicles_capacity
        self.ttmatrices = ttmatrices
        self.scenarios = [f'scenario_{i}' for i in range(1, self.scenario_nums + 1)]
        self._init_clusters_and_costs()

    def vrp(self, matrices_3d: np.ndarray) -> List[List[int]]:
        """
        对客户进行TSP求解，返回每个场景的客户访问顺序（去除depot）。
        入参：
        matrices_3d - 三维距离矩阵（场景数×时间矩阵）

        出参：
        giant_tour - 每个场景的客户路径列表
        """
        giant_tour = []
        for i in range(len(matrices_3d)):
            customer = elkai.DistanceMatrix(matrices_3d[i])#LKH-3算法
            giant_tour.append(customer.solve_tsp())
        for sublist in giant_tour:
            sublist.pop()#原问题是开放式路径
        return giant_tour

    def cluster_giant_tour(self, giant_tour: List[int], demands: List[int], travel_time_matrix: np.ndarray, K: int, Q: float, theta: float = 1000.0, initial_customer: Optional[int] = None) -> List[List[int]]:
        """
        将giant_tour划分为K个容量受限的簇（车辆路径），用于后续分配。
        入参：
            giant_tour - 客户访问顺序
            demands - 每个客户的需求量
            travel_time_matrix - 客户间旅行时间
            K - 车辆数量
            Q - 车辆最大容量
            theta - 容量惩罚系数
            initial_customer - 起始客户编号

        出参：
        簇划分结果，每个簇为客户编号列表
        """
        def calculate_cluster_load(cluster: List[int]) -> float:#簇容量计算
            return sum(demands[customer] for customer in cluster)
        # 计算将客户插入簇中某个位置后，路径旅行时间的变化量
        def calculate_travel_time_variation(cluster: List[int], customer: int, position: int) -> float:
            if len(cluster) == 0:
                return 0.0
            if position == 0:
                if len(cluster) > 0:
                    return travel_time_matrix[customer][cluster[0]]
                return 0.0
            elif position >= len(cluster):
                return travel_time_matrix[cluster[-1]][customer]
            else:
                prev_customer = cluster[position - 1]
                next_customer = cluster[position]
                old_time = travel_time_matrix[prev_customer][next_customer]
                new_time = (travel_time_matrix[prev_customer][customer] + travel_time_matrix[customer][next_customer])
                return new_time - old_time
        # 寻找将客户插入簇的最佳位置（使旅行时间增量最小）
        def find_best_insertion_position(cluster: List[int], customer: int) -> Tuple[int, float]:
            if len(cluster) == 0:
                return 0, 0.0
            best_position = 0
            min_increase = float('inf')
            for pos in range(len(cluster) + 1):
                increase = calculate_travel_time_variation(cluster, customer, pos)
                if increase < min_increase:
                    min_increase = increase
                    best_position = pos
            return best_position, min_increase
        # 按照giant_tour顺序，依次分配客户到各簇，直到达到容量上限
        def partition_procedure(initial_customer: int) -> List[List[int]]:
            clusters = []  # 存储所有簇
            tour_length = len(giant_tour)
            try:
                start_idx = giant_tour.index(initial_customer)  # 从指定客户开始
            except ValueError:
                start_idx = 0
            current_idx = start_idx
            processed = set()  # 已分配客户集合
            while len(processed) < tour_length:
                current_cluster = []  # 当前簇
                current_load = 0.0    # 当前簇已分配需求量
                # 持续向当前簇分配客户，直到容量上限或全部分配完毕
                while current_load <= Q and len(processed) < tour_length:
                    customer = giant_tour[current_idx]
                    if customer not in processed:
                        if current_load + demands[customer] <= Q:
                            current_cluster.append(customer)
                            current_load += demands[customer]
                            processed.add(customer)
                        else:
                            break  # 当前客户加入会超容量，结束本簇
                    current_idx = (current_idx + 1) % tour_length
                    # 如果回到起点且本簇还没分配任何客户，跳出
                    if current_idx == start_idx and len(current_cluster) == 0:
                        break
                if current_cluster:
                    clusters.append(current_cluster)
                # 跳过已分配客户，寻找下一个未分配客户
                while len(processed) < tour_length and giant_tour[current_idx] in processed:
                    current_idx = (current_idx + 1) % tour_length
            return clusters
        # 对于客户数较多的簇，按最大旅行时间断点进行二分
        def splitting_procedure(clusters: List[List[int]]) -> List[List[int]]:
            result_clusters = []
            for cluster in clusters:
                if len(cluster) <= 1:
                    result_clusters.append(cluster)
                    continue
                max_time = -1
                split_idx = -1
                # 找到簇内相邻客户间最大旅行时间的断点
                for i in range(len(cluster) - 1):
                    travel_time = travel_time_matrix[cluster[i]][cluster[i + 1]]
                    if travel_time > max_time:
                        max_time = travel_time
                        split_idx = i
                if split_idx != -1:
                    cluster1 = cluster[:split_idx + 1]
                    cluster2 = cluster[split_idx + 1:]
                    result_clusters.extend([cluster1, cluster2])
                else:
                    result_clusters.append(cluster)
            return result_clusters
        # 如果簇数多于车辆数，则将最小负载簇的客户重新分配到其他簇
        def repair_procedure(clusters: List[List[int]]) -> List[List[int]]:
            while len(clusters) > K:
                min_load = float('inf')
                min_cluster_idx = -1
                # 找到负载最小的簇
                for i, cluster in enumerate(clusters):
                    load = calculate_cluster_load(cluster)
                    if load < min_load:
                        min_load = load
                        min_cluster_idx = i
                if min_cluster_idx == -1:
                    break
                customers_to_reassign = clusters[min_cluster_idx].copy()
                del clusters[min_cluster_idx]
                # 依次将该簇的客户分配到其他簇（插入位置和容量惩罚最优）
                for customer in customers_to_reassign:
                    best_cluster_idx = -1
                    best_position = -1
                    min_score = float('inf')
                    for j, cluster in enumerate(clusters):
                        current_load = calculate_cluster_load(cluster)
                        customer_demand = demands[customer]
                        capacity_penalty = max(0, (current_load + customer_demand) - Q)
                        position, delta_time = find_best_insertion_position(cluster, customer)
                        score = delta_time + theta * capacity_penalty
                        if score < min_score:
                            min_score = score
                            best_cluster_idx = j
                            best_position = position
                    if best_cluster_idx != -1:
                        clusters[best_cluster_idx].insert(best_position, customer)
            return clusters
        # 通过交换不同簇的客户，进一步优化簇的可行性和均衡性
        def swapping_procedure(clusters: List[List[int]]) -> List[List[int]]:
            improved = True
            max_iterations = 100
            iteration = 0
            while improved and iteration < max_iterations:
                improved = False
                iteration += 1
                for i in range(len(clusters)):
                    for j in range(i + 1, len(clusters)):
                        cluster_i = clusters[i]
                        cluster_j = clusters[j]
                        if len(cluster_i) == 0 or len(cluster_j) == 0:
                            continue
                        for k, customer_k in enumerate(cluster_i):
                            for l, customer_l in enumerate(cluster_j):
                                load_i = calculate_cluster_load(cluster_i)
                                load_j = calculate_cluster_load(cluster_j)
                                new_load_i = load_i - demands[customer_k] + demands[customer_l]
                                new_load_j = load_j - demands[customer_l] + demands[customer_k]
                                # 交换后两簇均不超容量则交换
                                if new_load_i <= Q and new_load_j <= Q:
                                    clusters[i][k] = customer_l
                                    clusters[j][l] = customer_k
                                    improved = True
                                    break
                            if improved:
                                break
                        if improved:
                            break
                    if improved:
                        break
            return clusters
        # 如果未指定起始客户，则随机选择一个
        if initial_customer is None:
            initial_customer = random.choice(giant_tour)
        # 1. 按顺序分配客户到各簇
        clusters = partition_procedure(initial_customer)
        # 2. 若簇数不足，按最大断点二分
        if len(clusters) < K:
            clusters = splitting_procedure(clusters)
        # 3. 若簇数过多，修复合并
        if len(clusters) > K:
            clusters = repair_procedure(clusters)
        # 4. 交换优化
        clusters = swapping_procedure(clusters)
        # 5. 去除空簇
        clusters = [cluster for cluster in clusters if len(cluster) > 0]
        return clusters

    def calculate_allocation_cost(self, depot_i: int, cluster_j: List[int], travel_time_matrix) -> Tuple[float, List[int]]:
        """
        计算单个仓库到单个簇的分配成本（路径总延迟），并返回优化后的路径。
        入参：
        depot_i - 仓库编号，cluster_j - 簇客户编号列表，travel_time_matrix - 距离矩阵
        出参：
        总延迟, 优化路径
        """
        initial_path = [depot_i] + cluster_j
        optimizer = IntraVND(travel_time_matrix)
        optimized_path, total_latency = optimizer.optimize_route(initial_path)
        return total_latency, optimized_path

    def compute_all_allocation_costs(self, depots: List[int], clusters: List[List[int]], travel_time_matrix) -> Tuple[Dict, Dict]:
        """
        计算所有仓库-簇组合的分配成本和路径。
        入参：
        depots - 仓库编号列表，clusters - 簇划分，travel_time_matrix - 距离矩阵
        出参：
        分配成本字典, 路径字典
        """
        allocation_costs = {}
        route_info = {}
        for depot_i in depots:
            for cluster_idx, cluster_j in enumerate(clusters):
                cost, route = self.calculate_allocation_cost(depot_i, cluster_j, travel_time_matrix)
                allocation_costs[(depot_i, cluster_idx)] = cost
                route_info[(depot_i, cluster_idx)] = route
        return allocation_costs, route_info

    def opt_config(self, depot: List[int], cl: List[List[int]], Costs: Dict[Tuple[int, int], float], p: int, force_close: Optional[int] = None) -> Dict[str, Any]:
        """
        求解单场景下的仓库-簇分配优化问题，基于COPT求解器。
        入参：
            depot - 仓库编号列表
            cl - 簇划分
            Costs - 分配成本字典
            p - 最大可开启仓库数
            force_close - 强制关闭的仓库编号（可选）
        出参：
        包含分配矩阵、开启仓库、总延迟等信息的字典
        """
        env = cp.Envr()
        model = env.createModel('main configuration')
        Z = model.addVars(len(depot), vtype=COPT.BINARY, nameprefix="Z")
        A = model.addVars(len(depot), len(cl), vtype=COPT.BINARY, nameprefix="A")
        model.setObjective(cp.quicksum(A[i, j] * Costs[(i, j)] for i in range(len(depot)) for j in range(len(cl))), COPT.MINIMIZE)
        for j in range(len(cl)):
            model.addConstr(cp.quicksum(A[i, j] for i in range(len(depot))) == 1)
        model.addConstr(cp.quicksum(Z[i] for i in depot) <= p)
        for i in range(len(depot)):
            model.addConstr(Z[i] <= cp.quicksum(A[i, j] for j in range(len(cl))))
            for j in range(len(cl)):
                model.addConstr(A[i, j] <= Z[i])
        if force_close is not None:
            model.addConstr(Z[force_close] == 0)
        model.solve()
        A_sol: Dict[Tuple[int, int], int] = {(i, j): round(A[i, j].x) for i in range(len(depot)) for j in range(len(cl))}
        Z_sol: List[int] = [round(Z[i].x) for i in range(len(depot))]
        latency: float = model.objval
        open_depots: List[int] = [i for i, z in enumerate(Z_sol) if z == 1]
        return {
            "A": A_sol,
            "Z": Z_sol,
            "open_depots": open_depots,
            "latency": latency
        }

    def process_scenario(self, config_dict: Dict[Tuple[int, ...], Dict[str, Any]], depot: List[int], cl: List[List[int]], Costs: Dict[Tuple[int, int], float], p: int, NC: int) -> None:
        """
        处理单个场景，更新全局配置字典，包括扰动部分仓库后的配置。
        参数：
            config_dict - 全局配置字典（原地修改）
            depot - 仓库编号列表
            cl - 簇划分
            Costs - 分配成本字典
            p - 最大可开启仓库数
            NC - 扰动仓库数量
        """
        result = self.opt_config(depot, cl, Costs, p)
        key = tuple(sorted(result["open_depots"]))
        if key not in config_dict:
            result["selection_count"] = 1
            config_dict[key] = result
        else:
            config_dict[key]["selection_count"] += 1
            if result["latency"] < config_dict[key]["latency"]:
                result["selection_count"] = config_dict[key]["selection_count"]
                config_dict[key] = result
        open_depots = result["open_depots"]
        cluster_counts = {i: sum(1 for j in range(len(cl)) if result["A"][(i, j)] == 1) for i in open_depots}
        sorted_open_depots = sorted(open_depots, key=lambda i: cluster_counts[i])
        top_nc_depots = sorted_open_depots[:NC]
        for depot_to_close in top_nc_depots:
            disturbed = self.opt_config(depot, cl, Costs, p, force_close=depot_to_close)
            key = tuple(sorted(disturbed["open_depots"]))
            if key not in config_dict:
                disturbed["selection_count"] = 1
                config_dict[key] = disturbed
            else:
                config_dict[key]["selection_count"] += 1
                if disturbed["latency"] < config_dict[key]["latency"]:
                    disturbed["selection_count"] = config_dict[key]["selection_count"]
                    config_dict[key] = disturbed

    def solve_configurations(self) -> List[Dict[str, Any]]:
        """
        处理所有场景，输出最终的最优配置列表（Start）。
        返回：每个配置包含开启仓库、分配路径、延迟、被选次数等信息
        """
        config_dict = {}
        for scen in self.scenarios:
            self.process_scenario(
                config_dict=config_dict,
                depot=self.depot,
                cl=self.cl[scen],
                Costs=self.Costs[scen],
                p=self.p,
                NC=self.NC
            )
        Configs = list(config_dict.values())
        Configs.sort(key=lambda x: (-x["selection_count"], x["latency"]))
        Configs_new = []
        for conf in Configs:
            open_depots = conf["open_depots"]
            A = conf["A"]
            clist = self.cl[self.scenarios[0]] if len(self.cl) > 0 else []
            depot_paths = {}
            for depot_id in open_depots:
                depot_paths[depot_id] = []
                for j in range(len(clist)):
                    if A.get((depot_id, j), 0) == 1:
                        path = [depot_id] + clist[j]
                        depot_paths[depot_id].append(path)
            Configs_new.append({
                "open_depots": open_depots,
                "depot_paths": depot_paths,
                "latency": conf["latency"],
                "selection_count": conf["selection_count"]
            })
        return Configs_new[:min(self.MP, len(Configs_new))]

    def _init_clusters_and_costs(self):
        """
        初始化所有场景的簇划分和分配成本。
        """
        for i in range(self.scenario_nums):
            clusters_i = self.cluster_giant_tour(
                self.vrp(self.customer_ttmatrices)[i],
                self.demands_customer,
                self.customer_ttmatrices[i],
                self.vehicles_nums,
                self.vehicles_capacity,
                theta=1000.0,
                initial_customer=None
            )
            clusters_i = [[x + 5 for x in sublist] for sublist in clusters_i]
            self.cl[f"scenario_{i+1}"] = clusters_i
        for i in range(self.scenario_nums):
            scenario = f"scenario_{i+1}"
            cost, route = self.compute_all_allocation_costs(self.depot, self.cl[scenario], self.ttmatrices[i])
            self.Costs[scenario] = cost
            self.Routes[scenario] = route

    def get_start(self):
        """
        获取最终的Start结果（最优配置列表）。
        返回：List[Dict]，每个元素为一个配置方案
        """
        return self.solve_configurations()

# 用法示例
if __name__ == "__main__":
    Q = data_reader(r'C:\Users\17862\OneDrive\LLRP\LLRP-STT-main\LLRP-STT-main\coord20-5-1-E20-LOGN.dat')
    Q.read_ttmatrices()
    Q.read_cus_ttmatrices()
    phase = FristPhase(Q.scenario_nums, Q.customer_ttmatrices, Q.demands_customer, Q.vehicles_nums, Q.vehicles_capacity, Q.ttmatrices)
    Start = phase.get_start()
    print(Start)