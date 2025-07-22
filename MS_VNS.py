#%%
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import numpy as np
import copy
from data_read import data_reader
from frist_phase import FristPhase
from second_phase import SecondPhase

def compute_CVaR(latency_list, alpha=0.95):
    """
    计算CVaR（条件风险价值）：即超过VaR的损失的均值
    """
    latencies = np.array(latency_list)
    var = np.quantile(latencies, alpha)
    cvar = latencies[latencies >= var].mean()
    return cvar

def MS_VNS(Start, Q, Neigh=3, iter_max=100, iter_RR=50, SK=3, tryVNS=10, alpha=0.95):
    """
    多场景VNS主流程
    入参：
        Start: 第一阶段得到的所有depot配置（列表，每个元素是一个解）
        Q: 数据读取实例，包含所有场景的ttmatrices
        其它参数同VNS
        alpha: CVaR置信水平
    出参：
        最优depot配置、其CVaR、所有配置的CVaR列表
    """
    all_results = []
    for s_0_z in Start:
        scenario_latencies = []
        scenario_solutions = []
        for w, travel_time_matrix in enumerate(Q.ttmatrices):
            # update
            SecondPhase.update(s_0_z, travel_time_matrix, w+1)
            # 针对该场景做VNS
            best_sol = SecondPhase.VNS(s_0_z, travel_time_matrix, Neigh, iter_max, iter_RR, SK, tryVNS, w+1)
            scenario_latencies.append(SecondPhase.f(best_sol, w+1))
            scenario_solutions.append(best_sol)
        cvar = compute_CVaR(scenario_latencies, alpha)
        all_results.append({
            "depot_config": s_0_z,
            "scenario_latencies": scenario_latencies,
            "scenario_solutions": scenario_solutions,
            "CVaR": cvar
        })
    best_result = min(all_results, key=lambda x: x["CVaR"])
    return best_result, all_results
#%%
if __name__ == "__main__":
    # 数据准备
    Q = data_reader(r'C:\Users\17862\OneDrive\LLRP\framework\data\IN\LLRP-STT-main\coord10-5-1-E20-LOGN.dat')
    Q.read_ttmatrices()
    Q.read_cus_ttmatrices()
    phase = FristPhase(Q.scenario_nums, Q.customer_ttmatrices, Q.demands_customer, Q.vehicles_nums, Q.vehicles_capacity, Q.ttmatrices)
    Start = phase.get_start()

    # 多场景VNS
    best_result, all_results = MS_VNS(
        Start, Q,
        Neigh=3,
        iter_max=300,
        iter_RR=0.7,
        SK=15,
        tryVNS=100,
        alpha=0.95
    )
    print("最优CVaR:", best_result["CVaR"])
    print("最优depot配置:", best_result["depot_config"])
    print("所有场景下的延迟:", best_result["scenario_latencies"])
