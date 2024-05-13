import pathlib

import math
import numpy as np

from time import time, strftime
import itertools

import pandas as pd
from numpy.random import default_rng

from src.utils import get_memory_usage, print_elapsed

from src.neuroidal_adj import NeuroidalModel as NeuroidalModelAdj
# from src.neuroidal import NeuroidalModel as NeuroidalModel
from src.backend.adjacency import Backend as AdjacencyBackend
# from src.backend.graph_tool import Backend as GraphToolBackend


def print_join_update(model, S_len, H_if, total_if, m_len, m_total):
    print("Current Total Memories:", S_len)
    print("Batch Average Memory Size:", int(m_len / model.H))
    print("Running Average Memory Size:",
          int(m_total / (S_len - model.L)), "\n\n")
    if model.n < 10 ^ 5:
        print("Batch Interference Rate:", round(H_if / model.H, 6))
        print("Running Average Int. Rate:", round(total_if / S_len, 6))


def print_halt_msg(model, S_len, total_if, m_total):
    r_obs = int(m_total / (S_len - model.L))
    r_error = round(((model.r_approx - r_obs) / r_obs) * 100, 2)
    print("-- End of Simulation (Halted) --\n")
    print("Given: n=", model.n, "d=", model.d, "k=", model.k, "k_adj=",
          model.k_adj, "r_approx=", model.r_approx, "START_MEM=", model.L)
    print("we halted Memory Formation at",
          model.F * 100, "% Total Interference.\n")
    print("Empirical Memory Size:", int(m_total / (S_len - model.L)))
    print("Approximation Error of r:", r_error, "%")
    print("Total Average Interference Rate:", round(total_if / S_len, 6))
    print("Capacity:", model.L, "Initial Memories +",
          S_len - model.L, "JOIN Memories.")


def print_memorized_msg(model, S_len, m_total):
    r_obs = int(m_total / (S_len - model.L))
    r_error = round(((model.r_approx - r_obs) / r_obs) * 100, 2)
    print("-- End of Simulation (Completed) --\n")
    print("Given: n=", model.n, "d=", model.d, "k=", model.k, "k_adj=",
          model.k_adj, "r_approx=", model.r_approx, "START_MEM=", model.L)
    print("We memorized all combinations of", model.L, "memories",
          "\n", "with less than", model.F * 100, "% interference.\n")
    print("Empirical Memory Size:", int(m_total / (S_len - model.L)))
    print("Approximation Error of r:", r_error, "%")
    print("Contains:", model.L, "Initial Memories +",
          S_len - model.L, "JOIN Memories.")


def simulate(model, seed, use_QJOIN=True, disjoint=False, two_step=False, fast=True,
             verbose=True, vis = False):
    if vis:
        vis = model.Visualization(model.g)

    start_dir = pathlib.Path('../assets')
    start_time = strftime("%m-%d-%y_%H:%M:%S")
    out_path = (
            start_dir /
            f"{start_time}_neurons_{model.n}_degree_{model.d}_edge-weights_"
            f"{model.k}_replication_{model.r_approx}_start-mems_{model.L}"
    )

    m = 0
    H_if = 0
    m_len = 0
    m_total = 0
    total_if = 0
    first_join = False

    if verbose:
        print("-- Start of Simulation --\n")

    if vis:
        first_join = True
        out_path.mkdir()
        vis.visualize(out_path / f"graph_{len(model.S)}_memories.png")
        vis.visualize_if(out_path / f"graph_{len(model.S)}_if.png")

    rng = default_rng(int(seed))
    model.S = [rng.choice(np.arange(0, model.n - 1), size=model.r_approx)
               for _ in range(model.L)]

    init_pairs = itertools.combinations(range(model.L), 2)
    for A_i, B_i in init_pairs:
        A = list(model.S[A_i])
        B = list(model.S[B_i])

        if use_QJOIN:
            C = model.quick_JOIN(A, B, vis)
        else:
            C = model.JOIN(A, B, disjoint, two_step, fast, vis)
        # print(len(C))
        C_if = model.interference_check(A_i, B_i, C, vis)

        # print(A, B, C, C_if)

        m += 1
        m_len += len(C)
        model.S.append(C)
        m_total += len(C)

        if first_join and vis:
            vis.visualize_start(A, B, C, out_path / f"graph_start.png")

        if m % model.H == 0:
            if verbose:
                model.print_join_update(len(model.S), H_if,
                                        total_if, m_len, m_total)
            H_if = 0
            m_len = 0
            if vis:
                vis.visualize(
                    out_path /
                    f"graph_{len(model.S)}_memories.png"
                )
                vis.visualize_if(
                    out_path /
                    f"graph_{len(model.S)}_if.png"
                )

        if C_if > 0:
            H_if += C_if
            total_if += C_if
            if total_if / len(model.S) > model.F:
                if verbose:
                    model.print_halt_msg(len(model.S), total_if, m_total)
                if vis:
                    vis.visualize(out_path / f"graph_final_memories.png")
                    vis.visualize_if(out_path / f"graph_final_if.png")
                return m_total

    if verbose:
        model.print_memorized_msg(len(model.S), m_total)

    if vis:
        vis.visualize(out_path / f"graph_final_memories.png")
        vis.visualize_if(out_path / f"graph_final_if.png")

    return m_total


def main():
    # For small numbers of n
    # r = (n * k) / d * (2 ** 16 / 10 ** 5)
    # n=500
    # n=10k  d=512 k=16 -> r=234
    # n=100k d=512 k=16 -> r=2134
    # n=1m   d=512 k=16 -> r=23365

    # Interference
    # n=500, d=128, w=16, r=40 L=100 T=0.1 k=2 -> 350
    params = {'n': 500, 'd': 128, 'k': 16, 'r_approx': 40, 't': 0.1, 'k_adj': 2.0, 'L': 100, 'F': 0.1, 'H': 50}

    data = []
    k = 16
    d = 128
    step = 50
    sim_count = 100
    disjoint = True
    for n in range(500, 10000 + step, step):
        data.append([])
        for rep in range(sim_count):
            start = time()
            r_approx = int(n / math.pow(10, 5) * math.pow(2, 16) * k / d)
            params = {'n': n, 'd': d, 'k': k, 'r_approx': r_approx, 't': 0.1, 'k_adj': 2.0, 'L': 100, 'F': 0.1, 'H': 50}
            print(f'n={params["n"]}, d={params["d"]}, k={params["k"]} -> r={r_approx}')

            #
            # seed = time()
            # gtg = GraphToolBackend.create_gnp_graph(params['n'], params['d'] / params['n']) # p = d / n = 512 / 10000 = 0.0512
            # model = NeuroidalModel(gtg, **params)
            # m_total = simulate(model, seed,False, False, False, False, False)
            # capacity = int(m_total / (len(model.S) - model.L))
            # print(f"r: {params['r_approx']}, capacity: {capacity}")
            # print_elapsed(start, time())
            # print(f"Memory usage: {get_memory_usage()} bytes")

            # start = time()
            # ag = GraphToolBackend.to_adjacency(gtg)
            g = AdjacencyBackend.create_gnp_graph(params['n'], params['d'] / params['n'])
            model = NeuroidalModelAdj(g, **params)
            m_total = simulate(model, time(),False, disjoint, False, False, False)
            capacity = int(m_total / (len(model.S) - model.L))
            data[-1].append(capacity)
            print_elapsed(start, time())
            print('\tCapacity:', capacity)
        # print(f"r: {params['r_approx']}, capacity: {capacity}")
        # print(f"Memory usage: {get_memory_usage()} bytes")
    df = pd.DataFrame(data, columns=[f"sim_{i}" for i in range(sim_count)])
    df.to_csv("neuroidal_capacity.csv", index=False)

if __name__ == "__main__":
    main()

"""
Imported graph_tool Neuroidal Model
Imported graph_tool Neuroidal Model Backend
r: 54, capacity: 10000
Runtime: 00:00:03.5904
Memory usage: 3,209,682,944 bytes

Imported adjacency Neuroidal Model
Imported adjacency Neuroidal Model Backend
r: 54, capacity: 3715
Runtime: 00:00:01.3690
Memory usage: 1,824,276,480 bytes

"""
