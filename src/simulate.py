import math
import pathlib
import itertools

from time import time, strftime

try:
    import cupy as xp
except ImportError:
    import numpy as xp

from src.utils import get_memory_usage, print_elapsed
from src.neuroidal_adj import NeuroidalModel as NeuroidalModelAdj
from src.backend.adjacency import Backend as AdjacencyBackend

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
             verbose=True, vis = False) -> int:
    xp.random.seed(int(seed))

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


    # rng = cp.random.default_rng(int(seed))
    permutations = model.L * (model.L - 1) // 2 + model.L
    model.S = xp.zeros((permutations, model.n), dtype=int)

    for S_i in range(model.L):
        model.S[S_i] = xp.full(model.n, -1, dtype=int)
        model.S[S_i, :model.r_approx] = xp.random.choice(xp.arange(0, model.n - 1), size=model.r_approx, replace=False)

    S_i = model.L
    init_pairs = itertools.combinations(range(model.L), 2)
    for A_i, B_i in init_pairs:
        A = model.S[A_i]
        B = model.S[B_i]

        C = model.JOIN(A, B, disjoint, two_step, fast, vis)
        C_if = model.interference_check(A_i, B_i, S_i, C, vis)

        m += 1
        C_len = xp.count_nonzero(C + 1)
        m_len += C_len

        model.S[S_i] = C
        S_i += 1
        m_total += C_len

        if first_join and vis:
            vis.visualize_start(A, B, C, out_path / f"graph_start.png")

        if m % model.H == 0:
            if verbose:
                model.print_join_update(S_i, H_if, total_if, m_len, m_total)
            H_if = 0
            m_len = 0
            if vis:
                vis.visualize(
                    out_path /
                    f"graph_{S_i}_memories.png"
                )
                vis.visualize_if(
                    out_path /
                    f"graph_{S_i}_if.png"
                )

        if C_if > 0:
            H_if += C_if
            total_if += C_if
            if total_if / S_i > model.F:
                if verbose:
                    model.print_halt_msg(S_i, total_if, m_total)
                if vis:
                    vis.visualize(out_path / f"graph_final_memories.png")
                    vis.visualize_if(out_path / f"graph_final_if.png")
                return (m_total // (S_i - model.L)).item()

    if verbose:
        model.print_memorized_msg(S_i, m_total)

    if vis:
        vis.visualize(out_path / f"graph_final_memories.png")
        vis.visualize_if(out_path / f"graph_final_if.png")

    return (m_total // (S_i - model.L)).item()


def main():
    # For small numbers of n
    # r = (n * k) / d * (2 ** 16 / 10 ** 5)
    # n=500
    # n=10k  d=512 k=16 -> r=234
    # n=100k d=512 k=16 -> r=2134
    # n=1m   d=512 k=16 -> r=23365

    # Interference
    # n=500, d=128, w=16, r=40 L=100 T=0.1 k=2 -> 350
    # params = {'n': 500, 'd': 128, 'k': 16, 'r_approx': 40, 't': 0.1, 'k_adj': 2.0, 'L': 100, 'F': 0.1, 'H': 50}

    data = []

    # step = 50
    # sim_count = 100
    # for n in range(500, 10000 + step, step):
    #     data.append([])
    #     for rep in range(sim_count):
    n = 100
    k = 16
    d = 128
    disjoint = False

    start = time()
    r_approx = int(n / math.pow(10, 5) * math.pow(2, 16) * k / d)
    params = {'n': n, 'd': d, 'k': k, 'r_approx': r_approx, 't': 0.1, 'k_adj': 2.0, 'L': 100, 'F': 0.1, 'H': 50}
    print(f'n={params["n"]}, d={params["d"]}, k={params["k"]} -> r={r_approx}')

    g = AdjacencyBackend.create_gnp_graph(params['n'], params['d'] / params['n'])
    model = NeuroidalModelAdj(g, **params)
    capacity = simulate(model, time(),False, disjoint, False, False, False)
    print_elapsed(start, time())
    print('\tCapacity:', capacity)
    print(f"r: {params['r_approx']}, capacity: {capacity}")
    print(f"Memory usage: {get_memory_usage()} bytes")
    print(f'Reported memory usage: {model.memory_usage()} bytes')
    # data[-1].append(capacity)
    # df = pd.DataFrame(data, columns=[f"sim_{i}" for i in range(sim_count)])
    # df.to_csv("neuroidal_capacity.csv", index=False)

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
