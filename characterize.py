from time import time

import math
import random

try:
    import cupy as xp
except ImportError:
    import numpy as xp

import pandas as pd
from tqdm import tqdm
from tqdm.contrib import itertools

from src.backend.adjacency import Backend
from src.neuroidal_adj import NeuroidalModel
from src.simulate import simulate


EDGE_PROBABILITIES = xp.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
# [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]

STATS_COLUMNS = ['P', 'Edge Count', 'Local', 'Global', 'Path Length', 'DLocal', 'DGlobal', 'DPath Length']
def characterize_graph(g):
    local_mean = Backend.calculate_local_clustering(g)
    global_mean = Backend.calculate_global_clustering(g)
    path_mean = Backend.calculate_path_length(g)
    return xp.array([local_mean, global_mean, path_mean, xp.count_nonzero(g)])


CAPACITY_COLUMNS = STATS_COLUMNS + ['Computational Time', 'RAM Usage', 'Memory Capacity', 'Interference']
def characterize_capacity(g, disjoint=False):
    start = time()

    n = len(g)

    k = 16
    d = 128
    r = int(n / math.pow(10, 5) * math.pow(2, 16) * k / d)

    params = {'n': n, 'd': d, 't': 0.1, 'k': k, 'k_adj': 2.0, 'r_approx': r, 'L': int(n * 0.1), 'F': 0.1, 'H': int(n * 0.1)}
    model = NeuroidalModel(g, **params)

    results = xp.zeros(len(CAPACITY_COLUMNS) - 1)
    results[9] = simulate(model, time(), False, disjoint, False, False, False)
    results[8] = model.memory_usage()
    results[10] = model.F
    results[7] = time() - start

    results[0] = xp.count_nonzero(g).item()
    results[1:4] = Backend.calculate_stats_quick(g)
    results[4:7] = Backend.calculate_stats_quick(g, True)

    return results


def evaluate_gnp(n, columns, callback, count=100):
    for d in [True, False]:
        out = xp.zeros(shape=(len(EDGE_PROBABILITIES) * count, len(columns)))
        ix = 0
        for p, iy in itertools.product(EDGE_PROBABILITIES, range(count)):
            g = Backend.create_gnp_graph(int(n), p)
            out[ix, 0] = p
            out[ix, 1:] = callback(g, d)
            ix += 1
        df = pd.DataFrame(xp.asnumpy(out), columns=columns)
        df.to_csv(f'data/erdos_reyni_gnp_n_{n}{"_disjoint" if d else ""}.csv')

def evaluate_ba(n, columns, callback, m_step=50, count=100):
    # out = pd.DataFrame(columns=columns)
    # ix = 0
    # # 0 < M < N
    # m_values = list(range(0, n + m_step, m_step))
    # m_values[0] += 1
    # m_values[-1] -= 1
    # for m, iy in itertools.product(m_values, range(count)):
    #     g = Backend.create_ba_graph(int(n), m)
    #     out.loc[ix] = callback(g, d)
    #     ix += 1
    # out.to_csv(f'data/barabasi_albert_n_{n}_varying_m.csv')
    for d in [True, False]:
        out = xp.zeros(shape=(len(EDGE_PROBABILITIES) * count, len(columns)))
        ix = 0
        for p in tqdm(EDGE_PROBABILITIES):
            m = max(1, min(n - 1, int(random.normalvariate(n * p, math.log(n)))))
            for iy in range(count):
                g = Backend.create_ba_graph(int(n), m)
                out[ix, 0] = p
                out[ix, 1:] = callback(g, d)
                ix += 1
        df = pd.DataFrame(xp.asnumpy(out), columns=columns)
        df.to_csv(f'data/barabasi_albert_n_{n}{"_disjoint" if d else ""}.csv')

def evaluate_ws(n, columns, callback, k_step=50, count=100):
    # 0 << ln(n) << k << n
    k_values = list(range(0, n + k_step, k_step))
    k_values[0] = int(k_values[0] + math.log(n))
    k_values[-1] = int(k_values[-1] - math.log(n))

    for d in [True, False]:
        k = n // 2
        # for k in k_values:
        out = xp.zeros(shape=(len(EDGE_PROBABILITIES) * count, len(columns) + 1))
        out[:, 0] = k
        ix = 0
        for p, iy in itertools.product(EDGE_PROBABILITIES, range(count)):
            g = Backend.create_ws_graph(int(n), p, k)
            out[ix, 1] = p
            out[ix, 2:] = callback(g, d)
            ix += 1
        df = pd.DataFrame(xp.asnumpy(out), columns=['K'] + columns)
        df.to_csv(f'data/watts_strogatz_n_{n}_k_{k}{"_disjoint" if d else ""}.csv')

def main():
    # Small numbers of n (< 100k)
    # n = 100 → 1k, step 100; 1k → 10k, step 1k; 10k → 100k, step 10k
    print()

    test_count = 50
    for n in [
        100, 200, 300, 400, 500, 600, 700, 800, 900, 1000,
        2000, 3000, 4000, 5000,
        6000, 7000, 8000, 9000, 10000
    ]:
        print(f'\nProcessing n={n}')
        evaluate_gnp(n, CAPACITY_COLUMNS, characterize_capacity, count=test_count)
        # evaluate_ba(n, CAPACITY_COLUMNS, characterize_capacity, m_step=50, count=test_count)
        # evaluate_ws(n, CAPACITY_COLUMNS, characterize_capacity, k_step=50, count=test_count)



if __name__ == "__main__":
    main()
