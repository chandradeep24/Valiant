import numpy as np


def overlap(x, y, k):
    T = len(x) // k
    intersection = np.intersect1d(x, y)
    return len(intersection) > T


def process_worker(sub, i, j, result_queue):
    overlap_count = 0
    for idx in range(i, j):
        for idy in range(idx + 1, sub.shape[0]):
            row_i = sub[idx]
            row_j = sub[idy]
            if overlap(row_i, row_j, k=2):
                overlap_count += 1
    result_queue.put(overlap_count)
