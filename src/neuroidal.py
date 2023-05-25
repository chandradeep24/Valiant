import gc, os, sys
from math import comb
import numpy as np
from numpy.random import *
import graph_tool as gt
import networkx as nx


seed(42)
np.random.seed(42)
gt.seed_rng(42)


"""TODO: refactor the notebook functions into this class"""

# _ indicates private function
# public functions should return self


# Add more comments


class Neuroidal:
    def __init__(self, N, D, T, k, k_adj, H, STOP, START_MEM, r_exp):
        self.N = N
        self.D = D
        self.T = T
        self.k = k
        self.k_adj = k_adj
        self.H = H
        self.STOP = STOP
        self.START_MEM = START_MEM
        self.r_exp = r_exp
        self.P = D / (N - 1)
        self.total_inters = 0
        self.ind = 0
        self.inst_inters = 0
        self.inst_len = 0

    # Create the graph with the properties
    def create(self):
        self.g = gt.Graph()
        self.g.add_vertex(self.N)

        self.num_edges = self.P * comb(self.N, 2)
        gt.add_random_edges(self.g, self.num_edges, parallel=False, self_loops=False)

        gt.random_rewire(
            self.g, model="erdos", parallel_edges=False, self_loops=False, verbose=True
        )

        vprop_fired = self.g.new_vertex_property("int")
        vprop_memories = self.g.new_vertex_property("int")
        vprop_fired_now = self.g.new_vertex_property("int")
        vprop_weight = self.g.new_vertex_property("double")
        vprop_threshold = self.g.new_vertex_property("double")

        vprop_fired.a = 0
        vprop_memories.a = 0
        vprop_fired_now.a = 0
        vprop_weight.a = 0.0
        vprop_threshold.a = self.T

        eprop_fired = self.g.new_edge_property("int")
        eprop_weight = self.g.new_edge_property("double")

        eprop_fired.a = 0
        eprop_weight.a = self.T / (self.k_adj * self.k)

        return self

    def generate_memory_bank(self):
        self.memory_bank = []
        for i in np.arange(0, self.START_MEM):
            memory_A = np.random.default_rng().choice(
                np.arange(0, self.N - 1), size=self.r_expected
            )
            self.memory_bank.append(memory_A)

        i, j = np.meshgrid(
            np.arange(len(self.memory_bank)), np.arange(len(self.memory_bank))
        )
        mask = i != j
        i = i[mask]
        j = j[mask]
        pairs = np.unique(np.sort(np.stack((i, j), axis=1)), axis=0)
        np.random.shuffle(pairs)
        return self

    # Refactor from MatMul experiments
    # Check for interference
    def _interference_check(self, A_index, B_index, C):
        sum = 0
        for i in range(len(self.memory_bank)):
            if i != A_index and i != B_index:
                inter = list(set(C.tolist()) & set(C[i]))
                if len(inter) > len(self.memory_bank[i]) / 2:
                    sum += 2
        return sum

    # JOIN function
    def JOIN_one_step_shared(self, i, j):
        return
