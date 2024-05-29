import numpy as np
import scipy.sparse as sp
from numpy.random import default_rng


print('Imported adjacency Neuroidal Model')


class Visualization:
    def __init__(self, g: sp.csr_matrix):
        self.g = g

        n = len(g)

        self.v_num_memories = np.full(n, 0, dtype=int)
        self.v_interfering_memories = np.full(n, 0, dtype=int)

        self.v_text = np.full(n, "", dtype=str)
        # RGB
        self.v_color = np.full((n, 3), 0, dtype=float)
        max_edges = n * (n - 1)
        self.e_color = np.full((max_edges, 3), 0, dtype=float)

class NeuroidalModel:
    Visualization = Visualization

    def __init__(self, graph, n, d, t, k, k_adj, r_approx, L, F, H, S=None):
        # Initialize parameters
        self.g: sp.csr_matrix = graph
        self.n = n
        self.d = d
        self.t = t
        self.k = k
        self.k_adj = k_adj
        self.k_m = k * k_adj
        self.r_approx = r_approx
        self.L = L
        self.F = F
        self.H = H
        if S is None:
            self.S = []
        else:
            self.S = S

        # Create vertex values
        self.mode_q = np.full(self.n, 1, dtype=int)
        self.mode_f = np.full(self.n, 0, dtype=int)
        self.mode_T = np.full(self.n, self.t, dtype=int)

        # Create edge values
        # Make the same shape as adjacency matrix (Each spot is the edge)
        self.mode_qq = np.full((n, n), 1, dtype=int)
        self.mode_w = np.full((n, n), self.t / self.k_m, dtype=float)

        # Blank out the diagonal
        for ix in range(n):
            self.mode_qq[ix, ix] = 0
            self.mode_w[ix, ix] = 0

    def sum_weights(self, s_i):
        mask = (self.g[s_i] == 1) & (self.mode_f == 1)
        return self.mode_w[mask, s_i].sum()

    def _delta(self, s_i, w_i):
        self.mode_f[s_i] = w_i > self.mode_T[s_i]
        self.mode_q[s_i] = self.mode_f[s_i] + 1
        return self

    def update_graph(self, two_step=False, vis=False):
        C = []
        for s_i in range(self.n):
            w_i = self.sum_weights(s_i)
            self._delta(s_i, w_i)
            if self.mode_q[s_i] == 2:
                C.append(s_i)
                if not two_step:
                    self.mode_f[s_i] = 0
                if vis:
                    vis.v_num_memories[s_i] += 1
            if two_step:
                mask = (self.g[s_i] == 1) & (self.mode_f == 0)
                self.mode_qq[mask] = 1
                self.mode_qq[~mask] = 2
        return C


    def JOIN(self, A, B, disjoint=False, two_step=False, fast=True, vis=False):
        if disjoint:
            neurons_to_fire = list(set(A) & set(B))
        else:
            neurons_to_fire = A + B
        self.mode_f[neurons_to_fire] = 1
        C = self.update_graph(two_step, vis)
        self.mode_f.fill(0)
        self.mode_q.fill(1)
        return C

    def quick_JOIN(self, A, B, vis=False):
        firing_edge_weight = self.t / self.k_m

        self.mode_w.fill(0)
        self.mode_w[:, A + B] = self.g[:, A + B] * firing_edge_weight
        in_degrees = self.mode_w.sum(axis=1)
        C = np.where(in_degrees > self.t)[0]

        return C

    def interference_check(self, A_i, B_i, C, vis):
        interfering_set = [len(set(self.S[i]) & set(C)) > len(self.S[i]) / 2 for i in range(len(self.S))]
        interfering_set[A_i] = False
        interfering_set[B_i] = False
        return np.sum(interfering_set) * 2

        # count = 0
        # for D_i in range(len(self.S)):
        #     if D_i != A_i and D_i != B_i:
        #         D = self.S[D_i]
        #         interfering_set = set(C) & set(D)
        #         if len(interfering_set) > (len(D) / 2):
        #             count += 2

        # return count
