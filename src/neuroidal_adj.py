try:
    import cupy as xp
except ImportError:
    import numpy as xp


print('Imported adjacency Neuroidal Model')


class Visualization:
    def __init__(self, g):
        self.g = g

        n = len(g)

        self.v_num_memories = xp.full(n, 0, dtype=int)
        self.v_interfering_memories = xp.full(n, 0, dtype=int)

        self.v_text = xp.full(n, "", dtype=str)
        # RGB
        self.v_color = xp.full((n, 3), 0, dtype=float)
        max_edges = n * (n - 1)
        self.e_color = xp.full((max_edges, 3), 0, dtype=float)

class NeuroidalModel:
    Visualization = Visualization

    def __init__(self, graph, n, d, t, k, k_adj, r_approx, L, F, H):
        # Initialize parameters
        self.g = graph

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
        self.S = None

        # Create vertex values
        self.mode_q = xp.full(self.n, 1, dtype=int)
        self.mode_f = xp.full(self.n, 0, dtype=int)
        self.mode_T = xp.full(self.n, self.t, dtype=int)

        # Create edge values
        # Make the same shape as adjacency matrix (Each spot is the edge)
        # self.mode_qq = xp.full((n, n), 1, dtype=int)
        self.mode_w = xp.full((n, n), self.t / self.k_m, dtype=float)
        # 17887200, 19327200
        # Blank out the diagonal
        for ix in range(n):
            # self.mode_qq[ix, ix] = 0
            self.mode_w[ix, ix] = 0

    def memory_usage(self):
        size = self.g.nbytes
        size += self.mode_q.nbytes
        size += self.mode_f.nbytes
        size += self.mode_T.nbytes
        # size += self.mode_qq.nbytes
        size += self.mode_w.nbytes
        size += self.S.nbytes
        return size

    def sum_weights(self, s_i):
        mask = (self.g[s_i] == 1) & (self.mode_f == 1)
        return self.mode_w[mask, s_i].sum()

    def _delta(self, s_i, w_i):
        self.mode_f[s_i] = w_i > self.mode_T[s_i]
        self.mode_q[s_i] = self.mode_f[s_i] + 1
        return self

    def update_graph(self, two_step=False, vis=False):
        C_i = 0
        C = xp.full(self.n, -1)
        for s_i in range(self.n):
            w_i = self.sum_weights(s_i)
            self._delta(s_i, w_i)
            if self.mode_q[s_i] == 2:
                C[C_i] = s_i
                C_i += 1
                # if not two_step:
                self.mode_f[s_i] = 0
                # if vis:
                #     vis.v_num_memories[s_i] += 1
            # if two_step:
            #     mask = (self.g[s_i] == 1) & (self.mode_f == 0)
            #     self.mode_qq[mask] = 1
            #     self.mode_qq[~mask] = 2
        return C


    def JOIN(self, A, B, disjoint=False, two_step=False, fast=True, vis=False):
        if disjoint:
            neurons_to_fire = xp.intersect1d(A, B)
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

        C = xp.where(in_degrees > self.t)[0]

        return C

    def interference_check(self, A_i, B_i, S_len, C, vis):
        interfering_set = xp.zeros(S_len, dtype=bool)
        for S_i in range(S_len):
            interfering_set[S_i] = len(xp.intersect1d(self.S[S_i], C)) > xp.count_nonzero(self.S[S_i] + 1) / 2

        interfering_set[A_i] = False
        interfering_set[B_i] = False

        return xp.sum(interfering_set) * 2
