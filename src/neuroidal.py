from time import time

import numpy as np
from numpy.random import default_rng

from src.utils import color_by_value

try:
    import graph_tool.all as gt
except ImportError:
    gt = None


print('Imported graph_tool Neuroidal Model')


class Visualization:
    def __init__(self, g):
        self.g = g

        self.v_num_memories = g.new_vp("int")
        self.v_interfering_mems = g.new_vp("int")

        self.v_num_memories.a = 0
        self.v_interfering_mems.a = 0

        self.v_text = g.new_vp("string")
        self.v_color = g.new_vp("vector<float>")
        self.e_color = g.new_ep("vector<float>")

    def visualize(self, output_file_path):
        for i in self.g.iter_vertices():
            self.v_text[i] = str(self.v_num_memories[i])
            self.v_color[i] = color_by_value(self.v_num_memories[i], cap=50)
        gt.graph_draw(
            self.g,
            pos=gt.fruchterman_reingold_layout(self.g),
            output=str(output_file_path),
            output_size=(1000, 1000),
            vertex_fill_color=self.v_color,
            bg_color="black",
            vertex_text=self.v_text,
            vertex_text_color="white",
        )
        return self

    def visualize_if(self, output_file_path):
        for i in self.g.iter_vertices():
            self.v_text[i] = str(self.v_interfering_mems[i])
            self.v_color[i] = color_by_value(self.v_interfering_mems[i])
        gt.graph_draw(
            self.g,
            pos=gt.fruchterman_reingold_layout(self.g),
            output=str(output_file_path),
            output_size=(1000, 1000),
            vertex_fill_color=self.v_color,
            bg_color="black",
            vertex_text=self.v_text,
            vertex_text_color="white",
        )
        return self

    def visualize_start(self, A, B, C, output_file_path):
        abc_map = {"A": [1.00, 0.75, 0.80],
                   "B": [0.00, 0.00, 1.00],
                   "C": [0.00, 1.00, 0.00]}
        for i in self.g.iter_vertices():
            if i in A:
                self.v_text[i] += "A"
                self.v_color[i] = abc_map["A"]
                for e_ij in self.g.iter_out_edges(i):
                    self.e_color[e_ij] = abc_map["A"]
            elif i in B:
                self.v_text[i] += "B"
                self.v_color[i] = abc_map["B"]
                for e_ij in self.g.iter_out_edges(i):
                    self.e_color[e_ij] = abc_map["B"]
            elif i in C:
                self.v_text[i] += "C"
                self.v_color[i] = abc_map["C"]
            else:
                self.v_color[i] = [0.0, 0.0, 0.0]
        gt.graph_draw(
            self.g,
            pos=gt.fruchterman_reingold_layout(self.g),
            output=str(output_file_path),
            output_size=(1000, 1000),
            vertex_fill_color=self.v_color,
            bg_color="black",
            vertex_text=self.v_text,
            vertex_text_color="white",
            edge_color=self.e_color,
        )
        return self

class NeuroidalModel:
    # Make the visualization a class attribute
    Visualization = Visualization
    def __init__(self, graph, n, d, t, k, k_adj, r_approx, L, F, H, S=None):
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
        if S is None:
            self.S = []
        else:
            self.S = S

        # Create vertex values
        self.mode_q = self.g.new_vp("int")
        self.mode_f = self.g.new_vp("int")
        self.mode_T = self.g.new_vp("int")

        self.mode_q.a = 1
        self.mode_f.a = 0
        self.mode_T.a = self.t

        self.mode_qq = self.g.new_ep("int")
        self.mode_w = self.g.new_ep("double")

        self.mode_qq.a = 1
        self.mode_w.a = self.t / self.k_m

    def sum_weights(self, s_i, fast=True):
        if fast:
            W = self.g.get_in_edges(s_i, [self.mode_w])[:,2]
            F = np.array(self.g.get_in_neighbors(s_i, [self.mode_f])[:,1], 
                         dtype=bool)
            return W[F].sum()
        else:
            w_i = 0
            for s_ji in self.g.iter_in_edges(s_i):
                if self.mode_f[s_ji[0]] == 1:
                    w_i += self.mode_w[s_ji]
            return w_i

    def _delta(self, s_i, w_i):
        if w_i <= self.mode_T[s_i]:
            self.mode_f[s_i] = 0
            self.mode_q[s_i] = 1
        else:
            self.mode_f[s_i] = 1
            self.mode_q[s_i] = 2
        return self

    def _lambda(self, s_ji, f_j):
        if f_j == 0:
            self.mode_qq[s_ji] = 1
        else:
            self.mode_qq[s_ji] = 2
        return self

    def update_graph(self, two_step=False, fast=True, vis=False):
        C = []
        for s_i in self.g.iter_vertices():
            w_i = self.sum_weights(s_i, False)
            self._delta(s_i, w_i)
            if self.mode_q[s_i] == 2:
                C.append(s_i)
                if not two_step:
                    self.mode_f[s_i] = 0
                if vis:
                    vis.v_num_memories[s_i] += 1
            if two_step:
                for s_ji in self.g.iter_in_edges(s_i):
                    f_j = self.mode_f[s_ji[0]]
                    self._lambda(s_ji, f_j)
        return C

    def JOIN(self, A, B, disjoint=False, two_step=False, fast=True, vis=False):
        if disjoint:
            neurons_to_fire = A & B
        else:
            neurons_to_fire = A + B
        for i in neurons_to_fire:
            self.mode_f[i] = 1
        C = self.update_graph(two_step, fast, vis)
        self.mode_f.a = 0
        self.mode_q.a = 1
        return C

    def quick_JOIN(self, A, B, vis=False):
        firing_edge_weight = self.t / self.k_m

        self.mode_w.a = 0

        for i in A + B:
            out_edges = self.g.get_out_edges(i, [self.g.edge_index])
            self.mode_w.a[out_edges[:, 2]] = firing_edge_weight

        all_in_degrees = self.g.get_in_degrees(self.g.get_vertices(), 
                                               eweight=self.mode_w)
        C = self.g.get_vertices()[all_in_degrees > self.t]
        if vis:
            for i in C:
                vis.v_num_memories[i] += 1

        return C

    def interference_check(self, A_i, B_i, C, vis):
        count = 0
        for D_i in range(len(self.S)):
            if D_i != A_i and D_i != B_i:
                D = self.S[D_i]
                interfering_set = set(C) & set(D)
                if len(interfering_set) > (len(D) / 2):
                    count += 2
                    if vis:
                        for i in interfering_set:
                            vis.v_interfering_mems[i] += 1
        return count
