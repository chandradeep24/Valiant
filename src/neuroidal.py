import time
import pathlib
import itertools
import numpy as np
from numpy.random import *
import graph_tool.all as gt

rng = np.random.default_rng(seed=42)

class NeuroidalModel:
    def __init__(self, n, d, t, k, k_adj, L, F, H, S, r_approx, new_mems=False):
        self.n = n
        self.d = d
        self.p = d / n
        if n < 10^5:
            self.p = d / (n - 1)
        self.t = t
        self.k = k
        self.k_adj = k_adj
        self.k_m = k * k_adj
        self.L = L
        self.F = F
        self.H = H
        self.S = S
        self.r_approx = r_approx
        self.track_only_new_memories = new_mems

    # Generate an Erdos-Renyi G(n,p) gt.Graph where:
    # n: number of nodes
    # p: probability of edge existing between two nodes
    def create_gnp_graph(n: int, p: float, fast: bool) -> gt.Graph:
        g = gt.Graph(directed=True)
        g.add_vertex(n)
        if fast:
            num_edges = rng.binomial(n*(n-1)/2, p)
            sources = rng.integers(0, n, num_edges*2)
            targets = rng.integers(0, n, num_edges*2)
            mask = sources != targets # removes self-loops
            g.add_edge_list(np.column_stack((sources[mask], targets[mask])))
        else:
            all_edges = itertools.permutations(range(n), 2)
            for e in all_edges:
                if rng.random() < p:
                    g.add_edge(*e)
        return g

    def initialize_mode(self, fast=True):
        self.g = self.create_gnp_graph(self.n, self.p, fast)

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

        self.vprop_memories = self.g.new_vp("int")
        self.vprop_memories.a = 0

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
        if w_i > self.mode_T[s_i]:
            self.mode_f[s_i] = 1
            self.mode_q[s_i] = 2
        return self

    def _lambda(self, s_i, w_i, s_ji, f_j):
        if f_j == 1:
            self.mode_qq[s_ji] = 2
        return self

    def update_graph(self, one_step=True):
        C = []
        for s_i in self.g.iter_vertices():
            w_i = self.sum_weights(s_i, fast=True)
            self._delta(s_i, w_i)
            if self.mode_q[s_i] == 2:
                C.append(s_i)
                if one_step:
                    self.mode_f[s_i] = 0
            if not one_step:
                for s_ji in self.g.iter_in_edges(s_i):
                    f_j = self.mode_f[s_ji[0]]
                    self._lambda(s_i, w_i, s_ji, f_j)
        return C

    def JOIN_one_step_shared(self, A, B):
        for i in A + B:
            self.mode_f[i] = 1
        C = self.update_graph(one_step=True)
        self.mode_f.a = 0
        self.mode_q.a = 1
        return C

    def quick_JOIN(self, A, B):
        self.mode_w.a = 0
        for i in A + B:
            self.mode_w.a[self.g.get_out_edges(i, [self.g.edge_index])[:,2]] = self.t / self.k_m
        return self.g.get_vertices()[self.g.get_in_degrees(self.g.get_vertices(), eweight=self.mode_w) > self.t]

    def interference_check(self, A_i, B_i, C):
        sum = 0
        for D_i in range(len(self.S)):
            if D_i != A_i and D_i != B_i:
                D = self.S[D_i]
                if len(set(C) & set(D)) > (len(D) / 2):
                    sum += 2
        return sum

    def generate_color_by_value(value, cap=10):
        value = min(max(value, 0), cap)
        r = value / cap
        color = [r, 0.0, 0.0]
        return color

    def _visualize(self, output_file_path):
        self.vprop_colors = self.g.new_vp("vector<float>")
        self.vprop_text = self.g.new_v("string")
        for v in self.g.vertices():
            self.vprop_colors[v] = self.generate_color_by_value(self.vprop_memories[v])
            self.vprop_text[v] = str(self.vprop_memories[v])
        gt.graph_draw(
            self.g,
            pos=gt.fruchterman_reingold_layout(self.g),
            output=str(output_file_path),
            output_size=(1000, 1000),
            vertex_fill_color=self.vprop_colors,
            bg_color="black",
            vertex_text=self.vprop_text,
            vertex_text_color="white",
        )
        return self

    def _visualize_n(self, output_file_path):
        self.vprop_colors = self.g.new_vp("vector<float>")
        self.vprop_text = self.g.new_vp("string")
        for v in self.g.vertices():
            self.vprop_colors[v] = self.generate_color_by_value(
                self.vprop_memories[v], cap=50
            )
            self.vprop_text[v] = str(self.vprop_memories[v])
        gt.graph_draw(
            self.g,
            pos=gt.fruchterman_reingold_layout(self.g),
            output=str(output_file_path),
            output_size=(1000, 1000),
            vertex_fill_color=self.vprop_colors,
            bg_color="black",
            vertex_text=self.vprop_text,
            vertex_text_color="white",
        )
        return self

    def _visualize_first_join(self, A, B, C, output_file_path):
        self.vprop_colors = self.g.new_vp("vector<float>")
        self.vprop_text = self.g.new_vp("string")
        self.eprop_colors = self.g.new_ep("vector<float>")
        abc_map = {"A": [1.0, 0.75, 0.8],
                    "B": [0.0, 0.0, 1.0],
                    "C": [0.0, 1.0, 0.0]}
        for v in self.g.vertices():
            if v in A:
                self.vprop_colors[v] = abc_map["A"]
                self.vprop_text[v] += "A"
                for e in v.out_edges():
                    self.eprop_colors[e] = abc_map["A"]
            elif v in B:
                self.vprop_colors[v] = abc_map["B"]
                self.vprop_text[v] += "B"
                for e in v.out_edges():
                    self.eprop_colors[e] = abc_map["B"]
            elif v in C:
                self.vprop_colors[v] = abc_map["C"]
                self.vprop_text[v] += "C"
            else:
                self.vprop_colors[v] = [0.0, 0.0, 0.0]
        gt.graph_draw(
            self.g,
            pos=gt.fruchterman_reingold_layout(self.g),
            output=str(output_file_path),
            output_size=(1000, 1000),
            vertex_fill_color=self.vprop_colors,
            bg_color="black",
            vertex_text=self.vprop_text,
            vertex_text_color="white",
            edge_color=self.eprop_colors,
        )
        return self

    def print_join_update(self, S_len, H_if, total_if, m_len, m_total):
        print("Current Total Memories:", S_len)
        print("Batch Average Memory Size:", int(m_len/self.H))
        print("Running Average Memory Size:", 
                int(m_total/(S_len-self.L)),"\n\n")
        if self.n < 10^5:
            print("Batch Interference Rate:", round(H_if/self.H, 6))
            print("Running Average Int. Rate:", round(total_if/S_len, 6))

    def print_halt_msg(self, S_len, total_if, m_total):
        r_obs = int(m_total/(S_len-self.L))
        r_error = round(((self.r_approx - r_obs) / r_obs) * 100, 2)
        print("-- End of Simulation (Halted) --\n")
        print("Given: n=", self.n, "d=", self.d, "k=", self.k, "k_adj=", 
                self.k_adj, "r_approx=", self.r_approx, "START_MEM=", self.L)
        print("we halted Memory Formation at", 
                self.F*100, "% Total Interference.\n")
        print("Empirical Memory Size:", int(m_total/(S_len-self.L)))
        print("Approximation Error of r:", r_error, "%")
        print("Total Average Interference Rate:", round(total_if/S_len, 6))
        print("Capacity:", self.L, "Initial Memories +", 
                S_len-self.L, "JOIN Memories.")

    def print_memorized_msg(self, S_len, m_total):
        r_obs = int(m_total/(S_len-self.L))
        r_error = round(((self.r_approx - r_obs) / r_obs) * 100, 2)
        print("-- End of Simulation (Completed) --\n")
        print("Given: n=", self.n, "d=", self.d, "k=", self.k, "k_adj=", 
                self.k_adj, "r_approx=", self.r_approx, "START_MEM=", self.L)
        print("We memorized all combinations of", self.L,"memories",
                "\n","with less than", self.F*100, "% interference.\n")
        print("Empirical Memory Size:", int(m_total/(S_len-self.L)))
        print("Approximation Error of r:", r_error, "%")
        print("Contains:", self.L, "Initial Memories +",
                S_len-self.L, "JOIN Memories.")

    def simulate(self, fast=True, vis=False, verbose=False):
        m = 0
        H_if = 0
        m_len = 0
        m_total = 0
        total_if = 0
        first_join = False
        init_pairs = itertools.combinations(range(self.L), 2)
        self.S = [rng.choice(np.arange(0, self.n - 1), size=self.r_approx)
             for _ in range(self.L)]

        print("-- Start of Simulation --\n")
        start_dir = pathlib.Path('../assets')
        start_time = time.strftime("%m-%d-%y_%H:%M:%S")
        out_path = (start_dir / start_time / '_neurons_' / self.n /
                    '_degree_' / self.d / '_edge-weights_' / self.k /
                    '_replication_' / self.r_approx / '_start-mems_' / self.L)
        if vis:
            first_join = True
            out_path.mkdir()
            self._visualize(out_path/'graph_'/len(self.S)/'_memories.png')
            self._visualize_n(out_path/'graph_'/len(self.S)/'_n-memories.png')

        for A_i, B_i in init_pairs:
            A = list(self.S[A_i])
            B = list(self.S[B_i])
            if fast:
                C = self.quick_JOIN(A, B)
            else:
                C = self.JOIN_one_step_shared(A, B)

            m += 1
            m_len += len(C)
            self.S.append(C)
            m_total += len(C)
            if first_join:
                self._visualize_first_join(A, B, C, 
                                            out_path / 'graph_1st_join.png')
            if m % self.H == 0:
                if verbose:
                    self.print_join_update(len(self.S), H_if,
                                            total_if, m_len, m_total)
                H_if = 0
                m_len = 0
                if vis:
                    self._visualize(out_path / 'graph_' /
                                        len(self.S)/ '_memories.png')
                    self._visualize_n(out_path / 'graph_' /
                                        len(self.S)/ '_n-memories.png')

            C_if = self.interference_check(A_i, B_i, C)
            if C_if > 0:
                H_if += C_if
                total_if += C_if
                if total_if/len(self.S) > self.F:
                    self.print_halt_msg(len(self.S), total_if, m_total)
                    if vis:
                        self._visualize(out_path/'graph_final_memories.png')
                        self._visualize_n(out_path/'graph_final_n-memories.png')
                    return

        self.print_memorized_msg(len(self.S), m_total)
        if vis:
            self._visualize(out_path / 'graph_final_memories.png')
            self._visualize_n(out_path / 'graph_final_n-memories.png')
        return
