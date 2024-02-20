import os
import shutil
import itertools
import numpy as np
from math import comb
from numpy.random import *
import graph_tool.all as gt

rng = np.random.default_rng(seed=42)

# TODO: Move all visualization functions to a separate source file
# TODO: Investigate PostInterferenceUpdate() again or remove function
# TODO: Investigate GreedyGenerateMemoryBank() again or remove function

class NeuroidalModel:
    def __init__(self, n, d, t, k, k_adj, L, F, H, S, r_approx, new_mems=False):
        self.n = n
        self.d = d
        if n >= 10^5:
            self.p = d / n
        else:
            self.p = d / (n - 1)
        self.t = t
        self.k = k
        self.k_adj = k_adj
        self.k_m = k * k_adj
        self.L = L
        self.F = F
        self.H = H
        self.S = []
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
        self.g = create_gnp_graph(self.n, self.p, fast)

        self.mode_T = g.new_vp("int")
        self.mode_q = g.new_vp("int")
        self.mode_f = g.new_vp("int")
        self.mode_qq = g.new_ep("int")
        self.mode_w = g.new_ep("double")

        self.mode_T.a = t
        self.mode_q.a = 1
        self.mode_f.a = 0
        self.mode_qq.a = 1
        self.mode_w.a = t / k_m

        return self

    def sum_weights(s_i, fast=True):
        if fast:
            W = self.g.get_in_edges(s_i, [self.mode_w])[:,2]
            F = np.array(self.g.get_in_neighbors(s_i, [self.mode_f])[:,1], 
                                                                    dtype=bool)
            return W[F].sum()
        else:
            w_i = 0
            for s_ji in g.iter_in_edges(s_i):
                if mode_f[s_ji[0]] == 1:
                    w_i += mode_w[s_ji]
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

    def update_graph(one_step=True):
        C = []
        for s_i in self.g.iter_vertices():
            w_i = sum_weights(s_i, fast=True)
            _delta(self, s_i, w_i)
            if self.mode_q[s_i] == 2:
                C.append(s_i)
                if one_step:
                    self.mode_f[s_i] = 0
            if not one_step:
                for s_ji in self.g.iter_in_edges(s_i):
                    f_j = self.mode_f[s_ji[0]]
                    _lambda(self, s_i, w_i, s_ji, f_j)
        return C

    def JOIN_one_step_shared(A, B):
        for i in A + B:
            self.mode_f[i] = 1
        C = update_graph(one_step=True)
        self.mode_f.a = 0
        self.mode_q.a = 1
        return C

    def quick_JOIN(A, B):
        self.mode_w.a = 0
        for i in A + B:
            self.mode_w.a[self.g.get_out_edges(i, 
                [self.g.edge_index])[:,2]] = self.t / self.k_m
        return self.g.get_vertices()[self.g.get_in_degrees(self.g.get_vertices(),
                 eweight=self.mode_w) > self.t]

    def interference_check(self, A_i, B_i, C):
        sum = 0
        for D_i in range(len(self.S)):
            if D_i != A_i and D_i != B_i:
                D = self.S[D_i]
                if len(set(C) & set(D)) > (len(D) / 2):
                    sum += 2
        return sum
     
    """ 
    For updating weights after interference
        0. Initialize list of JOIN edges
        1. Incoming edge from A/B to C: Add to list of JOIN edges
        2. Incoming edge from elsewhere to C: decrease weight by 1/num of edges
        3. All other edges not in JOIN list: increase weight by 1/num of edges
    """
    def _post_interference_update(self, A_i, B_i, C):
        if len(self.S) > self.L:
            JOIN_set = set()
            # decrement = 1 - 1 / self.n
            for v in C:
                for edge in self.g.vertex(v).in_edges():
                    if (
                        edge.source() in self.S[A_i]
                        or edge.source() in self.S[B_i]
                    ):
                        JOIN_set.add(edge)
                    else:
                        if edge not in JOIN_set:
                            if self.vprop_memories[v] > 0:
                                self.eprop_weight[edge] = 0
                            # else:
                            #     self.eprop_weight[edge] *= decrement ** max(
                            #         1, self.vprop_memories[v]
                            #     )

        return sum

    def generate_color_by_value(value, cap=10):
        value = min(max(value, 0), cap)
        r = value / cap
        color = [r, 0.0, 0.0]
        return color

    def _visualize(self, output_file_name):
        self.vprop_colors = self.g.new_vertex_property("vector<float>")
        self.vprop_text = self.g.new_vertex_property("string")
        for v in self.g.vertices():
            self.vprop_colors[v] = generate_color_by_value(self.vprop_memories[v])
            self.vprop_text[v] = str(self.vprop_memories[v])
        gt.graph_draw(
            self.g,
            pos=gt.fruchterman_reingold_layout(self.g),
            output=output_file_name,
            output_size=(1000, 1000),
            vertex_fill_color=self.vprop_colors,
            bg_color="black",
            vertex_text=self.vprop_text,
            vertex_text_color="white",
        )
        return self

    def _visualize_n(self, output_file_name):
        self.vprop_colors = self.g.new_vertex_property("vector<float>")
        self.vprop_text = self.g.new_vertex_property("string")
        for v in self.g.vertices():
            self.vprop_colors[v] = generate_color_by_value(
                self.vprop_n_memories[v], cap=50
            )
            self.vprop_text[v] = str(self.vprop_n_memories[v])
        gt.graph_draw(
            self.g,
            pos=gt.fruchterman_reingold_layout(self.g),
            output=output_file_name,
            output_size=(1000, 1000),
            vertex_fill_color=self.vprop_colors,
            bg_color="black",
            vertex_text=self.vprop_text,
            vertex_text_color="white",
        )
        return self

    def _visualize_first_join(self, A, B, C, output_file_name):
        self.vprop_colors = self.g.new_vertex_property("vector<float>")
        self.vprop_text = self.g.new_vertex_property("string")
        self.eprop_colors = self.g.new_edge_property("vector<float>")
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
            output=output_file_name,
            output_size=(1000, 1000),
            vertex_fill_color=self.vprop_colors,
            bg_color="black",
            vertex_text=self.vprop_text,
            vertex_text_color="white",
            edge_color=self.eprop_colors,
        )
        return self

    def print_join_update(self, S_length, H_if, total_if, m_len, m_total):
        print("Current Total Memories:", S_length)
        print("Batch Average Memory Size:", int(m_len/self.H))
        print("Running Average Memory Size:", 
                int(m_total/(S_length-self.L)),"\n\n")
        if self.n < 10^5:
            print("Batch Interference Rate:", round(H_if/self.H, 6))
            print("Running Average Int. Rate:", round(total_if/S_length, 6))

    def print_halt_msg(self, S_length, total_if, m_total):
        r_obs = int(m_total/(S_length-self.L))
        r_error = round(((self.r_approx - r_obs) / r_obs) * 100, 2)
        print("-- End of Simulation (Halted) --\n")
        print("Given: n=", self.n, "d=", self.d, "k=", self.k, "k_adj=", 
                self.k_adj, "r_approx=", self.r_approx, "START_MEM=", self.L)
        print("we halted Memory Formation at", 
                self.F*100, "% Total Interference.\n")
        print("Empirical Memory Size:", int(m_total/(S_length-self.L)))
        print("Approximation Error of r:", r_error, "%")
        print("Total Average Interference Rate:", round(total_if/S_length, 6))
        print("Capacity:", self.L, "Initial Memories +", 
                S_length-self.L, "JOIN Memories.")

    def print_memorized_msg(self, S_length, m_total):
        r_obs = int(m_total/(S_length-self.L))
        r_error = round(((self.r_approx - r_obs) / r_obs) * 100, 2)
        print("-- End of Simulation (Completed) --\n")
        print("Given: n=", self.n, "d=", self.d, "k=", self.k, "k_adj=", 
                self.k_adj, "r_approx=", self.r_approx, "START_MEM=", self.L)
        print("We memorized all combinations of", self.L,"memories",
                "\n","with less than", self.F*100, "% interference.\n")
        print("Empirical Memory Size:", int(m_total/(S_length-self.L)))
        print("Approximation Error of r:", r_error, "%")
        print("Contains:", self.L, "Initial Memories +",
                S_length-self.L, "JOIN Memories.")

    def greedy_generate_memory_bank(self):
        self.memory_bank = self.S

        divider = int(0.8 * self.L)

        # random portion
        for i in np.arange(0, divider):
            memory_A = np.random.default_rng().choice(
                np.arange(0, self.N - 1), size=self.r_exp
            )
            self.memory_bank.append(memory_A)
            for v in memory_A:
                self.vprop_n_memories[v] += 1

        # greedy portion
        options = range(0, self.n)
        for i in np.arange(divider, self.L):
            options = sorted(options, key=lambda v: -1 * self.vprop_n_memories[v])
            memory_A = options[0 : self.r_exp]
            self.memory_bank.append(memory_A)
            for v in memory_A:
                self.vprop_n_memories[v] += 1
        return self

    def simulate(self, fast=True, vis=False, update=False, verbose=False):
        self.update = update
        m = 0
        H_if = 0
        m_len = 0
        m_total = 0
        total_if = 0
        first_join = False
        print("-- Start of Simulation --\n")
        init_pairs = itertools.combinations(range(self.L), 2)
        self.S = [rng.choice(np.arange(0,self.n - 1), size=self.r_approx)
             for _ in range(self.L)]

        output_directory = f"../assets/neurons_{self.n}_degree_{self.d}_replication_{self.r_approx}_edge_weights_{self.k}_threshold_{self.t}_startmem_{self.L}"
        if vis:
            first_join = True
            if os.path.exists(output_directory):
                shutil.rmtree(output_directory)
                os.makedirs(output_directory)
            else:
                os.makedirs(output_directory)
            self._visualize(os.path.join(output_directory, 
                    f"graph_{len(self.S)}_memories.png"))
            self._visualize_n(os.path.join(output_directory, 
                    f"graph_{len(self.S)}_n_memories.png"))

        for A_i,B_i in init_pairs:
            A = list(S[A_i])
            B = list(S[B_i])
            if fast:
                C = quick_JOIN(A, B)
            else:
                C = JOIN_one_step_shared(A, B)
            C_if = interference_check(self.S, A_i, B_i, C)
            m += 1
            S.append(C)
            m_len += len(C)
            m_total += len(C)
            if first_join:
                self._visualize_first_join(A, B, C,
                    os.path.join(output_directory, f"graph_first_join.png"))
            if m % self.H == 0::
                if verbose:
                    print_join_update(len(self.S), H_if,
                                      total_if, m_len, m_total)
                H_if = 0
                m_len = 0
                if vis:
                    self._visualize(os.path.join(output_directory,
                            f"graph_{len(self.memory_bank)}_memories.png"))
                    self._visualize_n(os.path.join(output_directory,
                            f"graph_{len(self.memory_bank)}_n_memories.png"))
            if C_if > 0:
                H_if += C_if
                total_if += C_if
                if total_if/len(self.S) > self.F:
                    print_halt_msg(self, len(self.S), total_if, m_total)
                    if vis:
                        self._visualize(os.path.join(output_directory,
                                f"graph_final_memories.png"))
                        self._visualize_n(os.path.join(output_directory,
                                f"graph_final_n_memories.png"))
                    return self
        print_memorized_msg(self, len(self.S), m_total)
        if vis:
            self._visualize(os.path.join(output_directory,
                f"graph_final_memories.png"))
            self._visualize_n(os.path.join(output_directory,
                f"graph_final_n_memories.png"))
        return self
