import gc, os, sys
from math import comb
import numpy as np
from numpy.random import *
import graph_tool.all as gt
import matplotlib.pyplot as plt
import shutil


seed(42)
np.random.seed(42)
gt.seed_rng(42)


"""TODO: fix issues and add more comments"""

# _ indicates private function
# public functions should return self to be composable


def generate_color_by_value(value, cap=10):
    value = min(max(value, 0), cap)
    r = value / cap
    color = [r, 0.0, 0.0]
    return color


abc_map = {"A": [1.0, 0.75, 0.8], "B": [0.0, 0.0, 1.0], "C": [0.0, 1.0, 0.0]}


class NeuroidalModel:
    def __init__(
        self, N, D, T, k, k_adj, H, STOP, START_MEM, r_exp, F=2, new_mems_only=False
    ):
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
        self.F = F

        self.track_only_new_memories = new_mems_only

    # Experimental!!
    # Generate an Erdos-Renyi G(n,p) gt.Graph where:
    # n: number of nodes
    # p: probability of edge existing between two nodes
    def create_gnp_graph(n: int, p: float) -> gt.Graph:
        # relationship between m and p
        m = int(p * (n * (n - 1)) / 2)
        g = gt.Graph()
        g.add_vertex(n)

        prob = lambda r, s: p

        # Edge generation
        sources = np.random.randint(0, n - 1, m)
        targets = np.random.randint(0, n - 1, m)

        mask = sources != targets
        edges = np.column_stack((sources[mask], targets[mask]))
        g.add_edge_list(edges)

        gt.random_rewire(g, model="erdos", edge_probs=prob)

        return g

    def create_graph(self):
        self.g = gt.Graph()
        self.g.add_vertex(self.N)

        self.vprop_fired = self.g.new_vertex_property("int")
        self.vprop_memories = self.g.new_vertex_property("int")
        self.vprop_n_memories = self.g.new_vertex_property("int")
        self.vprop_fired_now = self.g.new_vertex_property("int")
        self.vprop_weight = self.g.new_vertex_property("double")
        self.vprop_threshold = self.g.new_vertex_property("double")

        self.vprop_fired.a = 0
        self.vprop_memories.a = 0
        self.vprop_n_memories.a = 0
        self.vprop_fired_now.a = 0
        self.vprop_weight.a = 0.0
        self.vprop_threshold.a = self.T

        x, y = np.meshgrid(np.arange(self.N), np.arange(self.N))  # sparse=True)
        mask = x != y
        x = x[mask]
        y = y[mask]
        pairs = np.stack((x, y), axis=1)
        print("Number of all possible edges:", pairs.shape[0])

        z = np.random.default_rng().geometric(
            p=self.P, size=((self.N * self.N) - self.N)
        )
        num_edges = (z == 1).sum()

        index = np.random.default_rng().choice(
            pairs.shape[0], size=int(num_edges), replace=False
        )

        self.g.add_edge_list(pairs[index])

        self.eprop_fired = self.g.new_edge_property("int")
        self.eprop_weight = self.g.new_edge_property("double")

        self.eprop_fired.a = 0
        self.eprop_weight.a = self.T / (self.k_adj * self.k)

        return self

    def _generate_adj_mat(self):
        self.adjmat = gt.adjacency(self.g, self.eprop_weight)
        # print(self.adjmat)
        return self

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

    def generate_memory_bank(self):
        self.memory_bank = []
        self.interference_counts = []
        for i in np.arange(0, self.START_MEM):
            memory_A = np.random.default_rng().choice(
                np.arange(0, self.N - 1), size=self.r_exp
            )
            self.memory_bank.append(memory_A)
            if not self.track_only_new_memories:
                for v in memory_A:
                    self.vprop_n_memories[v] += 1

        print(
            "memory bank len:",
            len(self.memory_bank),
            "interference count len:",
            len(self.interference_counts),
        )
        return self

    def greedy_generate_memory_bank(self):
        self.memory_bank = []

        divider = int(0.8 * self.START_MEM)

        # random portion
        for i in np.arange(0, divider):
            memory_A = np.random.default_rng().choice(
                np.arange(0, self.N - 1), size=self.r_exp
            )
            self.memory_bank.append(memory_A)
            for v in memory_A:
                self.vprop_n_memories[v] += 1

        # greedy portion
        options = range(0, self.N)
        for i in np.arange(divider, self.START_MEM):
            options = sorted(options, key=lambda v: -1 * self.vprop_n_memories[v])
            memory_A = options[0 : self.r_exp]
            self.memory_bank.append(memory_A)
            for v in memory_A:
                self.vprop_n_memories[v] += 1
        return self

    # Check for interference
    def _interference_check(self, A_index, B_index, C):
        sum = 0
        c_position = len(self.interference_counts)
        self.interference_counts.append(0)
        for i in range(len(self.memory_bank)):
            if i != A_index and i != B_index:
                # print(type(C), type(i))
                inter = list(set(C) & set(self.memory_bank[i]))
                if len(inter) > len(self.memory_bank[i]) / self.F:
                    sum += 2
                    self.interference_counts[c_position] += 1
                    self.interference_counts[i] += 1
                    for v in inter:
                        self.vprop_memories[v] += 1
        return sum

    # Vectorized JOIN function
    def _JOIN_one_step_shared(self, i, j, output_directory):
        self._generate_adj_mat()

        memory_A = self.memory_bank[i]
        memory_B = self.memory_bank[j]

        state = np.zeros(self.adjmat.shape[0])
        state[memory_A] = 1
        state[memory_B] = 1

        fired = np.heaviside((self.adjmat @ state) - 1, 1)

        memory_C = np.nonzero(fired)[0]

        if self.first_join and self.vis:
            self._visualize_first_join(
                memory_A,
                memory_B,
                memory_C,
                os.path.join(output_directory, f"graph_first_join.png"),
            )
            self.first_join = False

        for v in memory_C:
            self.vprop_n_memories[v] += 1

        # print((self.adjmat @ state) - 1, fired, memory_C)
        inter = self._interference_check(i, j, memory_C)
        self.memory_bank.append(memory_C)

        return inter, len(memory_C)

    def simulate(self, vis=False):
        i, j = np.meshgrid(
            np.arange(len(self.memory_bank)), np.arange(len(self.memory_bank))
        )
        mask = i != j
        i = i[mask]
        j = j[mask]
        gc.collect()
        pairs = np.unique(np.sort(np.stack((i, j), axis=1)), axis=0)
        np.random.shuffle(pairs)

        total_inters = 0
        ind = 0
        inst_inters = 0
        inst_len = 0

        output_directory = f"../assets/neurons_{self.N}_degree_{self.D}_replication_{self.r_exp}_edge_weights_{self.k}_threshold_{self.T}_startmem_{self.START_MEM}"

        self.first_join = False

        self.vis = vis

        if vis:
            if os.path.exists(output_directory):
                shutil.rmtree(output_directory)
                os.makedirs(output_directory)
            else:
                os.makedirs(output_directory)

            self._visualize(
                os.path.join(
                    output_directory, f"graph_{len(self.memory_bank)}_memories.png"
                )
            )

            self._visualize_n(
                os.path.join(
                    output_directory, f"graph_{len(self.memory_bank)}_n_memories.png"
                )
            )
            self.first_join = True

        for i in range(len(self.memory_bank)):
            total_inters += self._interference_check(i, i, self.memory_bank[i])

        print(total_inters)

        for pair in pairs:
            ind += 1
            i = pair[0]
            j = pair[1]
            inter_flag, length = self._JOIN_one_step_shared(i, j, output_directory)
            inst_len += length
            if ind % self.H == 0:
                print("Memories: ", len(self.memory_bank))
                print("Instantaneous interference rate: ", inst_inters / self.H)
                # print(
                #     "Average interference rate: ", total_inters / len(self.memory_bank)
                # )
                print("Average size of memories created: ", inst_len / self.H, "\n\n")
                inst_inters = 0
                inst_len = 0
                if vis:
                    self._visualize(
                        os.path.join(
                            output_directory,
                            f"graph_{len(self.memory_bank)}_memories.png",
                        )
                    )
                    self._visualize_n(
                        os.path.join(
                            output_directory,
                            f"graph_{len(self.memory_bank)}_n_memories.png",
                        )
                    )
                print(
                    len(self.interference_counts),
                    self.interference_counts[0 : self.START_MEM],
                    self.interference_counts[self.START_MEM :],
                )
            if inter_flag > 0:
                total_inters += inter_flag
                inst_inters += inter_flag
                # if inst_inters / self.H > self.STOP:
                if total_inters / len(self.memory_bank) > self.STOP:
                    print(
                        "Config: N=",
                        self.N,
                        " D=",
                        self.D,
                        " k=",
                        self.k,
                        " k_adj=",
                        self.k_adj,
                        " R=",
                        self.r_exp,
                        "START_MEM=",
                        self.START_MEM,
                    )
                    print(
                        "Halting memory formation at ",
                        len(self.memory_bank),
                        " memories due to more than ",
                        self.STOP * 100,
                        "percent average total interference",
                    )
                    print("Instantaneous interference rate: ", inst_inters / self.H)
                    print(
                        "Average interference rate: ",
                        total_inters / len(self.memory_bank),
                    )
                    print(
                        len(self.interference_counts),
                        self.interference_counts[0 : self.START_MEM],
                        self.interference_counts[self.START_MEM :],
                    )
                    if vis:
                        self._visualize(
                            os.path.join(
                                output_directory,
                                f"graph_final_memories.png",
                            )
                        )
                        self._visualize_n(
                            os.path.join(
                                output_directory,
                                f"graph_final_n_memories.png",
                            )
                        )
                    break

        return self
