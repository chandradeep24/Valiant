import gc, os, sys
from math import comb
import numpy as np
from numpy.random import *
import graph_tool.all as gt


seed(42)
np.random.seed(42)
gt.seed_rng(42)


"""TODO: fix issues and add more comments"""

# _ indicates private function
# public functions should return self to be composable


class NeuroidalModel:
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

    # Create the graph with the properties
    def create_graph(self):
        self.g = gt.Graph()
        self.g.add_vertex(self.N)

        # Generating far less edges now
        self.num_edges = self.P * comb(self.N, 2)
        gt.add_random_edges(self.g, self.num_edges, parallel=False, self_loops=False)

        gt.random_rewire(
            self.g, model="erdos", parallel_edges=False, self_loops=False, verbose=False
        )

        self.vprop_fired = self.g.new_vertex_property("int")
        self.vprop_memories = self.g.new_vertex_property("int")
        self.vprop_fired_now = self.g.new_vertex_property("int")
        self.vprop_weight = self.g.new_vertex_property("double")
        self.vprop_threshold = self.g.new_vertex_property("double")

        self.vprop_fired.a = 0
        self.vprop_memories.a = 0
        self.vprop_fired_now.a = 0
        self.vprop_weight.a = 0.0
        self.vprop_threshold.a = self.T

        self.eprop_fired = self.g.new_edge_property("int")
        self.eprop_weight = self.g.new_edge_property("double")

        self.eprop_fired.a = 0
        self.eprop_weight.a = self.T / (self.k_adj * self.k)

        return self

    def _generate_adj_mat(self):
        self.adjmat = gt.adjacency(self.g)
        return self

    def generate_memory_bank(self):
        self.memory_bank = []
        for i in np.arange(0, self.START_MEM):
            memory_A = np.random.default_rng().choice(
                np.arange(0, self.N - 1), size=self.r_exp
            )
            self.memory_bank.append(memory_A)
        return self

    # Check for interference
    def _interference_check(self, A_index, B_index, C):
        sum = 0
        for i in range(len(self.memory_bank)):
            if i != A_index and i != B_index:
                # print(type(C), type(i))
                inter = list(set(C) & set(self.memory_bank[i]))
                if len(inter) > len(self.memory_bank[i]) / 2:
                    sum += 2
        return sum

    # Vectorized JOIN function
    # JOIN function
    # def _JOIN_one_step_shared(self, i, j):
    #     self._generate_adj_mat()

    #     memory_A = self.memory_bank[i]
    #     memory_B = self.memory_bank[j]

    #     state = np.zeros(self.adjmat.shape[0])
    #     state[memory_A] = 1
    #     state[memory_B] = 1

    #     fired = np.heaviside((self.adjmat @ state) - 1, 1)

    #     # This needs to be changed to comparizon with threshold
    #     memory_C = np.nonzero(fired)[0]

    #     print(type(memory_C))

    #     inter = self._interference_check(i, j, memory_C)
    #     self.memory_bank.append(memory_C)

    #     return inter, len(memory_C)

    def _check_and_fire_and_add(self, v, memory_C):
        sum = 0
        for s, t in self.g.iter_in_edges(v):
            if self.vprop_fired_now[s] > 0:
                sum += self.eprop_weight[self.g.edge(s, t)]
        if sum > self.vprop_threshold[v]:
            self.vprop_fired[v] += 1
            memory_C.append(v)

    def _JOIN_one_step_shared(self, i, j):
        """
        Choose two random groups of neurons to become A and B
        Basing this on the expected value of r from Valiant (2005)
        Set A, then B to fire
        Trace C from the firing nodes outward from A and B
        Check for interference
        """

        memory_A = self.memory_bank[i]
        memory_B = self.memory_bank[j]

        # Fire A
        for v in memory_A:
            self.vprop_fired_now[v] = 1
            self.vprop_fired[v] += 1
            self.vprop_memories[v] += 1

        # Fire B
        for v in memory_B:
            self.vprop_fired_now[v] = 1
            self.vprop_fired[v] += 1
            self.vprop_memories[v] += 1

        memory_C = []
        # Check and fire adjacent nodes:
        for v in self.g.iter_vertices():
            self._check_and_fire_and_add(v, memory_C)

        inter = self._interference_check(i, j, memory_C)
        self.vprop_fired.a = 0
        self.vprop_fired_now.a = 0
        self.memory_bank.append(memory_C)

        return inter, len(memory_C)

    def simulate(self):
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

        for pair in pairs:
            ind += 1
            i = pair[0]
            j = pair[1]
            inter_flag, length = self._JOIN_one_step_shared(i, j)
            inst_len += length
            if ind % self.H == 0:
                print("Memories: ", len(self.memory_bank))
                print("Instantaneous interference rate: ", inst_inters / self.H)
                print(
                    "Average interference rate: ", total_inters / len(self.memory_bank)
                )
                print("Average size of memories created: ", inst_len / self.H, "\n\n")
                inst_inters = 0
                inst_len = 0
            if inter_flag > 0:
                total_inters += inter_flag
                inst_inters += inter_flag
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
                        "percent total interference",
                    )
                    print("Instantaneous interference rate: ", inst_inters / self.H)
                    print(
                        "Average interference rate: ",
                        total_inters / len(self.memory_bank),
                    )
                    break

        return self
