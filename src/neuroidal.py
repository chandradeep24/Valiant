import gc, os, sys
from math import comb
import numpy as np
from numpy.random import *
import graph_tool as gt


seed(42)
np.random.seed(42)
gt.seed_rng(42)


"""TODO: refactor the notebook functions into this class"""


N = 500
D = 128
T = 1
k = 16
k_adj = 1.55
P = D / (N - 1)

H = 100
STOP = 0.25
START_MEM = 100
r_expected = 40


# _ indicates private function
# public functions should return self


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

    # Create the graph with the properties
    def create(self):
        return self

    # Check and fire
    def _check_and_fire(self):
        pass

    # Check for interference
    def _interference_check(self):
        pass

    # JOIN function
    def JOIN_one_step_shared(self):
        return self
