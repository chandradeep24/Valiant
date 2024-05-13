import numpy as np
import graph_tool.all as gt
from numpy.random import default_rng

print('Imported graph_tool Neuroidal Model Backend')

class Backend:
    @staticmethod
    def to_adjacency(g: gt.Graph) -> np.ndarray:
        return gt.adjacency(g).toarray()

    @staticmethod
    def create_gnm_graph(n: int, p: float, seed=None) -> gt.Graph:
        rng = default_rng(seed)
        k = rng.binomial(n * (n - 1) // 2, p)
        sources = rng.integers(0, n, k * 2)
        targets = rng.integers(0, n, k * 2)
        mask = sources != targets

        g = gt.Graph(directed=True)
        g.add_vertex(n)
        g.add_edge_list(np.column_stack((sources[mask], targets[mask])))

        return g

    @staticmethod
    def create_gnp_graph(n: int, p: float, seed=None) -> gt.Graph:
        rng = default_rng(seed)

        g = gt.Graph(directed=True)
        g.add_vertex(n)

        for i in range(n):
            for j in range(i):
                if rng.uniform() < p:
                    g.add_edge(i, j)

        return g

    @staticmethod
    def create_ws_graph(n: int, p: float, k: int = -1, seed=None) -> gt.Graph:
        if k == -1:
            k = int((n + np.log(n)) // 2)

        if k > n:
            raise Exception("k cannot be larger than n")

        if k == n:
            return gt.complete_graph(n)

        g = gt.Graph()

        rng = default_rng(seed)
        for n_ix in range(n):
            for n_iy in range(n_ix + 1, n_ix + k + 1):
                if rng.uniform() < p:
                    g.add_edge(n_ix, rng.choice(list(set(range(n)) - {n_ix, n_iy % n})))
                else:
                    g.add_edge(n_iy % n, n_ix)

        return g


    @staticmethod
    def create_ba_graph(n: int, m: int, seed=None) -> gt.Graph:
        if m < 1 or m >= n:
            raise Exception(
                "Barabási–Albert network must have m >= 1 and m < n, m = %d, n = %d" % (
                    m, n))

        g = gt.Graph(directed=True)

        for ix in range(1, m + 1):
            g.add_edge(0, ix)
            g.add_edge(ix, 0)

        rng = default_rng(seed)
        # Connect node 0 to nodes 1 to m
        for n_ix in range(m + 1, n):
            p = np.zeros(n_ix, dtype=float)
            for n_iy in range(n_ix):
                p[n_iy] = g.get_out_degrees(g.vertex(n_iy))
            p /= g.num_edges()

            for n_iy in rng.choice(np.arange(n_ix), size=m, p=p, replace=False, shuffle=True):
                g.add_edge(n_ix, n_iy)
                g.add_edge(n_iy, n_ix)

        return g

    @staticmethod
    def calculate_local_clustering(g: gt.Graph) -> float:
        mean, _ = gt.vertex_average(g, gt.local_clustering(g))
        return mean

    @staticmethod
    def calculate_global_clustering(g: gt.Graph) -> float:
        return gt.global_clustering(g, sampled=False)

    @staticmethod
    def calculate_path_length(g: gt.Graph) -> float:
        mean, _ = gt.vertex_average(g, gt.shortest_distance(g))
        return mean

