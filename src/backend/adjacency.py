import scipy.sparse as sp
import numpy as np
from numpy.random import default_rng

print('Imported adjacency Neuroidal Model Backend')


class Backend:
    @staticmethod
    def graph_from_array(data: np.ndarray) -> sp.csr_matrix:
        adj_matrix = sp.coo_matrix(data, data.shape, dtype=np.int8)
        adj_matrix.setdiag(0)
        return adj_matrix.tocsr()

    @staticmethod
    def create_gnm_graph(n: int, p: float, seed=None) -> sp.csr_matrix:
        rng = default_rng(seed)
        k = rng.binomial(n * (n - 1) // 2, p)
        sources = rng.integers(0, n, k * 2)
        targets = rng.integers(0, n, k * 2)

        adj_matrix = sp.coo_matrix(
            (np.ones(shape=sources.shape), (targets, sources)), (n, n),
            dtype=np.int8)
        adj_matrix.setdiag(0)
        return adj_matrix.tocsr()

    @staticmethod
    def create_gnp_graph(n: int, p: float, seed=None) -> np.ndarray:
        rng = default_rng(seed)
        data = rng.binomial(1, p, size=(n, n))
        np.fill_diagonal(data, 0)
        return data
        # data = np.zeros(shape=(n, n), dtype=np.uint8)
        # for i in range(n):
        #     data[i, :i] = rng.binomial(1, p, size=i)
        # return data
        # adj_matrix = sp.coo_matrix(data, (n, n), dtype=np.int8)
        # adj_matrix.setdiag(0)
        # return adj_matrix.tocsr()

    @staticmethod
    def create_ws_graph(n: int, p: float, k: int = -1, directed: bool = True,
                        seed=None) -> sp.coo_matrix:
        if k == -1:
            k = int((n + np.log(n)) // 2)

        if k > n:
            raise Exception("k cannot be larger than n")

        if k == n:
            return sp.eye(n, dtype=np.uint8)

        rng = default_rng(seed)
        data = np.zeros(shape=(n, n), dtype=np.uint8)
        for n_ix in range(n):
            for n_iy in range(n_ix + 1, n_ix + k + 1):
                if rng.uniform() < p:
                    r_ix = rng.choice(list(set(range(n)) - {n_ix, n_iy % n}))
                    data[r_ix, n_ix] = 1
                else:
                    data[n_iy % n, n_ix] = 1

        adj_matrix = sp.coo_matrix(data, (n, n), dtype=np.uint8)
        return adj_matrix.tocsr()


    @staticmethod
    def create_ba_graph(n: int, m: int, seed=None) -> sp.csr_matrix:
        if m < 1 or m >= n:
            raise Exception(
                "Barabási–Albert network must have m >= 1 and m < n, m = %d, n = %d" % (
                    m, n))

        rng = default_rng(seed)
        data = np.zeros(shape=(n, n), dtype=np.uint8)
        # Connect node 0 to nodes 1 to m
        data[1:m + 1, 0] = 1
        data[0, 1:m + 1] = 1
        for n_ix in range(m + 1, n):
            p = np.sum(data[:n_ix, :], axis=1) / np.sum(data)

            np.put(data[:, n_ix],
                   rng.choice(np.arange(n_ix), size=m, p=p, replace=False,
                              shuffle=True), 1)
            data[n_ix, :] = data[:, n_ix]

        adj_matrix = sp.coo_matrix(data, (n, n), dtype=np.uint8)
        return adj_matrix.tocsr()

    @staticmethod
    def calculate_local_clustering(graph: sp.csr_matrix) -> float:
        # Returns the Local Clustering Coefficient
        # As defined upon an undirected graph, no matter the result
        A = graph.toarray()
        A = np.maximum(A, A.T)

        k_i = np.sum(A, axis=1)
        if np.all(k_i == 0) or np.all(k_i == 1):
            return 0
        k_j = k_i * (k_i - 1)

        B = np.diag(A @ A @ A)
        return np.sum(np.divide(1, k_j, where=k_j != 0) * B) / A.shape[0]

    @staticmethod
    def calculate_global_clustering(graph: sp.csr_matrix) -> float:
        # The Global Clustering Coefficient is defined as
        # Closed Triples / All Triplets

        A = graph.toarray()
        A = np.maximum(A, A.T)
        k = np.sum(A @ A) - np.trace(A @ A)
        if k == 0:
            return 0
        return np.trace(A @ A @ A) / k

    @staticmethod
    def calculate_path_length(g: sp.csr_matrix) -> float:
        A = g.toarray()
        A = np.maximum(A, A.T)

        d = np.array(A, dtype=float)
        d[d == 0] = np.inf
        np.fill_diagonal(d, 0)

        for k in range(g.shape[0]):
            d = np.minimum(d, d[:, k:k + 1] + d[k:k + 1, :])

        return np.mean(d[d != 0], axis=0)

