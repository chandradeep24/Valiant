try:
    import cupy as xp
except ImportError:
    import numpy as xp

print('Imported adjacency Neuroidal Model Backend')

class Backend:
    @staticmethod
    def create_gnp_graph(n: int, p: float, seed=None) -> xp.ndarray:
        rng = xp.random.default_rng(seed)
        data = rng.binomial(1, p, size=(n, n))
        xp.fill_diagonal(data, 0)
        return data

    @staticmethod
    def create_gnm_graph(n: int, p: float, seed=None) -> xp.ndarray:
        rng = xp.random.default_rng(seed)
        k = rng.binomial(n * (n - 1) // 2, p)

        sources = rng.integers(0, n, k * 2)
        targets = rng.integers(0, n, k * 2)

        data = xp.zeros(shape=(n, n), dtype=xp.uint8)
        for source, target in zip(sources, targets):
            data[source, target] = 1
        return data

    @staticmethod
    def create_ws_graph(n: int, p: float, k: int = -1, directed: bool = True,
                        seed=None) -> xp.ndarray:
        if k == -1:
            k = int((n + xp.log(n)) // 2)

        if k > n:
            raise Exception("k cannot be larger than n")

        if k == n:
            return xp.eye(n, dtype=xp.uint8)

        rng = xp.random.default_rng(seed)
        data = xp.zeros(shape=(n, n), dtype=xp.uint8)
        for n_ix in range(n):
            for n_iy in range(n_ix + 1, n_ix + k + 1):
                if rng.uniform() < p:
                    r_ix = rng.choice(list(set(range(n)) - {n_ix, n_iy % n}))
                    data[r_ix, n_ix] = 1
                else:
                    data[n_iy % n, n_ix] = 1
        return data

    @staticmethod
    def create_ba_graph(n: int, m: int, seed=None) -> xp.ndarray:
        if m < 1 or m >= n:
            raise Exception(
                "Barabási–Albert network must have m >= 1 and m < n, m = %d, n = %d" % (
                    m, n))

        rng = xp.random.default_rng(seed)
        data = xp.zeros(shape=(n, n), dtype=xp.uint8)
        # Connect node 0 to nodes 1 to m
        data[1:m + 1, 0] = 1
        data[0, 1:m + 1] = 1
        for n_ix in range(m + 1, n):
            p = xp.sum(data[:n_ix, :], axis=1) / xp.sum(data)

            xp.put(data[:, n_ix],
                   rng.choice(xp.arange(n_ix), size=m, p=p, replace=False,
                              shuffle=True), 1)
            data[n_ix, :] = data[:, n_ix]
        return data

    @staticmethod
    def calculate_local_clustering(A, directed=False) -> float:
        if not directed:
            A = xp.maximum(A, A.T)

        k_i = xp.sum(A, axis=1)
        if xp.all(k_i == 0) or xp.all(k_i == 1):
            return 0
        k_j = k_i * (k_i - 1)

        B = xp.diag(A @ A @ A)
        return xp.sum(xp.divide(1, k_j, where=k_j != 0) * B) / A.shape[0]

    @staticmethod
    def calculate_global_clustering(A, directed=False) -> float:
        if not directed:
            A = xp.maximum(A, A.T)

        k = xp.sum(A @ A) - xp.trace(A @ A)
        if k == 0:
            return 0
        return xp.trace(A @ A @ A) / k

    @staticmethod
    def calculate_path_length(A, directed=False) -> float:
        if not directed:
            A = xp.maximum(A, A.T)
        else:
            A = xp.array(A, copy=True)

        A[A == 0] = xp.inf
        xp.fill_diagonal(A, 0)

        for k in range(A.shape[0]):
            A = xp.minimum(A, A[:, k:k + 1] + A[k:k + 1, :])

        return xp.mean(A[A != 0], axis=0)

