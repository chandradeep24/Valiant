import itertools

try:
    import cupy as xp
except ImportError:
    import numpy as xp

print('Imported adjacency Neuroidal Model Backend')

class Backend:
    @staticmethod
    def create_gnp_graph(n: int, p: float, seed=None) -> xp.ndarray:
        data = (xp.random.random_sample(size=(n,  n)) < p).astype(float)
        xp.fill_diagonal(data, 0)
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

        xp.random.seed(seed)
        data = xp.zeros(shape=(n, n), dtype=xp.uint8)

        pool = xp.random.random_sample(size=n * k)
        for pix, (n_ix, iy) in enumerate(itertools.product(range(n), range(1, k + 1))):
            idx = n_iy = (n_ix + iy) % n
            if pool[pix] < p:
                while idx == n_ix or idx == n_iy:
                    idx = int(xp.random.random_sample() * n)
            data[idx, n_ix] = 1
        return data

    @staticmethod
    def create_ba_graph(n: int, m: int, seed=None) -> xp.ndarray:
        if m < 1 or m >= n:
            raise Exception(
                "Barabási–Albert network must have m >= 1 and m < n, m = %d, n = %d" % (
                    m, n))

        xp.random.seed(seed)
        data = xp.zeros(shape=(n, n), dtype=xp.uint8)
        # Connect node 0 to nodes 1 to m
        data[1:m + 1, 0] = 1
        data[0, 1:m + 1] = 1
        for n_ix in range(m + 1, n):
            p = xp.sum(data[:n_ix, :], axis=1) / xp.sum(data)
            xp.put(data[:, n_ix], xp.random.choice(xp.arange(n_ix), size=m, p=p), 1)
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

        A = xp.array(A, dtype=float, copy=True)
        A[A == 0] = xp.inf
        xp.fill_diagonal(A, 0)

        for k in range(A.shape[0]):
            A = xp.minimum(A, A[:, k:k + 1] + A[k:k + 1, :])

        return xp.mean(A[A != 0], axis=0).item()

    @staticmethod
    def calculate_stats_quick(A, directed=False) -> xp.ndarray:
        if not directed:
            A = xp.maximum(A, A.T, dtype=float)
        else:
            A = xp.array(A, dtype=float, copy=True)

        k_i = xp.sum(A, axis=0)
        k_i *= k_i - 1
        kd = xp.sum(k_i)
        k_i[k_i != 0] = 1 / k_i[k_i != 0]

        A3 = A @ A @ A

        r = xp.zeros(3)
        r[0] = xp.sum(k_i * xp.diag(A3)) / A.shape[0]
        r[1] = xp.trace(A3) / kd

        A[A == 0] = xp.inf
        xp.fill_diagonal(A, 0)

        for k in range(A.shape[0]):
            A = xp.minimum(A, A[:, k:k + 1] + A[k:k + 1, :])

        r[2] = xp.mean(A[A != 0], axis=0).item()
        return r