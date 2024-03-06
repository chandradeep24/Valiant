import numpy as np
from numpy.random import default_rng

try:
    import graph_tool.all as gt
except ImportError:
    print(
        'Graph-Tool not found. Please install it to use the NeuroidalModel class.')
    exit(1)


class Backend:
    ERDOS_RENYI = 'erdos_renyi'
    WATTS_STROGATZ = 'watts_strogatz'

    @staticmethod
    def create_gnp_graph(n: int, p: float, seed=None) -> gt.Graph:
        """Returns an Erdos-Reyni random graph.

        Parameters
        ----------
        n : int
            The number of nodes
        p : float
            The probability of creating an edge between any two nodes
        seed : int, optional
            Seed for the random number generator

        See Also
        --------
        random_graph
        binomial_graph

        Notes
        -----
        First create a graph with $n$ nodes and no edges. Then for each of the
        $\binom{n}{2}$ possible edges, add the edge with probability $p$.

        References
        ----------
        - P. Erdős and A. Rényi, On Random Graphs, Publ. Math. 6, 290 (1959).
        - E. N. Gilbert, Random Graphs, Ann. Math. Stat., 30, 1141 (1959).
        """

        g = gt.Graph(directed=True)
        g.add_vertex(n)

        rng = default_rng(seed)

        k = rng.binomial(n * (n - 1) // 2, p)

        sources = rng.integers(0, n, k * 2)
        targets = rng.integers(0, n, k * 2)
        mask = sources != targets  # removes self-loops
        g.add_edge_list(np.column_stack((sources[mask], targets[mask])))

        return g

    @staticmethod
    def create_sw_graph(n: int, k: int, p: float, seed=None) -> gt.Graph:
        """Returns a Watts–Strogatz small-world graph.

        Parameters
        ----------
        n : int
            The number of nodes
        k : int
            Each node is joined with its `k` nearest neighbors in a ring
            topology.
        p : float
            The probability of rewiring each edge
        seed : int, optional
            Seed for the random number generator

        See Also
        --------
        newman_watts_strogatz_graph
        connected_watts_strogatz_graph

        Notes
        -----
        First create a ring over $n$ nodes [1]_.  Then each node in the ring is joined
        to its $k$ nearest neighbors (or $k - 1$ neighbors if $k$ is odd).
        Then shortcuts are created by replacing some edges as follows: for each
        edge $(u, v)$ in the underlying "$n$-ring with $k$ nearest neighbors"
        with probability $p$ replace it with a new edge $(u, w)$ with uniformly
        random choice of existing node $w$.

        In contrast with :func:`newman_watts_strogatz_graph`, the random rewiring
        does not increase the number of edges. The rewired graph is not guaranteed
        to be connected as in :func:`connected_watts_strogatz_graph`.

        References
        ----------
        - D. J. Watts and Steven Strogatz,
        “Collective dynamics of ‘small-world’ networks”,
        Nature, vol. 393, pp 440-442, 1998.
        """
        if k > n:
            raise Exception("k cannot be larger than n")

        # If k == n, then the graph is complete, not Watts-Strogatz
        if k == n:
            return gt.complete_graph(n)

        rng = default_rng(seed)
        g = gt.Graph(directed=False)

        # Add n nodes
        nodes = list(g.add_vertex(n))

        print('Add Nodes')

        # Connect each node to k nearest neighbors
        for n_ix in range(n):
            for k_ix in range(n + 1, n + k + 1):
                g.add_edge(nodes[n_ix], nodes[k_ix % n])
            # targets = nodes[n_ix + 1:n_ix + k + 1]
            # g.add_edge_list(zip(nodes, targets))

        print('Connect each node to k neighbors')

        # Rewire edges from each node based on p
        # No self-loop or multiple edges allowed
        rewire_count = rng.binomial(n * (n - 1) // 2, p)
        gt.random_rewire(g, 'configuration', rewire_count, edge_sweep=False)

        print('Rewire edges from each node based on p')
        return g

if __name__ == "__main__":
    # Backend.create_gnp_graph(1000, 0.5)
    g = Backend.create_sw_graph(100, 25, 0.5)

    pos = gt.sfdp_layout(g, cooling_step=0.95)
    gt.graph_draw(g, pos=pos, output='output.png')