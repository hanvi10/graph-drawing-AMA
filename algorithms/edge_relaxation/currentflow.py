"""
algorithms/edge_relaxation_currentflow.py
==========================================
Edge Current Flow Betweenness (ECFB).

Uses an electrical resistance model instead of shortest paths. Each edge is
treated as a resistor; its betweenness is the expected fraction of current
flowing through it across all source/sink pairs. Unlike EBC, ECFB considers
ALL paths weighted by random-walk probability — it detects soft bottlenecks
that EBC misses because they are not on any shortest path.

Note: O(n³) computation — significantly slower than EBC for large graphs.
Requires a connected graph (guaranteed by the dataset preprocessing).
"""

import networkx as nx
import numpy as np

from algorithms.base import GraphDrawingAlgorithm
from metrics import count_crossings


class EdgeRelaxationCurrentFlow(GraphDrawingAlgorithm):
    """
    Edge Relaxation using Edge Current Flow Betweenness as edge score (B1).
    """

    def __init__(
        self,
        k_r: float = 0.1,
        k_w: float = 0.05,
        max_iter: int = 100,
        patience: int = 20,
        seed: int = 42,
        initial_layout_iterations: int = 50,
        loop_spring_iters: int = 50,
    ):
        self.k_r = k_r
        self.k_w = k_w
        self.max_iter = max_iter
        self.patience = patience
        self.seed = seed
        self.initial_layout_iterations = initial_layout_iterations
        self.loop_spring_iters = loop_spring_iters

    @property
    def name(self) -> str:
        return f"edge_relaxation_currentflow(kr={self.k_r},kw={self.k_w})"

    def layout(self, G: nx.Graph) -> dict:
        G = G.copy()
        np.random.seed(self.seed)

        pos = self._initial_layout(G, self.seed, self.initial_layout_iterations)

        raw_ecfb = nx.edge_current_flow_betweenness_centrality(G, normalized=False)
        ecfb = {}
        for e in G.edges():
            ecfb[e] = raw_ecfb.get(e, raw_ecfb.get((e[1], e[0]), 0.0))

        scale  = {e: 1.0 for e in G.edges()}
        weight = {e: 1.0 for e in G.edges()}
        scores = {e: weight[e] * ecfb[e] for e in G.edges()}

        best_crossings    = np.inf
        best_crossings_it = -1
        best_pos          = None
        last_pos          = pos.copy()

        for it in range(self.max_iter):
            selected = max(scores, key=scores.get)

            scale[selected]  *= self.k_r
            weight[selected] *= self.k_w
            scores[selected]  = weight[selected] * ecfb[selected]

            G[selected[0]][selected[1]]['relax'] = scale[selected]

            pos = nx.spring_layout(G, pos=last_pos.copy(), weight='relax',
                                   iterations=self.loop_spring_iters)
            crossings = count_crossings(G, pos)

            if crossings < best_crossings:
                best_crossings    = crossings
                best_crossings_it = it
                best_pos          = pos

            if best_crossings_it + self.patience < it:
                break

            last_pos = pos

        return best_pos if best_pos is not None else pos
