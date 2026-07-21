"""
algorithms/edge_relaxation_stress.py
======================================
Geometric Stress Score.

An edge's stress is the squared deviation of its drawn length from the mean
edge length. Stressed edges are either stretched (pulling nodes too far apart)
or compressed (pushing nodes too close together) relative to the global average.
Relaxing high-stress edges releases distorting forces and lets the layout breathe.
"""

import math

import networkx as nx
import numpy as np

from algorithms.base import GraphDrawingAlgorithm
from metrics import count_crossings


class EdgeRelaxationStress(GraphDrawingAlgorithm):
    """
    Edge Relaxation using geometric stress (deviation from mean edge length) as score (A2).
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
        return f"edge_relaxation_stress(kr={self.k_r},kw={self.k_w})"

    def layout(self, G: nx.Graph) -> dict:
        G = G.copy()
        np.random.seed(self.seed)

        pos = self._initial_layout(G, self.seed, self.initial_layout_iterations)

        lengths = {(u, v): math.hypot(pos[u][0] - pos[v][0], pos[u][1] - pos[v][1])
                   for u, v in G.edges()}
        mean_len = np.mean(list(lengths.values())) if lengths else 1.0
        if mean_len == 0:
            mean_len = 1.0

        base_score = {e: ((lengths[e] / mean_len) - 1.0) ** 2 for e in G.edges()}

        scale  = {e: 1.0 for e in G.edges()}
        weight = {e: 1.0 for e in G.edges()}
        scores = {e: weight[e] * base_score[e] for e in G.edges()}

        best_crossings    = np.inf
        best_crossings_it = -1
        best_pos          = None
        last_pos          = pos.copy()

        for it in range(self.max_iter):
            selected = max(scores, key=scores.get)

            scale[selected]  *= self.k_r
            weight[selected] *= self.k_w
            scores[selected]  = weight[selected] * base_score[selected]

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
