"""
algorithms/edge_relaxation_angular.py
=======================================
Inverse Angular Resolution Score.

Edges that form very small angles with their neighboring edges at each endpoint
are visually crowded and have high crossing risk. Score = 1 / (min_angle + ε),
so edges in tight angular clusters score highest and are relaxed first.
This targets the angular resolution aesthetic criterion directly.
"""

import math

import networkx as nx
import numpy as np

from algorithms.base import GraphDrawingAlgorithm
from metrics import count_crossings


def _min_incident_angle(G, pos, u, v) -> float:
    """Minimum angle between edge (u,v) and any other edge incident to u or v."""
    cx, cy = pos[u]
    tx, ty = pos[v]
    main_vec = (tx - cx, ty - cy)
    main_len = math.hypot(*main_vec)

    angles = []

    if main_len > 0:
        for n in G.neighbors(u):
            if n == v:
                continue
            nx_, ny_ = pos[n]
            other_vec = (nx_ - cx, ny_ - cy)
            other_len = math.hypot(*other_vec)
            if other_len == 0:
                continue
            cos_a = (main_vec[0] * other_vec[0] + main_vec[1] * other_vec[1]) / (main_len * other_len)
            angles.append(math.degrees(math.acos(max(-1.0, min(1.0, cos_a)))))

    cx2, cy2 = pos[v]
    tx2, ty2 = pos[u]
    rev_vec = (tx2 - cx2, ty2 - cy2)
    rev_len = math.hypot(*rev_vec)

    if rev_len > 0:
        for n in G.neighbors(v):
            if n == u:
                continue
            nx_, ny_ = pos[n]
            other_vec = (nx_ - cx2, ny_ - cy2)
            other_len = math.hypot(*other_vec)
            if other_len == 0:
                continue
            cos_a = (rev_vec[0] * other_vec[0] + rev_vec[1] * other_vec[1]) / (rev_len * other_len)
            angles.append(math.degrees(math.acos(max(-1.0, min(1.0, cos_a)))))

    return min(angles) if angles else 180.0


class EdgeRelaxationAngular(GraphDrawingAlgorithm):
    """
    Edge Relaxation using inverse minimum angular resolution as edge score (A3).
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
        return f"edge_relaxation_angular(kr={self.k_r},kw={self.k_w})"

    def layout(self, G: nx.Graph) -> dict:
        G = G.copy()
        np.random.seed(self.seed)

        pos = self._initial_layout(G, self.seed, self.initial_layout_iterations)

        eps = 1e-6
        base_score = {(u, v): 1.0 / (_min_incident_angle(G, pos, u, v) + eps)
                      for u, v in G.edges()}

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
