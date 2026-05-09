"""
algorithms/edge_relaxation_embeddedness.py
===========================================
Inverse Neighborhood Overlap (Embeddedness).

An edge is "embedded" if its endpoints share many common neighbors (Jaccard
similarity of neighborhoods). Low embeddedness = bridge-like edge with few
shared neighbors = inter-cluster connection. Score = 1 - Jaccard(N(u), N(v)).

This is a fast local approximation to betweenness: O(m·d) instead of O(nm),
and captures local bridge structure without computing global path counts.
"""

import networkx as nx
import numpy as np

from algorithms.base import GraphDrawingAlgorithm
from metrics import count_crossings


class EdgeRelaxationEmbeddedness(GraphDrawingAlgorithm):
    """
    Edge Relaxation using inverse neighborhood overlap as edge score (B2).
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
        return f"edge_relaxation_embeddedness(kr={self.k_r},kw={self.k_w})"

    def layout(self, G: nx.Graph) -> dict:
        G = G.copy()
        np.random.seed(self.seed)

        pos = self._initial_layout(G, self.seed, self.initial_layout_iterations)

        base_score = {}
        for u, v in G.edges():
            nu = set(G.neighbors(u)) - {v}
            nv = set(G.neighbors(v)) - {u}
            union = len(nu | nv)
            jaccard = len(nu & nv) / union if union > 0 else 0.0
            base_score[(u, v)] = 1.0 - jaccard

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
