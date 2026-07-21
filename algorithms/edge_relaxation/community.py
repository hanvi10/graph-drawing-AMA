"""
algorithms/edge_relaxation_community.py
========================================
Community Membership Distance.

Runs Louvain community detection, then scores each edge based on whether
it connects nodes in different communities. Inter-community edges are the
layout-distorting bridges the paper targets — this method finds them via
mesoscale structure rather than path counting.

Scoring: inter-community edges get 1 + normalized_EBC; intra-community
edges get 0 + normalized_EBC. This separates the two classes while using
EBC as a tie-breaker within each class.
"""

import networkx as nx
import numpy as np

from algorithms.base import GraphDrawingAlgorithm
from metrics import count_crossings


class EdgeRelaxationCommunity(GraphDrawingAlgorithm):
    """
    Edge Relaxation using community membership as edge score (C1).
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
        return f"edge_relaxation_community(kr={self.k_r},kw={self.k_w})"

    def layout(self, G: nx.Graph) -> dict:
        G = G.copy()
        np.random.seed(self.seed)

        pos = self._initial_layout(G, self.seed, self.initial_layout_iterations)

        communities = nx.community.louvain_communities(G, seed=self.seed)
        node_community = {}
        for comm_id, comm in enumerate(communities):
            for node in comm:
                node_community[node] = comm_id

        raw_eb = nx.edge_betweenness_centrality(G, normalized=True)
        eb = {e: raw_eb.get(e, raw_eb.get((e[1], e[0]), 0.0)) for e in G.edges()}

        base_score = {}
        for u, v in G.edges():
            inter = 1.0 if node_community[u] != node_community[v] else 0.0
            base_score[(u, v)] = inter + eb[(u, v)]

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
