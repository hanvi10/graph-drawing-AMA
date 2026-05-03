"""
algorithms/edge_relaxation.py
==============================
The paper's algorithm: Graph Drawing using Edge Relaxation.
Ported faithfully from the original src/experiments/run_experiment.py
and src/graph_utils.py in graph-drawing-edge-relaxation/.

Core idea:
    Some edges act as "bridges" between clusters and force the layout to
    distort, creating many crossings. By gradually reducing the weight of
    these bridge edges, the force-directed algorithm can spread the clusters
    apart and reduce crossings.

    Bridge edges are identified using Edge Betweenness Centrality (EBC):
    an edge has high EBC if many shortest paths in the graph pass through it.

Two separate dicts (matching the original exactly):
    scale[e]  -- the spring weight used in nx.spring_layout.
                 Multiplied by k_r each time edge e is selected.
                 Controls how strongly the edge pulls its two nodes together.

    weight[e] -- the selection score multiplier.
                 Multiplied by k_w each time edge e is selected.
                 Controls how likely e is to be selected again.

    Selection score for edge e = weight[e] * EBC[e]

Stopping condition (from original):
    Stop when: best_crossing_iteration + patience < current_iteration
    i.e. no improvement for more than `patience` steps.
"""

import networkx as nx
import numpy as np
from algorithms.base import GraphDrawingAlgorithm
from metrics import count_crossings


class EdgeRelaxation(GraphDrawingAlgorithm):
    """
    Graph Drawing using Edge Relaxation (Higueras, Escofet, Cortadella 2021).
    Faithful port of getLayout() from the original run_experiment.py.
    """

    def __init__(
        self,
        k_r: float = 0.1,    # relaxation constant: how much to reduce spring weight
        k_w: float = 0.05,   # selection constant:  how much to reduce re-selection chance
        max_iter: int = 100,  # hard cap on iterations
        patience: int = 20,   # stop if no improvement for this many steps
        seed: int = 42,
        initial_layout_iterations: int = 50,
        loop_spring_iters: int = 50,  # spring iters per relaxation step; warm start → fewer needed
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
        return f"edge_relaxation(kr={self.k_r},kw={self.k_w})"

    def layout(self, G: nx.Graph) -> dict:
        """
        Run edge relaxation and return the best layout found.
        Mirrors getLayout() from the original run_experiment.py.
        """
        G = G.copy()
        np.random.seed(self.seed)

        pos = self._initial_layout(G, self.seed, self.initial_layout_iterations)

        # Compute EBC once; normalize keys to canonical (u,v) edge direction
        raw_eb = nx.edge_betweenness_centrality(G, normalized=False)
        eb = {}
        for e in G.edges():
            eb[e] = raw_eb.get(e, raw_eb.get((e[1], e[0]), 0.0))

        scale  = {e: 1.0 for e in G.edges()}
        weight = {e: 1.0 for e in G.edges()}

        # Pre-compute scores once; only the selected edge changes each iteration
        scores = {e: weight[e] * eb[e] for e in G.edges()}

        best_crossings    = np.inf
        best_crossings_it = -1
        best_pos          = None
        last_pos          = pos.copy()

        for it in range(self.max_iter):
            selected = max(scores, key=scores.get)

            scale[selected]  *= self.k_r
            weight[selected] *= self.k_w
            scores[selected]  = weight[selected] * eb[selected]

            # Write updated spring weight for the selected edge only
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
