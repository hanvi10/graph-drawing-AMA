"""
algorithms/edge_relaxation/ebc.py
=================================
The paper's algorithm: Graph Drawing using Edge Relaxation.
Ported from the original src/experiments/run_experiment.py and
src/graph_utils.py in graph-drawing-edge-relaxation/, with two upgrades:

    1. ForceAtlas2 instead of the spring layout.
       The paper runs nx.spring_layout (Fruchterman-Reingold) for the initial
       layout and for every relaxation step. We use ForceAtlas2
       (nx.forceatlas2_layout) for both, which is more robust: its
       degree-weighted repulsion gives every hub room for its neighbourhood,
       so structurally equivalent nodes (e.g. leaves on the same hub) spread
       out instead of collapsing onto one point, and its adaptive speed keeps
       the layout from oscillating. The initial layout is the Baseline's
       spectral -> ForceAtlas2 layout.

    2. Minimum distance between nodes.
       The final layout guarantees no two nodes are closer than 0.30 * k,
       where k = sqrt(bounding-box area / n) is the average spacing per node,
       so every node stays distinguishable when drawn. It is imposed only
       once, after relaxation (Baseline.finalize), never during it, so the
       relaxation itself is unchanged.

The edge selection and relaxation logic below is the paper's, unchanged.

Core idea:
    Some edges act as "bridges" between clusters and force the layout to
    distort, creating many crossings. By gradually reducing the weight of
    these bridge edges, the force-directed algorithm can spread the clusters
    apart and reduce crossings.

    Bridge edges are identified using Edge Betweenness Centrality (EBC):
    an edge has high EBC if many shortest paths in the graph pass through it.

Two separate dicts (matching the original exactly):
    scale[e]  -- the edge weight used by nx.forceatlas2_layout.
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
from algorithms.baseline import Baseline
from metrics import count_crossings


class EdgeRelaxation(GraphDrawingAlgorithm):
    """
    Graph Drawing using Edge Relaxation (Higueras, Escofet, Cortadella 2021).
    Port of getLayout() from the original run_experiment.py, upgraded to
    ForceAtlas2 with a minimum node distance on the final layout.
    """

    def __init__(
        self,
        k_r: float = 0.1,    # relaxation constant: how much to reduce an edge's attraction weight
        k_w: float = 0.05,   # selection constant:  how much to reduce re-selection chance
        max_iter: int = 100,  # hard cap on iterations
        patience: int = 20,   # stop after more than this many steps without improvement
        seed: int = 42,
        initial_layout_iterations: int = 300,
        loop_fa2_iters: int = 50,  # FA2 iters per relaxation step; warm start → fewer needed
    ):
        self.k_r = k_r
        self.k_w = k_w
        self.max_iter = max_iter
        self.patience = patience
        self.seed = seed
        self.initial_layout_iterations = initial_layout_iterations
        self.loop_fa2_iters = loop_fa2_iters

    @property
    def name(self) -> str:
        return f"edge_relaxation(kr={self.k_r},kw={self.k_w})"

    def layout(self, G: nx.Graph) -> dict:
        """
        Run edge relaxation and return the best layout found.
        The selection/relaxation loop mirrors getLayout() from the original
        run_experiment.py.
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

            # Write updated attraction weight for the selected edge only
            G[selected[0]][selected[1]]['relax'] = scale[selected]

            # ForceAtlas2 warm-started from the last layout; 'relax' scales
            # each edge's attraction (edges without it count as 1)
            pos = nx.forceatlas2_layout(G, pos=last_pos.copy(), weight='relax',
                                        max_iter=self.loop_fa2_iters)
            crossings = count_crossings(G, pos)

            if crossings < best_crossings:
                best_crossings    = crossings
                best_crossings_it = it
                best_pos          = pos

            if best_crossings_it + self.patience < it:
                break

            last_pos = pos

        # Min separation + rescale happen once, here, after relaxation
        final = best_pos if best_pos is not None else pos
        return Baseline(seed=self.seed).finalize(final)
