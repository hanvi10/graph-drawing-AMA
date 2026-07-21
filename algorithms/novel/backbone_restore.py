"""
algorithms/novel/backbone_restore.py
=====================================
Backbone-Restore Layout (N2).

The paper's Fig. 1b showed that a *human* who knows which edges distort the
drawing can simply leave them out during layout and add them back afterwards —
cutting crossings roughly in half. Edge relaxation only approximates this by
gradually down-weighting edges, which the paper admits can over-relax.

This algorithm automates the human procedure directly:

1. DELETE   — Remove the top-scoring bridge edges (by EBC) outright, skipping
              any removal that would disconnect the graph.
2. LAYOUT   — Lay out the clean skeleton (spectral + spring). No residual
              bridge forces at all, unlike relaxation where the weight never
              reaches zero.
3. RESTORE  — Put the deleted edges back and either (a) keep the skeleton
              positions as-is, or (b) run a short spring refinement with the
              restored edges at a small weight epsilon.

Several deletion budgets and restore strengths are tried; the layout with the
fewest crossings (measured on the FULL graph) wins.

Cost: one EBC + a few short layouts + ~6 crossing counts. Far cheaper than
the relaxation loop.
"""

import networkx as nx
import numpy as np

from algorithms.base import GraphDrawingAlgorithm
from metrics import count_crossings


class BackboneRestore(GraphDrawingAlgorithm):
    """Delete top-EBC bridges, lay out the skeleton, restore and refine (N2)."""

    def __init__(
        self,
        seed: int = 42,
        budgets: tuple = (0.03, 0.08),      # fraction of edges to delete
        restore_weights: tuple = (0.01, 0.1),
        skeleton_iterations: int = 50,
        restore_iterations: int = 30,
    ):
        self.seed = seed
        self.budgets = budgets
        self.restore_weights = restore_weights
        self.skeleton_iterations = skeleton_iterations
        self.restore_iterations = restore_iterations

    @property
    def name(self) -> str:
        return "backbone_restore"

    def layout(self, G: nx.Graph) -> dict:
        G = G.copy()
        np.random.seed(self.seed)
        m = G.number_of_edges()

        raw_eb = nx.edge_betweenness_centrality(G, normalized=False)
        eb = {}
        for e in G.edges():
            eb[e] = raw_eb.get(e, raw_eb.get((e[1], e[0]), 0.0))
        ranked = sorted(eb, key=eb.get, reverse=True)

        best_pos = None
        best_crossings = np.inf

        for frac in self.budgets:
            budget = max(3, int(round(frac * m)))

            # ── 1. Delete bridges, preserving connectivity ───────────────────
            H = G.copy()
            removed = []
            for e in ranked:
                if len(removed) >= budget:
                    break
                H.remove_edge(*e)
                if nx.is_connected(H):
                    removed.append(e)
                else:
                    H.add_edge(*e)
            if not removed:
                continue

            # ── 2. Skeleton layout (bridges completely absent) ───────────────
            skeleton_pos = self._initial_layout(H, self.seed, self.skeleton_iterations)

            # Candidate (a): skeleton positions used directly on the full graph
            crossings = count_crossings(G, skeleton_pos)
            if crossings < best_crossings:
                best_crossings = crossings
                best_pos = skeleton_pos

            # ── 3. Restore with small weight + short refinement ──────────────
            removed_set = {frozenset(e) for e in removed}
            for eps in self.restore_weights:
                for u, v in G.edges():
                    G[u][v]['restore'] = eps if frozenset((u, v)) in removed_set else 1.0
                pos = nx.spring_layout(
                    G, pos={k: np.array(p) for k, p in skeleton_pos.items()},
                    weight='restore', iterations=self.restore_iterations,
                    seed=self.seed)
                crossings = count_crossings(G, pos)
                if crossings < best_crossings:
                    best_crossings = crossings
                    best_pos = pos

        if best_pos is None:   # no edge was removable — fall back to baseline
            best_pos = self._initial_layout(G, self.seed, self.skeleton_iterations)

        return {v: tuple(p) for v, p in best_pos.items()}
