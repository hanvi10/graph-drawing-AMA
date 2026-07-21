"""
algorithms/novel/cf_cross_sep_min.py
=====================================
Minimal separation variant (N9b): cf_cross_sep_min.

Same first two stages as cf_cross_sep — current-flow relaxation (cached) then
crossing repair — but the separation stage does ONLY what its name promises:
project the layout onto the two minimum-distance floors and stop.

    node–node distance          >= 0.4 k
    node to non-incident edge    >= 0.1 k      (k = sqrt(area / n))

No annealed ramp, no gated relocate/smooth passes, no crossing reclaim — the
projection is iterated (recomputing k as the layout grows) until the floors
hold or it stops improving. Whatever crossings that costs, it costs.

This exists to answer the question: how much of cf_cross_sep's cost is the
minimal separation itself, versus the polish machinery layered on top.
"""

import math

import networkx as nx
import numpy as np

from algorithms.baseline import Baseline
from algorithms.edge_relaxation.currentflow import EdgeRelaxationCurrentFlow
from algorithms.novel.cf_cross_sep import CFCrossSep, _project_floors, _floor_violations
from layout_cache import cached_layout


class CFCrossSepMin(CFCrossSep):
    """currentflow → crossing repair → minimal floor projection (N9b)."""

    def __init__(self, seed: int = 42, sep_frac: float = 0.4,
                 overlap_frac: float = 0.1, max_iter: int = 25):
        super().__init__(seed=seed, sep_frac=sep_frac, overlap_frac=overlap_frac)
        self.max_iter = max_iter

    @property
    def name(self) -> str:
        return "cf_cross_sep_min"

    def _separate(self, G: nx.Graph, pos: dict) -> dict:
        """Project onto the floors until they hold or stop improving. Keeps the
        best (fewest total violations) layout seen, since on dense graphs the
        projection can plateau or drift once the floors become infeasible."""
        nodes = list(G.nodes())
        idx = {v: i for i, v in enumerate(nodes)}
        coords = np.array([pos[v] for v in nodes], dtype=float)
        E = np.array([(idx[u], idx[v]) for u, v in G.edges()], dtype=int)
        if len(nodes) < 3 or len(E) == 0:
            return pos
        end_mask = np.zeros((len(E), len(nodes)), dtype=bool)
        end_mask[np.arange(len(E)), E[:, 0]] = True
        end_mask[np.arange(len(E)), E[:, 1]] = True
        rng = np.random.default_rng(self.seed)

        def viol(C):
            w, h = C.max(0) - C.min(0)
            k = math.sqrt(max(w * h, 1e-12) / len(nodes))
            return sum(_floor_violations(C, E, end_mask,
                                         self.sep_frac * k, self.overlap_frac * k))

        best, best_v = coords.copy(), viol(coords)
        for _ in range(self.max_iter):
            if best_v == 0:
                break
            w, h = coords.max(0) - coords.min(0)
            k = math.sqrt(max(w * h, 1e-12) / len(nodes))
            coords = _project_floors(coords, E, end_mask,
                                     self.sep_frac * k, self.overlap_frac * k,
                                     rng, rounds=8)
            v = viol(coords)
            if v < best_v:
                best, best_v = coords.copy(), v
        return {nodes[i]: tuple(best[i]) for i in range(len(nodes))}

    def layout(self, G: nx.Graph) -> dict:
        cached_layout(f"baseline_s{self.seed}", G,
                      lambda: Baseline(seed=self.seed).layout(G))
        pos = cached_layout(
            f"currentflow_s{self.seed}", G,
            lambda: EdgeRelaxationCurrentFlow(seed=self.seed).layout(G))
        pos = self._repair.repair(G, pos)      # verified crossing repair
        return self._separate(G, pos)          # minimal floor projection
