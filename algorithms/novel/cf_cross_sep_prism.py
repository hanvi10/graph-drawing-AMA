"""
algorithms/novel/cf_cross_sep_prism.py
=======================================
COMPARISON BASELINE (N9d): cf_cross_sep with its separation stage replaced by
PRISM (Gansner & Hu, "Efficient Node Overlap Removal", 2009).

Kept NOT because it is good — it is the opposite. It exists to show, in the
benchmark table, that a standard stress-based overlap-removal method degrades a
crossing-optimised layout badly: PRISM's objective matches proximity distances,
which is not crossing-aware, so on near-planar / hub graphs it shreds the
low-crossing structure crossing repair built (e.g. genetic_mouse 2 -> 48
crossings; aggregate ~-6% vs baseline, worse than EBC). That is the evidence
that cf_cross_sep's minimal, isotropic, crossing-reclaiming separation is doing
real work the literature method does not.

cf -> crossing repair -> PRISM node-overlap removal -> light node-edge cleanup.
"""

import math

import networkx as nx
import numpy as np
from scipy.spatial import Delaunay, QhullError

from algorithms.baseline import Baseline
from algorithms.edge_relaxation.currentflow import EdgeRelaxationCurrentFlow
from algorithms.novel.cf_cross_sep import (CFCrossSep, _project_floors,
                                           _floor_violations)
from layout_cache import cached_layout


def _prism(coords, n, sep_frac, max_outer=25, smacof_iters=12):
    """PRISM overlap removal: grow only the overlapping proximity (Delaunay)
    edges to `sep` and re-settle by SMACOF stress majorisation, re-triangulate,
    repeat. sep = sep_frac * k, recomputed each round."""
    for _ in range(max_outer):
        w_, h_ = coords.max(0) - coords.min(0)
        k = math.sqrt(max(w_ * h_, 1e-12) / n)
        sep = sep_frac * k
        try:
            tri = Delaunay(coords)
        except (QhullError, ValueError):
            break
        edges = set()
        for s in tri.simplices:
            for a, b in ((0, 1), (1, 2), (2, 0)):
                i, j = int(s[a]), int(s[b])
                edges.add((min(i, j), max(i, j)))
        E = np.array(sorted(edges))
        d = np.linalg.norm(coords[E[:, 0]] - coords[E[:, 1]], axis=1)
        if (d >= sep - 1e-12).all():
            break
        L = np.where(d < sep, sep, np.maximum(d, 1e-9))
        w = 1.0 / (L * L)
        for _ in range(smacof_iters):
            diff = coords[E[:, 0]] - coords[E[:, 1]]
            dist = np.maximum(np.linalg.norm(diff, axis=1), 1e-9)
            unit = diff / dist[:, None]
            num = np.zeros((n, 2))
            den = np.zeros(n)
            np.add.at(num, E[:, 0], w[:, None] * (coords[E[:, 1]] + L[:, None] * unit))
            np.add.at(num, E[:, 1], w[:, None] * (coords[E[:, 0]] - L[:, None] * unit))
            np.add.at(den, E[:, 0], w)
            np.add.at(den, E[:, 1], w)
            m = den > 0
            coords[m] = num[m] / den[m, None]
    return coords


class CFCrossSepPrism(CFCrossSep):
    """currentflow → repair → PRISM overlap removal (comparison baseline)."""

    @property
    def name(self) -> str:
        return "cf_cross_sep_prism"

    def _separate(self, G: nx.Graph, pos: dict) -> dict:
        nodes = list(G.nodes())
        idx = {v: i for i, v in enumerate(nodes)}
        coords = np.array([pos[v] for v in nodes], dtype=float)
        n = len(nodes)
        E = np.array([(idx[u], idx[v]) for u, v in G.edges()], dtype=int)
        if n < 4 or len(E) == 0:
            return pos
        end_mask = np.zeros((len(E), n), dtype=bool)
        end_mask[np.arange(len(E)), E[:, 0]] = True
        end_mask[np.arange(len(E)), E[:, 1]] = True
        rng = np.random.default_rng(self.seed)

        coords = _prism(coords, n, self.sep_frac)
        for _ in range(6):                      # residual + node-edge cleanup
            w_, h_ = coords.max(0) - coords.min(0)
            k = math.sqrt(max(w_ * h_, 1e-12) / n)
            sep, clear = self.sep_frac * k, self.overlap_frac * k
            if _floor_violations(coords, E, end_mask, sep, clear) == (0, 0):
                break
            coords = _project_floors(coords, E, end_mask, sep, clear, rng, rounds=6)
        return {nodes[i]: tuple(coords[i]) for i in range(n)}

    def layout(self, G: nx.Graph) -> dict:
        cached_layout(f"baseline_s{self.seed}", G,
                      lambda: Baseline(seed=self.seed).layout(G))
        pos = cached_layout(
            f"currentflow_s{self.seed}", G,
            lambda: EdgeRelaxationCurrentFlow(seed=self.seed).layout(G))
        pos = self._repair.repair(G, pos)
        return self._separate(G, pos)
