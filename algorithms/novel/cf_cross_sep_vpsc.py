"""
algorithms/novel/cf_cross_sep_vpsc.py
======================================
COMPARISON BASELINE (N9e): cf_cross_sep with its separation stage replaced by
VPSC / Fast Node Overlap Removal (Dwyer, Marriott & Stuckey, 2005).

Like the PRISM baseline, kept to show a proven literature method underperforms
here. VPSC is minimal-displacement and provably removes overlaps, but its
separation is AXIS-ALIGNED (remove x-overlaps, then y-overlaps), so fanning a
coincident hub apart lands its neighbours on a grid whose spokes cross each
other — even worse than PRISM on hub graphs (genetic_mouse 2 -> ~278). The
right theory (minimal displacement) with the wrong geometry for radial graph
structure. Evidence that cf_cross_sep's isotropic projection + crossing-reclaim
polish is what actually preserves crossings while separating.

cf -> crossing repair -> VPSC overlap removal (iterated to the floor) ->
light node-edge cleanup.
"""

import math

import networkx as nx
import numpy as np

from algorithms.baseline import Baseline
from algorithms.edge_relaxation.currentflow import EdgeRelaxationCurrentFlow
from algorithms.novel.cf_cross_sep import (CFCrossSep, _project_floors,
                                           _floor_violations)
from algorithms.novel.vpsc import remove_overlaps
from layout_cache import cached_layout


class CFCrossSepVpsc(CFCrossSep):
    """currentflow → repair → VPSC/FNOR overlap removal (comparison baseline)."""

    @property
    def name(self) -> str:
        return "cf_cross_sep_vpsc"

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

        # VPSC node-node separation, iterated (k grows as the layout spreads)
        for _ in range(12):
            w_, h_ = coords.max(0) - coords.min(0)
            k = math.sqrt(max(w_ * h_, 1e-12) / n)
            D = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)
            np.fill_diagonal(D, np.inf)
            if D.min() >= self.sep_frac * k - 1e-9:
                break
            coords = remove_overlaps(coords, self.sep_frac * k)

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
