"""
algorithms/novel/cross_sep.py
==============================
Crossing Repair + Annealed Separation, no relaxation (N10): cross_sep.

cf_cross_sep without its expensive first stage: the pipeline starts from the
plain baseline (spectral + spring, ~1s) instead of the currentflow
relaxation (~30-60s, O(V^3) betweenness plus up to 100 spring re-layouts).

  1. baseline               — spectral + spring (disk-cached)
  2. crossing repair        — direct crossing reduction (packs nodes)
  3. annealed separation    — floors ramp up inside the gated polish loop
  4. verified polish        — crossings reclaimed at full floors

Same guarantees as cf_cross_sep: node-node distance >= 0.4k and node to
non-incident edge >= 0.1k (no nodes on top of each other, no edges lying on
nodes or on each other).

The question this algorithm answers: how much of cf_cross_sep's quality is
bought by the costly global untangling, and how much by the repair+floors
machinery alone. Everything except the first stage is inherited unchanged.
"""

import networkx as nx

from algorithms.baseline import Baseline
from algorithms.novel.cf_cross_sep import CFCrossSep
from layout_cache import cached_layout


class CrossSep(CFCrossSep):
    """baseline → repair → annealed separation polish (N10)."""

    @property
    def name(self) -> str:
        return "cross_sep"

    def layout(self, G: nx.Graph) -> dict:
        pos = cached_layout(f"baseline_s{self.seed}", G,
                            lambda: Baseline(seed=self.seed).layout(G))
        pos = self._repair.repair(G, pos)          # verified vs baseline
        pos = self._polish.anneal_expand(G, pos)   # books the continuity cost
        pos = self._polish.repair(G, pos)          # verified vs post-expand
        return self._ensure_floors(G, pos)         # guarantee of last resort
