"""
algorithms/novel/cross_sep.py
==============================
Crossing Repair + Annealed Separation, no relaxation (N10): cross_sep.

The ablation of cf_cross_sep: identical pipeline, but stage 1 is the plain
Baseline (spectral + ForceAtlas2 + min separation) instead of the currentflow
relaxation:

  1. baseline               — spectral + ForceAtlas2 (disk-cached)
  2. crossing repair        — greedy crossing reduction
  3. annealed separation    — the node-node floor ramps up inside the polish
  4. Baseline.finalize()    — guaranteed 0.30k floor + [-1, 1] rescale

The question it answers: how much of cf_cross_sep's result is due to the
edge relaxation, and how much to the repair + separation machinery alone.
Everything except stage 1 is inherited unchanged from CFCrossSep (same
clearance rule, gates, polish and parameters), so any difference between
the two comes from the starting layout.

Ported from the main branch (where it started from spectral + spring).
"""

import networkx as nx

from algorithms.baseline import Baseline
from algorithms.novel.cf_cross_sep import CFCrossSep
from layout_cache import cached_layout


class CrossSep(CFCrossSep):
    """baseline → crossing repair → annealed separation polish (N10)."""

    @property
    def name(self) -> str:
        return "cross_sep"

    def _start_layout(self, G: nx.Graph) -> dict:
        return cached_layout(f"baseline_s{self.seed}", G,
                             lambda: Baseline(seed=self.seed).layout(G))
