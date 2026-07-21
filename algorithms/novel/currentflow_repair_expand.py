"""
algorithms/novel/currentflow_repair_expand.py
==============================================
Current-Flow Relaxation + Crossing Repair + Expansion (N6).

Extends currentflow_repair with a final readability pass: nodes that ended up
clumped together (the repair/smoothing passes reward packing, since none of
the paper's metrics penalize it) are pushed apart until they are at least a
fraction of the mean edge length from each other — spring layout's repulsion
force, reintroduced as a guarded post-pass.

The expansion deliberately sacrifices some mean edge length (spreading
lengthens edges) in exchange for a drawing with distinguishable nodes.
Crossings and path continuity are protected move-by-move by the same guards
used everywhere else in the repair machinery.
"""

import networkx as nx

from algorithms.base import GraphDrawingAlgorithm
from algorithms.edge_relaxation.currentflow import EdgeRelaxationCurrentFlow
from algorithms.novel.crossing_repair import CrossingRepair


class CurrentFlowRepairExpand(GraphDrawingAlgorithm):
    """currentflow relaxation → crossing repair → anti-clumping expansion (N6)."""

    def __init__(self, seed: int = 42, min_sep_frac: float = 0.6,
                 expand_rounds: int = 5, cross_slack: int = 0):
        self.seed = seed
        self.min_sep_frac = min_sep_frac
        self.expand_rounds = expand_rounds
        self.cross_slack = cross_slack
        self._relax = EdgeRelaxationCurrentFlow(seed=seed)
        self._repair = CrossingRepair(seed=seed)

    @property
    def name(self) -> str:
        return "currentflow_repair_expand"

    def layout(self, G: nx.Graph) -> dict:
        pos = self._relax.layout(G)
        pos = self._repair.repair(G, pos)
        return self._repair.expand(G, pos,
                                   min_sep_frac=self.min_sep_frac,
                                   rounds=self.expand_rounds,
                                   cross_slack=self.cross_slack)
