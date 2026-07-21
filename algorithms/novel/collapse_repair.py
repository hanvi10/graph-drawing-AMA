"""
algorithms/novel/collapse_repair.py
====================================
Collapse-Expand + Crossing Repair (N4).

Pipeline of the two novel ideas:
  1. CommunityCollapse produces a structurally clean layout (bridges never
     distort the per-community layouts).
  2. CrossingRepair then directly removes the remaining crossings by
     relocating the worst-offending nodes.

Both stages are cheap, so the pipeline still costs a fraction of one
edge-relaxation run.
"""

import networkx as nx

from algorithms.base import GraphDrawingAlgorithm
from algorithms.novel.community_collapse import CommunityCollapse
from algorithms.novel.crossing_repair import CrossingRepair


class CollapseRepair(GraphDrawingAlgorithm):
    """CommunityCollapse layout followed by a CrossingRepair pass (N4)."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        self._collapse = CommunityCollapse(seed=seed)
        self._repair = CrossingRepair(seed=seed)

    @property
    def name(self) -> str:
        return "collapse_repair"

    def layout(self, G: nx.Graph) -> dict:
        pos = self._collapse.layout(G)
        return self._repair.repair(G, pos)
