"""
algorithms/novel/currentflow_repair.py
=======================================
Current-Flow Edge Relaxation + Crossing Repair (N5).

Combines the two strongest methods found so far:
  1. EdgeRelaxationCurrentFlow (best relaxation variant: strong edge-length
     and continuity gains) produces the base layout.
  2. CrossingRepair then directly removes remaining crossings by relocating
     the worst-offending nodes, with its angle guard and verified
     path-continuity escalation protecting the relaxation's quality gains.

Runtime is dominated by the relaxation stage (~1 min/graph); the repair pass
adds only a few seconds.
"""

import networkx as nx

from algorithms.base import GraphDrawingAlgorithm
from algorithms.edge_relaxation.currentflow import EdgeRelaxationCurrentFlow
from algorithms.novel.crossing_repair import CrossingRepair


class CurrentFlowRepair(GraphDrawingAlgorithm):
    """EdgeRelaxationCurrentFlow layout followed by a CrossingRepair pass (N5)."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        self._relax = EdgeRelaxationCurrentFlow(seed=seed)
        self._repair = CrossingRepair(seed=seed)

    @property
    def name(self) -> str:
        return "currentflow_repair"

    def layout(self, G: nx.Graph) -> dict:
        pos = self._relax.layout(G)
        return self._repair.repair(G, pos)
