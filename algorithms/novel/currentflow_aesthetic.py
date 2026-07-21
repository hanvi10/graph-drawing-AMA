"""
algorithms/novel/currentflow_aesthetic.py
==========================================
Current-Flow Relaxation + Aesthetic Repair (N8).

The readability-first pipeline, in four stages:
  1. currentflow relaxation  — global cluster untangling
  2. crossing repair         — maximum crossing reduction (packs nodes)
  3. expansion               — relieve the packing
  4. aesthetic polish        — separation-INVARIANT repair rounds: reclaim
     crossings with candidates snapped to the separation-feasible region,
     steer residual crossings toward perpendicular, keep node-edge
     clearance from degrading, straighten under the separation floor.

Stages 2-3 get the metric gains (they cannot be reached under a hard
separation constraint); stage 4 makes separation an invariant so the final
drawing keeps distinguishable nodes — and the invariant guarantees the
polish never re-creates the clumps. True path continuity is verified with
escalation and fallback inside both stage 2 and stage 4.
"""

import networkx as nx

from algorithms.base import GraphDrawingAlgorithm
from algorithms.edge_relaxation.currentflow import EdgeRelaxationCurrentFlow
from algorithms.novel.crossing_repair import CrossingRepair
from algorithms.novel.aesthetic_repair import AestheticRepair


class CurrentFlowAesthetic(GraphDrawingAlgorithm):
    """currentflow → repair → expand → separation-invariant polish (N8)."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        self._relax = EdgeRelaxationCurrentFlow(seed=seed)
        self._repair = CrossingRepair(seed=seed)
        self._polish = AestheticRepair(seed=seed)

    @property
    def name(self) -> str:
        return "currentflow_aesthetic"

    def layout(self, G: nx.Graph) -> dict:
        pos = self._relax.layout(G)
        pos = self._repair.repair(G, pos)
        pos = self._repair.expand(G, pos)
        return self._polish.repair(G, pos)
