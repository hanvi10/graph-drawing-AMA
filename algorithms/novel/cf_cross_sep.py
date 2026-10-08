"""
algorithms/novel/cf_cross_sep.py
=================================
Current-Flow + Crossing Repair + Annealed Separation (N9): cf_cross_sep.

Pipeline:
  1. currentflow relaxation — global cluster untangling. ForceAtlas2 is used
     for every layout step, and its output already carries Baseline's 0.30k
     node-node floor (Baseline.finalize). Disk-cached (layout_cache): it is
     the expensive stage and deterministic within one environment. NOTE:
     when the cache is warm, runtime_s in the results excludes this stage.
  2. crossing repair        — greedy crossing reduction (packs nodes);
     continuity verified against the currentflow layout.
  3. annealed separation polish — each round r projects the layout onto a
     RAMPED node-node floor (k = sqrt(bounding-box area / n), ramp -> 1):
         node–node distance >= ramp * SEP      (SEP = 0.30k)
     then runs the gated relocate/smooth passes at that same ramped floor.
     By the final rounds the full floor holds and the gates keep it.
  4. Baseline.finalize() — GUARANTEES the node-node floor and rescales to
     [-1, 1], exactly as for every other algorithm.

SEP is the same 0.30k floor as Baseline's minimum separation, so every
algorithm is compared under the same hard readability constraint.

Node-to-edge clearance — "no worse", not a floor. A node lying on a foreign
edge is NOT a crossing (the crossing test is strict), so a crossing-greedy
repair can trade crossings for nodes-on-edges and look better on the metric
while reading worse. Unconstrained, this pipeline put 2.4x as many nodes on
edges as the baseline (45 rendered graphs). So every move made to reduce
crossings or smooth paths (repair relocate/smooth, polish relocate/smooth)
must not increase the node's clearance penalty — the summed depth by which
it and its edges come within CLEAR = 0.10k (about the drawn node radius) of
foreign edges and nodes. Overlaps already in the input may stay; new ones
cannot be created by these moves. (The main-branch version enforced a hard
0.075k node-edge FLOOR instead; it was dropped because no other algorithm has
one and the projection did not converge on dense graphs. The floor
projection itself, _project_sep, and Baseline.finalize are not gated: they
only move nodes apart by the small amounts the node-node floor requires.)

Why annealed (measured on main's spring-based pipeline): projecting straight
to the full floor and polishing after loses badly — inflating a packed
cluster in one step sweeps its edges across each other, and the gated polish
can only move one node at a time UNDER the new floor, so it cannot
re-coordinate the cluster. Raising the floor gradually lets the relocate pass
reroute edges while the pressure builds instead of after the damage.

The polish gates are AestheticRepair's G1 (crossings), G2 (continuity),
G3 (separation) and G4 (clearance), with G4 replaced by the depth-based
"no worse" clearance rule above. Continuity keeps the parent's verified
escalation and fallback; whatever layout comes out, step 4 restores the
floor.

Ported from the main branch. Only step 1 ever ran a force-directed layout,
so building on the FA2 currentflow makes the whole pipeline FA2-based; the
repair and polish are geometric.
"""

import math

import networkx as nx
import numpy as np

from algorithms.base import GraphDrawingAlgorithm
from algorithms.baseline import Baseline
from algorithms.edge_relaxation.currentflow import EdgeRelaxationCurrentFlow
from algorithms.novel.crossing_repair import CrossingRepair
from algorithms.novel.aesthetic_repair import AestheticRepair, _clearance_penalty
from layout_cache import cached_layout


def _sep_violations(coords, sep):
    """Number of node pairs closer than sep."""
    D = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)
    return int(np.triu(D < sep, k=1).sum())


def _project_sep(coords, sep, rng, rounds=6):
    """Jacobi projection of the layout onto {node-node >= sep}: every pair
    below the floor is pushed apart along its joining line, each node by half
    the deficit; total moves are capped at sep per round. Returns coords."""
    for _ in range(rounds):
        disp = np.zeros_like(coords)
        diff = coords[:, None, :] - coords[None, :, :]
        D = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(D, np.inf)
        pairs = np.argwhere(np.triu(D < sep, k=1))
        if len(pairs) == 0:
            break
        for i, j in pairs:
            d = D[i, j]
            if d > 1e-12:
                v = diff[i, j] / d
            else:                   # coincident: no direction, pick a random one
                v = rng.normal(size=2)
                v /= np.linalg.norm(v)
            push = 0.5 * (sep - d)
            disp[i] += v * push
            disp[j] -= v * push
        L = np.linalg.norm(disp, axis=1)
        coords = coords + disp * np.minimum(1.0, sep / np.maximum(L, 1e-12))[:, None]
    return coords


def _clearance_k(coords):
    """k = sqrt(bounding-box area / n), the average spacing per node."""
    w, h = coords.max(axis=0) - coords.min(axis=0)
    return math.sqrt(max(w * h, 1e-12) / len(coords))


class _ClearanceGatedRepair(CrossingRepair):
    """CrossingRepair whose moves may not increase the node's clearance
    penalty (node-edge intrusion within clearance_frac * k)."""

    def __init__(self, seed, clearance_frac):
        super().__init__(seed=seed)
        self.clearance_frac = clearance_frac

    def _extra_gate(self, coords, i, cur, cand, u_other, other_E) -> bool:
        thresh = self.clearance_frac * _clearance_k(coords)
        return (_clearance_penalty(coords, i, cand, u_other, other_E, thresh)
                <= _clearance_penalty(coords, i, cur, u_other, other_E, thresh))


class _AnnealedSepPolish(AestheticRepair):
    """AestheticRepair with (a) its expand pass replaced by a node-node floor
    projection each round and (b) G4 replaced by the depth-based "no worse"
    clearance rule (see module docstring).

    Two modes share `_repair_once`:
      * anneal_expand(G, pos) — the floor ramps up over the rounds while the
        gated passes reroute edges under the growing pressure. UNVERIFIED on
        purpose: spreading packed clusters inherently costs more than the 5%
        continuity budget, so this stage books that cost. This is where most
        of cf_cross_sep's path-continuity loss comes from.
      * repair(G, pos)       — parent's verified entry, ramp pinned at 1.0:
        reclaims crossings at the full floor, continuity checked against the
        post-expansion layout with escalation and fallback.
    """

    def __init__(self, seed, sep_frac, clearance_frac, max_rounds, n_jitter):
        super().__init__(seed=seed, sep_frac=sep_frac, max_rounds=max_rounds,
                         n_jitter=n_jitter)
        self.clearance_frac = clearance_frac
        self._annealing = False

    def anneal_expand(self, G: nx.Graph, pos: dict) -> dict:
        self._annealing = True
        try:
            return self._repair_once(G, pos, self.angle_slack_deg)
        finally:
            self._annealing = False

    # G4: the parent compares this at the candidate vs the current position
    # and rejects the move if it grew — the "no worse" clearance rule
    def _clearance_viol(self, coords, i, cand, u_other, other_E, thresh):
        return _clearance_penalty(coords, i, cand, u_other, other_E, thresh)

    def _repair_once(self, G: nx.Graph, pos: dict, slack: float) -> dict:
        nodes, coords, E, share_mask, node_data = self._prepare(G, pos)
        if len(E) == 0:
            return pos
        rng = np.random.default_rng(self.seed)

        anneal = max(1, self.max_rounds - 2)   # full floor 2 rounds early
        for t in range(self.max_rounds):
            ramp = min(1.0, (t + 1) / anneal) if self._annealing else 1.0
            w, h = coords.max(axis=0) - coords.min(axis=0)
            span = float(max(w, h)) or 1.0
            k = math.sqrt(max(w * h, 1e-12) / len(nodes))
            sep = ramp * self.sep_frac * k
            clear = self.clearance_frac * k      # G4 threshold (not ramped)

            coords = _project_sep(coords, sep, rng)
            moved = self._relocate_pass(coords, E, share_mask, node_data,
                                        sep, clear, slack, span, rng)
            for _ in range(2):
                self._smooth_sep_pass(coords, node_data, sep, clear, slack, span)

            if ramp >= 1.0 and moved == 0 and _sep_violations(coords, sep) == 0:
                break

        return {nodes[i]: tuple(coords[i]) for i in range(len(nodes))}


class CFCrossSep(GraphDrawingAlgorithm):
    """currentflow (FA2) → crossing repair → annealed separation polish (N9)."""

    def __init__(
        self,
        seed: int = 42,
        sep_frac: float = 0.30,      # node-node floor, fraction of k — the same
                                     # as Baseline's. (On main, fanning packed
                                     # hubs to 0.4k sprayed crossings; 0.30 kept
                                     # every graph unclumped.)
        clearance_frac: float = 0.10,  # node-edge clearance that crossing moves
                                       # may not make worse (~ drawn node radius)
        polish_rounds: int = 8,
        n_jitter: int = 10,
    ):
        self.seed = seed
        self.sep_frac = sep_frac
        self._repair = _ClearanceGatedRepair(seed=seed, clearance_frac=clearance_frac)
        self._polish = _AnnealedSepPolish(seed=seed, sep_frac=sep_frac,
                                          clearance_frac=clearance_frac,
                                          max_rounds=polish_rounds,
                                          n_jitter=n_jitter)

    @property
    def name(self) -> str:
        return "cf_cross_sep"

    def _start_layout(self, G: nx.Graph) -> dict:
        """Stage 1: the layout the repair starts from — the currentflow
        relaxation (disk-cached). CrossSep overrides only this."""
        return cached_layout(
            f"currentflow_s{self.seed}", G,
            lambda: EdgeRelaxationCurrentFlow(seed=self.seed).layout(G))

    def layout(self, G: nx.Graph) -> dict:
        pos = self._start_layout(G)
        pos = self._repair.repair(G, pos)          # verified vs start layout
        pos = self._polish.anneal_expand(G, pos)   # books the continuity cost
        pos = self._polish.repair(G, pos)          # verified vs post-expand
        # guaranteed node-node floor + [-1, 1] rescale, as for every algorithm
        return Baseline(seed=self.seed, sep_frac=self.sep_frac).finalize(
            {v: np.asarray(p, dtype=float) for v, p in pos.items()})
