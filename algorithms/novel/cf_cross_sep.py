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
     RAMPED pair of floors (k = sqrt(bounding-box area / n), ramp -> 1):
         node–node distance          >= ramp * SEP      (SEP     = 0.30k)
         node to non-incident edge   >= ramp * OVERLAP  (OVERLAP = 0.075k)
     then runs the gated relocate/smooth passes at those same ramped floors.
     By the final rounds the full floors hold and the gates keep them.
  4. floor guarantee — remaining violations of both floors are projected out
     (a few rounds), then Baseline.finalize() GUARANTEES the node-node floor
     and rescales to [-1, 1], exactly as for every other algorithm.

SEP is the same 0.30k floor as Baseline's minimum separation, so every
algorithm is compared at the same node spacing. OVERLAP adds a node-to-edge
floor that only this pipeline has, and it is APPROXIMATE (measured on the 45
rendered graphs): the typical graph ends with its closest node-edge pair at
~0.073k (nodes are pushed to exactly the floor, then k grows slightly as the
layout expands), and on ~10% of graphs a node stays within ~0.01-0.03k of an
edge — on dense graphs (zebras, kangaroo, the social graphs) the projection
cycles instead of converging, because pushing a node off one edge pushes it
into another. This behaviour is inherited unchanged from main. The
node-node floor always holds.

The OVERLAP floor is what forbids edges lying on top of each other: an edge
through a foreign node, or two incident edges so collinear that one lies on
the other (the far endpoint would sit on the longer edge), both violate it.

Why annealed (measured on main's spring-based pipeline): projecting straight
to the full floors and polishing after loses badly — inflating a packed
cluster in one step sweeps its edges across each other, and the gated polish
can only move one node at a time UNDER the new floors, so it cannot
re-coordinate the cluster. Raising the floor gradually lets the relocate pass
reroute edges while the pressure builds instead of after the damage.

The polish gates are AestheticRepair's, with one change: the clearance gate
is depth-monotone (the worst overlap may only get shallower) instead of
count-based — a count-equal trade could otherwise slide a node from almost-
clear to exactly ON an edge. Same only-gets-looser pattern G3 uses for
separation. Continuity keeps the parent's verified escalation; if even the
zero-slack retry fails, the fallback layout gets one hard projection so the
separation guarantee survives (paid for with continuity, reported honestly
by the metrics).

Ported from the main branch. Only step 1 ever ran a force-directed layout,
so building on the FA2 currentflow makes the whole pipeline FA2-based; the
repair and polish are geometric and unchanged.
"""

import math

import networkx as nx
import numpy as np

from algorithms.base import GraphDrawingAlgorithm
from algorithms.baseline import Baseline
from algorithms.edge_relaxation.currentflow import EdgeRelaxationCurrentFlow
from algorithms.novel.crossing_repair import CrossingRepair
from algorithms.novel.aesthetic_repair import (AestheticRepair, _pt_to_segs,
                                               _segs_to_pts)
from layout_cache import cached_layout


def _floor_violations(coords, E, end_mask, sep, clear):
    """(#node-node below sep, #node-edge below clear)."""
    D = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)
    np.fill_diagonal(D, np.inf)
    nn = int((np.triu(D < sep, k=1)).sum())
    Dne = _segs_to_pts(coords[E[:, 0]], coords[E[:, 1]], coords)
    Dne[end_mask] = np.inf
    ne = int((Dne < clear).sum())
    return nn, ne


def _project_floors(coords, E, end_mask, sep, clear, rng, rounds=6):
    """Jacobi projection of the layout onto {node-node >= sep} ∩
    {node-to-nonincident-edge >= clear}. Node-node deficits split evenly;
    node-edge deficits mostly on the node, the endpoints yielding a little.
    Mutates and returns coords."""

    def apply_capped(disp, cap):
        L = np.linalg.norm(disp, axis=1)
        f = np.minimum(1.0, cap / np.maximum(L, 1e-12))
        return coords + disp * f[:, None]

    for _ in range(rounds):
        # node-node floor
        disp = np.zeros_like(coords)
        n_nn = 0
        diff = coords[:, None, :] - coords[None, :, :]
        D = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(D, np.inf)
        for i, j in np.argwhere(np.triu(D < sep, k=1)):
            d = D[i, j]
            if d > 1e-12:
                v = diff[i, j] / d
            else:
                v = rng.normal(size=2)
                v /= np.linalg.norm(v)
            push = 0.5 * (sep - d)
            disp[i] += v * push
            disp[j] -= v * push
            n_nn += 1
        coords = apply_capped(disp, sep)

        # overlap floor, on the updated positions
        n_ne = 0
        for _ in range(3):
            disp = np.zeros_like(coords)
            sub = 0
            Dne = _segs_to_pts(coords[E[:, 0]], coords[E[:, 1]], coords)
            Dne[end_mask] = np.inf
            for e_i, w in np.argwhere(Dne < clear):
                a, b = coords[E[e_i, 0]], coords[E[e_i, 1]]
                ab = b - a
                denom = float(ab @ ab)
                t = 0.0 if denom < 1e-18 else float(np.clip(
                    (coords[w] - a) @ ab / denom, 0.0, 1.0))
                vec = coords[w] - (a + t * ab)
                d = float(np.linalg.norm(vec))
                if d > 1e-12:
                    v = vec / d
                else:                   # node exactly on the edge
                    v = np.array([-ab[1], ab[0]])
                    nv = np.linalg.norm(v)
                    v = v / nv if nv > 1e-12 else rng.normal(size=2)
                    v *= rng.choice((-1.0, 1.0))
                deficit = clear - d
                disp[w] += v * (0.7 * deficit)
                disp[E[e_i, 0]] -= v * (0.25 * deficit)
                disp[E[e_i, 1]] -= v * (0.25 * deficit)
                sub += 1
            if sub == 0:
                break
            n_ne += sub
            coords = apply_capped(disp, sep)

        if n_nn == 0 and n_ne == 0:
            break
    return coords


class _AnnealedSepPolish(AestheticRepair):
    """AestheticRepair with (a) a depth-monotone clearance gate and (b) the
    expand pass replaced by a floor projection each round.

    Two modes share `_repair_once`:
      * anneal_expand(G, pos) — floors ramp up over the rounds while the
        gated passes reroute edges under the growing pressure. UNVERIFIED on
        purpose: spreading packed clusters inherently costs more than the 5%
        continuity budget, so this stage books that cost. This is where most
        of cf_cross_sep's path-continuity loss comes from.
      * repair(G, pos)       — parent's verified entry, ramp pinned at 1.0:
        reclaims crossings at full floors, continuity checked against the
        post-expansion layout with escalation and fallback.
    """

    def __init__(self, seed, sep_frac, overlap_frac, max_rounds, n_jitter):
        super().__init__(seed=seed, sep_frac=sep_frac, max_rounds=max_rounds,
                         n_jitter=n_jitter)
        self.overlap_frac = overlap_frac
        self._annealing = False

    def anneal_expand(self, G: nx.Graph, pos: dict) -> dict:
        self._annealing = True
        try:
            return self._repair_once(G, pos, self.angle_slack_deg)
        finally:
            self._annealing = False

    # G4, depth-monotone: the worst violation of the on-top floor may only
    # get shallower (dominant continuous term; soft count breaks ties)
    def _clearance_viol(self, coords, i, cand, u_other, other_E, thresh):
        soft = super()._clearance_viol(coords, i, cand, u_other, other_E, thresh)
        floor = 0.5 * thresh
        dmin = floor                        # capped: distances above floor tie
        if len(other_E):
            d = _pt_to_segs(cand, coords[other_E[:, 0]], coords[other_E[:, 1]])
            dmin = min(dmin, float(d.min()))
        if len(u_other):
            A = np.repeat(cand[None, :], len(u_other), axis=0)
            D = _segs_to_pts(A, coords[u_other], coords)
            D[:, i] = np.inf
            for j, u in enumerate(u_other):
                D[j, u] = np.inf
            dmin = min(dmin, float(D.min()))
        depth = (floor - dmin) / max(floor, 1e-12)
        return soft + 1e6 * depth

    def _repair_once(self, G: nx.Graph, pos: dict, slack: float) -> dict:
        nodes, coords, E, share_mask, node_data = self._prepare(G, pos)
        if len(E) == 0:
            return pos
        rng = np.random.default_rng(self.seed)

        end_mask = np.zeros((len(E), len(nodes)), dtype=bool)
        end_mask[np.arange(len(E)), E[:, 0]] = True
        end_mask[np.arange(len(E)), E[:, 1]] = True

        anneal = max(1, self.max_rounds - 2)   # full floors 2 rounds early
        for t in range(self.max_rounds):
            ramp = min(1.0, (t + 1) / anneal) if self._annealing else 1.0
            w, h = coords.max(axis=0) - coords.min(axis=0)
            span = float(max(w, h)) or 1.0
            k = math.sqrt(max(w * h, 1e-12) / len(nodes))
            sep = ramp * self.sep_frac * k
            clear = ramp * self.overlap_frac * k
            thresh = 2.0 * clear    # gate threshold; on-top floor = clear

            coords = _project_floors(coords, E, end_mask, sep, clear, rng)
            moved = self._relocate_pass(coords, E, share_mask, node_data,
                                        sep, thresh, slack, span, rng)
            for _ in range(2):
                self._smooth_sep_pass(coords, node_data, sep, thresh, slack, span)

            if ramp >= 1.0 and moved == 0:
                nn_v, ne_v = _floor_violations(coords, E, end_mask, sep, clear)
                if nn_v == 0 and ne_v == 0:
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
        overlap_frac: float = 0.075, # node-to-nonincident-edge floor (0.25*sep)
        polish_rounds: int = 8,
        n_jitter: int = 10,
    ):
        self.seed = seed
        self.sep_frac = sep_frac
        self.overlap_frac = overlap_frac
        self._repair = CrossingRepair(seed=seed)
        self._polish = _AnnealedSepPolish(seed=seed, sep_frac=sep_frac,
                                          overlap_frac=overlap_frac,
                                          max_rounds=polish_rounds,
                                          n_jitter=n_jitter)

    @property
    def name(self) -> str:
        return "cf_cross_sep"

    def _ensure_floors(self, G: nx.Graph, pos: dict) -> dict:
        """Last resort: if the polish fell back (continuity escalation) or
        left residual violations, project them out hard. Best-effort for the
        node-edge floor on dense graphs (see module docstring); layout()
        then guarantees the node-node floor with Baseline.finalize()."""
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

        for _ in range(4):                      # k drifts as the layout grows
            w, h = coords.max(axis=0) - coords.min(axis=0)
            k = math.sqrt(max(w * h, 1e-12) / len(nodes))
            sep, clear = self.sep_frac * k, self.overlap_frac * k
            if _floor_violations(coords, E, end_mask, sep, clear) == (0, 0):
                break
            coords = _project_floors(coords, E, end_mask, sep, clear, rng,
                                     rounds=8)
        return {nodes[i]: tuple(coords[i]) for i in range(len(nodes))}

    def layout(self, G: nx.Graph) -> dict:
        pos = cached_layout(
            f"currentflow_s{self.seed}", G,
            lambda: EdgeRelaxationCurrentFlow(seed=self.seed).layout(G))
        pos = self._repair.repair(G, pos)          # verified vs cf layout
        pos = self._polish.anneal_expand(G, pos)   # books the continuity cost
        pos = self._polish.repair(G, pos)          # verified vs post-expand
        pos = self._ensure_floors(G, pos)          # both floors, best-effort
        # guaranteed node-node floor + [-1, 1] rescale, as for every algorithm
        return Baseline(seed=self.seed, sep_frac=self.sep_frac).finalize(
            {v: np.asarray(p, dtype=float) for v, p in pos.items()})
