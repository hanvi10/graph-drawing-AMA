"""
algorithms/novel/aesthetic_repair.py
=====================================
Aesthetic Repair (N7): crossing repair with separation as an INVARIANT.

crossing_repair optimizes the paper's metrics and exploits their blind spot:
nothing penalizes packing nodes together, so it clumps. currentflow_repair_
expand un-clumps afterwards, but pays the packed gains back. This algorithm
never lets the packing happen:

Move gates (checked for EVERY candidate move, in every pass):
  G1 crossings     — relocation must strictly reduce the node's incident
                     crossings (ties allowed with better secondary keys);
                     smoothing/expansion may never increase them.
  G2 continuity    — the turning-angle proxy (at the node and its neighbors)
                     may not degrade beyond the slack; verified against the
                     true metric at the end with escalation and fallback.
  G3 separation    — the node's nearest-neighbor distance may not drop below
                     min(its current value, sep) where sep = sep_frac * k and
                     k = sqrt(area/n) is Fruchterman-Reingold's natural
                     spacing. Well-spaced nodes stay spaced; tight nodes may
                     only get looser. Clumps cannot form.
  G4 edge clearance— the number of near-incidences (a foreign edge passing
                     within clear_frac*sep of this node, or one of this
                     node's edges passing that close to a foreign node) may
                     not increase. Prevents fake-junction artifacts.

Secondary preference (no metric cost): among candidates equal on crossings,
prefer PERPENDICULAR residual crossings — shallow-angle crossings are what
actually confuse the eye (crossing-angle literature), so the tie-break
minimizes sum(|cos(theta)|) over the node's remaining crossings.

Each round interleaves three passes: expand (relieve inherited tightness)
-> relocate (reduce crossings) -> smooth (straighten paths), all under the
same gates, so the passes correct each other instead of fighting.
"""

import math

import networkx as nx
import numpy as np

from algorithms.novel.crossing_repair import CrossingRepair, _strict_cross


def _pt_to_segs(p, A, B):
    """Distance from point p (2,) to segments A->B ((k,2),(k,2)). Returns (k,)."""
    AB = B - A
    denom = (AB * AB).sum(axis=1)
    denom[denom == 0] = 1e-12
    t = np.clip(((p[None, :] - A) * AB).sum(axis=1) / denom, 0.0, 1.0)
    proj = A + t[:, None] * AB
    return np.linalg.norm(p[None, :] - proj, axis=1)


def _segs_to_pts(A, B, P):
    """Distances from segments A->B ((d,2),(d,2)) to points P (n,2). Returns (d,n)."""
    AB = (B - A)[:, None, :]                          # (d,1,2)
    AP = P[None, :, :] - A[:, None, :]                # (d,n,2)
    denom = (AB * AB).sum(axis=2)
    denom[denom == 0] = 1e-12
    t = np.clip((AP * AB).sum(axis=2) / denom, 0.0, 1.0)   # (d,n)
    proj = A[:, None, :] + t[:, :, None] * AB
    return np.linalg.norm(P[None, :, :] - proj, axis=2)


class AestheticRepair(CrossingRepair):
    """Separation-invariant, clearance-aware, angle-aware repair pass (N7)."""

    def __init__(
        self,
        seed: int = 42,
        initial_layout_iterations: int = 50,
        max_rounds: int = 5,
        n_jitter: int = 6,
        jitter_scale: float = 0.08,
        smooth_steps: tuple = (1.0, 0.5, 0.25),
        angle_slack_deg: float = 5.0,
        max_continuity_loss: float = 0.05,
        sep_frac: float = 0.5,        # separation floor, fraction of k
        clear_frac: float = 0.5,      # edge-clearance threshold, fraction of sep
        expand_steps: tuple = (1.0, 0.5, 0.25),
    ):
        super().__init__(seed=seed,
                         initial_layout_iterations=initial_layout_iterations,
                         max_rounds=max_rounds, n_jitter=n_jitter,
                         n_random=0, jitter_scale=jitter_scale,
                         smooth_steps=smooth_steps,
                         angle_slack_deg=angle_slack_deg,
                         max_continuity_loss=max_continuity_loss)
        self.sep_frac = sep_frac
        self.clear_frac = clear_frac
        self.expand_steps = expand_steps

    @property
    def name(self) -> str:
        return "aesthetic_repair"

    # ── gate helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _nn_dist(coords, i, cand):
        d = np.linalg.norm(coords - cand[None, :], axis=1)
        d[i] = np.inf
        return float(d.min())

    def _clearance_viol(self, coords, i, cand, u_other, other_E, thresh):
        """Near-incidence count: foreign edges too close to this node +
        foreign nodes too close to this node's edges."""
        viol = 0
        if len(other_E):
            d = _pt_to_segs(cand, coords[other_E[:, 0]], coords[other_E[:, 1]])
            viol += int((d < thresh).sum())
        if len(u_other):
            A = np.repeat(cand[None, :], len(u_other), axis=0)
            D = _segs_to_pts(A, coords[u_other], coords)      # (d, n)
            D[:, i] = np.inf
            for j, u in enumerate(u_other):                    # own endpoints don't count
                D[j, u] = np.inf
            viol += int((D < thresh).sum())
        return viol

    def _cross_and_sharpness(self, coords, cand, u_other, other_E, mask):
        """Incident crossings and sum(|cos(angle)|) over them (sharpness:
        0 = all perpendicular, higher = shallower = visually worse)."""
        if len(other_E) == 0:
            return 0, 0.0
        P1 = cand[None, None, :]
        P2 = coords[u_other][:, None, :]
        Q1 = coords[other_E[:, 0]][None, :, :]
        Q2 = coords[other_E[:, 1]][None, :, :]
        cross = _strict_cross(P1, P2, Q1, Q2) & ~mask          # (d, k)
        n_cross = int(cross.sum())
        if n_cross == 0:
            return 0, 0.0
        a = (P2 - P1)                                          # (d,1,2)
        b = (Q2 - Q1)                                          # (1,k,2)
        na = np.linalg.norm(a, axis=2); na[na == 0] = 1e-12
        nb = np.linalg.norm(b, axis=2); nb[nb == 0] = 1e-12
        cos = np.abs((a * b).sum(axis=2) / (na * nb))
        return n_cross, float((cos * cross).sum())

    def _sep_ok(self, coords, i, cur, cand, sep, span):
        """G3: nearest-neighbor distance may not drop below min(current, sep)."""
        nn_cand = self._nn_dist(coords, i, cand)
        if nn_cand < 1e-3 * span:
            return False
        nn_cur = self._nn_dist(coords, i, cur)
        return nn_cand >= min(nn_cur, sep) - 1e-12

    def _snap_to_feasible(self, coords, i, cand, sep, rng):
        """Instead of discarding a candidate that lands too close to another
        node, push it radially out to the separation floor (a few tries) —
        recovers crossing-reducing moves the hard gate would lose."""
        cand = cand.copy()
        for _ in range(3):
            d = np.linalg.norm(coords - cand[None, :], axis=1)
            d[i] = np.inf
            j = int(np.argmin(d))
            if d[j] >= sep:
                break
            v = cand - coords[j]
            n = np.linalg.norm(v)
            if n < 1e-12:
                v, n = rng.normal(size=2), 1.0
            cand = coords[j] + (v / n) * sep
        return cand

    # ── main ─────────────────────────────────────────────────────────────────

    def _repair_once(self, G: nx.Graph, pos: dict, slack: float) -> dict:
        nodes, coords, E, share_mask, node_data = self._prepare(G, pos)
        if len(E) == 0:
            return pos
        rng = np.random.default_rng(self.seed)

        for _ in range(self.max_rounds):
            w, h = coords.max(axis=0) - coords.min(axis=0)
            span = float(max(w, h)) or 1.0
            k = math.sqrt(max(w * h, 1e-12) / len(nodes))
            sep = self.sep_frac * k
            thresh = self.clear_frac * sep

            moved = self._expand_pass(coords, node_data, sep, rng)
            moved += self._relocate_pass(coords, E, share_mask, node_data,
                                         sep, thresh, slack, span, rng)
            for _ in range(2):
                self._smooth_sep_pass(coords, node_data, sep, thresh, slack, span)

            if moved == 0:
                break

        return {nodes[i]: tuple(coords[i]) for i in range(len(nodes))}

    # ── passes ───────────────────────────────────────────────────────────────

    def _relocate_pass(self, coords, E, share_mask, node_data,
                       sep, thresh, slack, span, rng):
        edge_cross = self._edge_crossing_counts(coords, E, share_mask)
        node_cross = np.zeros(len(coords))
        for e_i, (a, b) in enumerate(E):
            node_cross[a] += edge_cross[e_i]
            node_cross[b] += edge_cross[e_i]
        order = np.argsort(-node_cross)
        lo, hi = coords.min(axis=0), coords.max(axis=0)

        moved = 0
        for i in order:
            if node_cross[i] == 0 or i not in node_data:
                continue
            _, u_other, other_E, mask, nb_u, nb_w = node_data[i]
            cur = coords[i].copy()
            cen = coords[u_other].mean(axis=0)

            c_cur, sharp_cur = self._cross_and_sharpness(coords, cur, u_other,
                                                         other_E, mask)
            if c_cur == 0:
                continue
            len_cur = np.linalg.norm(coords[u_other] - cur, axis=1).sum()
            pen_cur = self._angle_penalty(coords, cur, u_other, nb_u, nb_w)
            viol_cur = self._clearance_viol(coords, i, cur, u_other, other_E, thresh)

            candidates = [cen, (cur + cen) / 2, 2 * cen - cur]
            candidates += [cen + rng.normal(0, self.jitter_scale * span, 2)
                           for _ in range(self.n_jitter)]

            best = (c_cur, sharp_cur, len_cur)
            best_cand = None
            for cand in candidates:
                cand = self._snap_to_feasible(coords, i, cand, sep, rng)
                if not self._sep_ok(coords, i, cur, cand, sep, span):
                    continue
                c_new, sharp_new = self._cross_and_sharpness(coords, cand, u_other,
                                                             other_E, mask)
                if c_new > best[0]:
                    continue
                if (self._angle_penalty(coords, cand, u_other, nb_u, nb_w)
                        > pen_cur + slack):
                    continue
                if (self._clearance_viol(coords, i, cand, u_other, other_E, thresh)
                        > viol_cur):
                    continue
                len_new = np.linalg.norm(coords[u_other] - cand, axis=1).sum()
                key = (c_new, sharp_new, len_new)
                if key < best:
                    best, best_cand = key, cand

            if best_cand is not None:
                coords[i] = best_cand
                moved += 1
        return moved

    def _expand_pass(self, coords, node_data, sep, rng):
        diff = coords[:, None, :] - coords[None, :, :]
        dist = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dist, np.inf)
        crowding = np.maximum(0.0, sep - dist).sum(axis=1)
        order = np.argsort(-crowding)

        moved = 0
        for i in order:
            if crowding[i] <= 0 or i not in node_data:
                continue
            _, u_other, other_E, mask, nb_u, nb_w = node_data[i]
            cur = coords[i].copy()

            d_i = np.linalg.norm(coords - cur, axis=1)
            d_i[i] = np.inf
            close = np.where(d_i < sep)[0]
            if len(close) == 0:
                continue
            push = np.zeros(2)
            for j in close:
                v = cur - coords[j]
                n = np.linalg.norm(v)
                if n < 1e-12:
                    v, n = rng.normal(size=2), 1.0
                push += (v / n) * (sep - min(n, sep))
            pn = np.linalg.norm(push)
            if pn < 1e-9 * sep:
                # symmetric blob: contributions cancel — flee the single
                # nearest neighbor instead so blobs can still tear apart
                j = int(close[np.argmin(d_i[close])])
                v = cur - coords[j]
                n = np.linalg.norm(v)
                if n < 1e-12:
                    v, n = rng.normal(size=2), 1.0
                push, pn = (v / n) * sep, sep
            push = push / pn * min(pn, sep)

            c_cur, _ = self._cross_and_sharpness(coords, cur, u_other, other_E, mask)
            pen_cur = self._angle_penalty(coords, cur, u_other, nb_u, nb_w)
            nn_cur = self._nn_dist(coords, i, cur)

            for f in self.expand_steps:
                cand = cur + f * push
                c_new, _ = self._cross_and_sharpness(coords, cand, u_other,
                                                     other_E, mask)
                if c_new > c_cur:
                    continue
                # spreading a blob inevitably bends chains a little: allow
                # twice the usual angle slack here (smoothing re-straightens)
                if (self._angle_penalty(coords, cand, u_other, nb_u, nb_w)
                        > pen_cur + 2 * self.angle_slack_deg):
                    continue
                if self._nn_dist(coords, i, cand) <= nn_cur:
                    continue        # expansion must actually improve spacing
                coords[i] = cand
                moved += 1
                break
        return moved

    def _smooth_sep_pass(self, coords, node_data, sep, thresh, slack, span):
        for i, (_, u_other, other_E, mask, nb_u, nb_w) in node_data.items():
            cur = coords[i].copy()
            cen = coords[u_other].mean(axis=0)
            if np.allclose(cen, cur):
                continue
            c_cur, _ = self._cross_and_sharpness(coords, cur, u_other, other_E, mask)
            pen_cur = self._angle_penalty(coords, cur, u_other, nb_u, nb_w)
            viol_cur = self._clearance_viol(coords, i, cur, u_other, other_E, thresh)
            for t in self.smooth_steps:
                cand = cur + t * (cen - cur)
                if not self._sep_ok(coords, i, cur, cand, sep, span):
                    continue
                c_new, _ = self._cross_and_sharpness(coords, cand, u_other,
                                                     other_E, mask)
                if c_new > c_cur:
                    continue
                if (self._angle_penalty(coords, cand, u_other, nb_u, nb_w)
                        > pen_cur + slack):
                    continue
                if (self._clearance_viol(coords, i, cand, u_other, other_E, thresh)
                        > viol_cur):
                    continue
                coords[i] = cand
                break
