"""
algorithms/novel/crossing_repair.py
====================================
Crossing-Guided Node Relocation (N3).

Force-directed algorithms never look at the crossing count they are being
judged on — they optimize spring energy and hope crossings follow. This
algorithm optimizes the objective directly, as a repair pass on top of the
standard spectral+spring layout:

1. Count, for every node, how many crossings its incident edges participate in.
2. Visit nodes in decreasing order of involvement. For each, evaluate a small
   set of candidate positions (neighbor centroid, midpoints, reflection,
   jittered/random samples) and count only the crossings of the node's own
   incident edges — moving one node cannot change any other crossing, so the
   local delta equals the global delta.
3. Accept a move only if it strictly reduces the node's crossings (ties broken
   by shorter total incident edge length, which also helps the edge-length
   metric). Repeat for a few rounds until no move improves.
4. SMOOTHING: relocation moves can leave paths jagged, hurting the paper's
   path-continuity metric. After each round, every node is pulled toward its
   neighbors' centroid as far as possible WITHOUT increasing its incident
   crossings (constrained Laplacian smoothing). This straightens paths and
   shortens edges while preserving the crossing gains.

All segment-intersection tests are vectorized with NumPy, so a full round over
a 1000-edge graph costs a fraction of a second.
"""

import math

import networkx as nx
import numpy as np

from algorithms.base import GraphDrawingAlgorithm


def _cross2d(a, b):
    """z-component of the 2D cross product (np.cross on 2D is deprecated)."""
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def _strict_cross(p1, p2, q1, q2):
    """Vectorized proper-intersection test. Inputs broadcast to (..., 2)."""
    d1 = _cross2d(q2 - q1, p1 - q1)
    d2 = _cross2d(q2 - q1, p2 - q1)
    d3 = _cross2d(p2 - p1, q1 - p1)
    d4 = _cross2d(p2 - p1, q2 - p1)
    return (d1 * d2 < 0) & (d3 * d4 < 0)


class CrossingRepair(GraphDrawingAlgorithm):
    """Greedy crossing-reducing node relocation on top of spectral+spring (N3)."""

    def __init__(
        self,
        seed: int = 42,
        initial_layout_iterations: int = 50,
        max_rounds: int = 5,
        n_jitter: int = 4,
        n_random: int = 0,          # far-flung moves hurt path continuity
        jitter_scale: float = 0.08,
        max_step: float = None,     # optional cap on move distance (× layout span)
        smooth_rounds: int = 2,     # constrained-smoothing rounds after each pass
        smooth_steps: tuple = (1.0, 0.5, 0.25),
        angle_slack_deg: float = 5.0,   # max allowed worsening of the local
                                        # turning-angle proxy per move
        max_continuity_loss: float = 0.05,   # verified global tolerance (5%)
    ):
        self.seed = seed
        self.initial_layout_iterations = initial_layout_iterations
        self.max_rounds = max_rounds
        self.n_jitter = n_jitter
        self.n_random = n_random
        self.jitter_scale = jitter_scale
        self.max_step = max_step
        self.smooth_rounds = smooth_rounds
        self.smooth_steps = smooth_steps
        self.angle_slack_deg = angle_slack_deg
        self.max_continuity_loss = max_continuity_loss

    @property
    def name(self) -> str:
        return "crossing_repair"

    # ── helpers ──────────────────────────────────────────────────────────────

    def _edge_crossing_counts(self, coords, E, share_mask):
        """Crossings per edge over all edge pairs (m, )."""
        P1 = coords[E[:, 0]][:, None, :]
        P2 = coords[E[:, 1]][:, None, :]
        Q1 = coords[E[:, 0]][None, :, :]
        Q2 = coords[E[:, 1]][None, :, :]
        cross = _strict_cross(P1, P2, Q1, Q2) & ~share_mask
        return cross.sum(axis=1)

    def _node_incident_crossings(self, coords, cand, u_other, other_E, mask):
        """
        Crossings of one node's incident edges (endpoints cand→u_other) against
        all non-incident edges `other_E`, excluding pairs sharing an endpoint.
        """
        P1 = cand[None, None, :]                     # (1, 1, 2)
        P2 = coords[u_other][:, None, :]             # (d, 1, 2)
        Q1 = coords[other_E[:, 0]][None, :, :]       # (1, k, 2)
        Q2 = coords[other_E[:, 1]][None, :, :]
        cross = _strict_cross(P1, P2, Q1, Q2) & ~mask
        return int(cross.sum())

    @staticmethod
    def _angle_penalty(coords, cand, u_other, nb_u=None, nb_w=None):
        """
        Local path-continuity proxy for placing a node at `cand`: the mean
        turning angle (degrees, 0 = straight) of
          (a) paths bending AT this node (all pairs of incident edges), and
          (b) paths bending AT each neighbor u where one leg is the edge to
              this node (pairs given by nb_u/nb_w: turn of w — u — node).
        Lower is better. (b) is what dense graphs punish: moving a node bends
        paths at every neighbor, not just at itself.
        """
        parts = []
        d = len(u_other)
        if d >= 2:
            U = coords[u_other] - cand[None, :]
            norms = np.linalg.norm(U, axis=1)
            norms[norms == 0] = 1e-12
            Un = U / norms[:, None]
            cos = np.clip(Un @ Un.T, -1.0, 1.0)
            ang = np.degrees(np.arccos(cos))
            iu = np.triu_indices(d, k=1)
            # turning of a -> node -> b is 180° minus the incident-edge angle
            parts.append(180.0 - ang[iu])
        if nb_u is not None and len(nb_u):
            A = cand[None, :] - coords[nb_u]          # u -> node
            B = coords[nb_w] - coords[nb_u]           # u -> w
            na = np.linalg.norm(A, axis=1); na[na == 0] = 1e-12
            nb = np.linalg.norm(B, axis=1); nb[nb == 0] = 1e-12
            cos = np.clip((A * B).sum(axis=1) / (na * nb), -1.0, 1.0)
            parts.append(180.0 - np.degrees(np.arccos(cos)))
        if not parts:
            return 0.0
        return float(np.concatenate(parts).mean())

    # ── main ─────────────────────────────────────────────────────────────────

    def layout(self, G: nx.Graph) -> dict:
        pos = self._initial_layout(G, self.seed, self.initial_layout_iterations)
        return self.repair(G, pos)

    def repair(self, G: nx.Graph, pos: dict) -> dict:
        """
        Apply the relocation pass to an existing layout `pos`, verifying the
        TRUE path-continuity metric afterwards. Escalation: if continuity
        degraded beyond `max_continuity_loss`, retry with a zero-slack angle
        guard; if that still fails, return the original layout unchanged.
        """
        from metrics import path_continuity
        cont_before = path_continuity(G, pos)

        for slack in (self.angle_slack_deg, 0.0):
            result = self._repair_once(G, pos, slack)
            if cont_before <= 0:
                return result
            if path_continuity(G, result) <= cont_before * (1 + self.max_continuity_loss):
                return result

        return {v: tuple(p) for v, p in pos.items()}

    def _prepare(self, G: nx.Graph, pos: dict):
        """Shared setup for the repair and expansion passes."""
        nodes = list(G.nodes())
        idx = {v: i for i, v in enumerate(nodes)}
        coords = np.array([pos[v] for v in nodes], dtype=float)

        E = np.array([(idx[u], idx[v]) for u, v in G.edges()], dtype=int)
        if len(E) == 0:
            return nodes, coords, E, None, {}

        # Static masks: which edge pairs share an endpoint (never count those)
        share_mask = (
            (E[:, None, 0] == E[None, :, 0]) | (E[:, None, 0] == E[None, :, 1]) |
            (E[:, None, 1] == E[None, :, 0]) | (E[:, None, 1] == E[None, :, 1])
        )

        # Adjacency lists (integer indices) for the neighbor-angle proxy
        adj = {i: [] for i in range(len(nodes))}
        for a, b in E:
            adj[a].append(b)
            adj[b].append(a)

        # Per-node static data: incident edge partners, non-incident edges,
        # and (neighbor, neighbor-of-neighbor) pairs for the angle proxy
        node_data = {}
        for i in range(len(nodes)):
            inc = np.where((E[:, 0] == i) | (E[:, 1] == i))[0]
            if len(inc) == 0:
                continue
            u_other = np.where(E[inc, 0] == i, E[inc, 1], E[inc, 0])
            other = np.where((E[:, 0] != i) & (E[:, 1] != i))[0]
            other_E = E[other]
            # exclude other-edges sharing the incident edge's far endpoint
            mask = ((other_E[None, :, 0] == u_other[:, None]) |
                    (other_E[None, :, 1] == u_other[:, None]))
            nb_u, nb_w = [], []
            for u in u_other:
                for w in adj[u]:
                    if w != i:
                        nb_u.append(u)
                        nb_w.append(w)
            nb_u = np.array(nb_u, dtype=int)
            nb_w = np.array(nb_w, dtype=int)
            node_data[i] = (inc, u_other, other_E, mask, nb_u, nb_w)

        return nodes, coords, E, share_mask, node_data

    def _repair_once(self, G: nx.Graph, pos: dict, slack: float) -> dict:
        nodes, coords, E, share_mask, node_data = self._prepare(G, pos)
        if len(E) == 0:
            return pos

        rng = np.random.default_rng(self.seed)

        for _ in range(self.max_rounds):
            edge_cross = self._edge_crossing_counts(coords, E, share_mask)
            node_cross = np.zeros(len(nodes))
            for e_i, (a, b) in enumerate(E):
                node_cross[a] += edge_cross[e_i]
                node_cross[b] += edge_cross[e_i]

            order = np.argsort(-node_cross)
            lo, hi = coords.min(axis=0), coords.max(axis=0)
            span = float((hi - lo).max()) or 1.0

            moved = 0
            for i in order:
                if node_cross[i] == 0 or i not in node_data:
                    continue
                _, u_other, other_E, mask, nb_u, nb_w = node_data[i]

                cur = coords[i].copy()
                cen = coords[u_other].mean(axis=0)

                c_cur = self._node_incident_crossings(coords, cur, u_other, other_E, mask)
                if c_cur == 0:
                    continue
                len_cur = np.linalg.norm(coords[u_other] - cur, axis=1).sum()
                pen_cur = self._angle_penalty(coords, cur, u_other, nb_u, nb_w)

                candidates = [cen, (cur + cen) / 2, 2 * cen - cur]
                candidates += [cen + rng.normal(0, self.jitter_scale * span, 2)
                               for _ in range(self.n_jitter)]
                candidates += [rng.uniform(lo, hi) for _ in range(self.n_random)]
                if self.max_step is not None:
                    limit = self.max_step * span
                    candidates = [c for c in candidates
                                  if np.linalg.norm(c - cur) <= limit]

                best_c, best_len, best_cand = c_cur, len_cur, None
                for cand in candidates:
                    # degenerate guard: never land (almost) on a neighbor —
                    # a zero-length edge is unreadable and breaks the angles
                    if (np.linalg.norm(coords[u_other] - cand, axis=1).min()
                            < 1e-3 * span):
                        continue
                    c_new = self._node_incident_crossings(coords, cand, u_other,
                                                          other_E, mask)
                    if c_new > best_c:
                        continue
                    # continuity guard: don't let the turning-angle proxy
                    # (at this node AND at its neighbors) degrade beyond slack
                    if (self._angle_penalty(coords, cand, u_other, nb_u, nb_w)
                            > pen_cur + slack):
                        continue
                    len_new = np.linalg.norm(coords[u_other] - cand, axis=1).sum()
                    if c_new < best_c or len_new < best_len - 1e-12:
                        best_c, best_len, best_cand = c_new, len_new, cand

                if best_cand is not None:
                    coords[i] = best_cand
                    moved += 1

            # Constrained Laplacian smoothing: straighten paths / shorten
            # edges without giving back any crossings.
            for _ in range(self.smooth_rounds):
                self._smooth_pass(coords, node_data, slack)

            if moved == 0:
                break

        return {nodes[i]: tuple(coords[i]) for i in range(len(nodes))}

    # ── expansion (readability) pass ─────────────────────────────────────────

    def expand(self, G: nx.Graph, pos: dict,
               min_sep_frac: float = 0.6,
               rounds: int = 5,
               cross_slack: int = 0,
               step_fracs: tuple = (1.0, 0.5, 0.25)) -> dict:
        """
        Anti-clumping pass: push apart nodes that sit closer than
        `min_sep_frac` × k, where k = sqrt(area / n) is Fruchterman-Reingold's
        natural node spacing. (Not a fraction of the mean edge length: after
        repair the edges are short BECAUSE of the clumping, so that scale
        would legitimize the artifact it is meant to fix. The bounding box is
        set by the layout's periphery and stays meaningful.)

        This is spring layout's repulsion force reintroduced as a guarded
        post-pass — a node only moves if its incident crossings increase by at
        most `cross_slack` (default: not at all) and the turning-angle proxy
        stays within the usual slack.

        Readability is bought with edge length (expansion lengthens edges);
        crossings and continuity are protected by the guards.
        """
        nodes, coords, E, _, node_data = self._prepare(G, pos)
        if len(E) == 0:
            return pos
        rng = np.random.default_rng(self.seed)

        for _ in range(rounds):
            w, h = coords.max(axis=0) - coords.min(axis=0)
            k = math.sqrt(max(w * h, 1e-12) / len(nodes))
            sep = min_sep_frac * k
            if sep <= 0:
                break

            # pairwise distances; who is crowded?
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

                # repulsion displacement away from everyone inside `sep`
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
                        v, n = rng.normal(size=2), 1.0   # coincident: random dir
                    push += (v / n) * (sep - min(n, sep))
                pn = np.linalg.norm(push)
                if pn < 1e-12:
                    continue
                push = push / pn * min(pn, sep)          # cap step at `sep`

                c_cur = self._node_incident_crossings(coords, cur, u_other,
                                                      other_E, mask)
                pen_cur = self._angle_penalty(coords, cur, u_other, nb_u, nb_w)

                for f in step_fracs:
                    cand = cur + f * push
                    if (self._node_incident_crossings(coords, cand, u_other,
                                                      other_E, mask)
                            > c_cur + cross_slack):
                        continue
                    if (self._angle_penalty(coords, cand, u_other, nb_u, nb_w)
                            > pen_cur + self.angle_slack_deg):
                        continue
                    coords[i] = cand
                    moved += 1
                    break

            if moved == 0:
                break

        return {nodes[i]: tuple(coords[i]) for i in range(len(nodes))}

    def _smooth_pass(self, coords, node_data, slack):
        """Pull each node toward its neighbor centroid as far as its incident
        crossing count and angle proxy allow (strongest step first)."""
        span = float((coords.max(axis=0) - coords.min(axis=0)).max()) or 1.0
        for i, (_, u_other, other_E, mask, nb_u, nb_w) in node_data.items():
            cur = coords[i].copy()
            cen = coords[u_other].mean(axis=0)
            if np.allclose(cen, cur):
                continue
            c_cur = self._node_incident_crossings(coords, cur, u_other, other_E, mask)
            pen_cur = self._angle_penalty(coords, cur, u_other, nb_u, nb_w)
            for t in self.smooth_steps:
                cand = cur + t * (cen - cur)
                if (np.linalg.norm(coords[u_other] - cand, axis=1).min()
                        < 1e-3 * span):
                    continue    # would create a (near-)zero-length edge
                if (self._node_incident_crossings(coords, cand, u_other,
                                                  other_E, mask) <= c_cur
                        and self._angle_penalty(coords, cand, u_other, nb_u, nb_w)
                            <= pen_cur + slack):
                    coords[i] = cand
                    break
