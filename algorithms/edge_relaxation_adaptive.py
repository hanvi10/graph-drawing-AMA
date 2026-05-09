"""
algorithms/edge_relaxation_adaptive.py
========================================
Adaptive Crossing-Participation Re-scoring.

Starts with crossing participation score (like edge_relaxation_crossing.py),
but recomputes the base scores from the current layout every `rescore_every`
iterations. As the layout evolves, different edges become the crossing
bottlenecks — static scores go stale. Adaptive re-scoring keeps the selection
aligned with the drawing as it actually looks at each stage.
"""

from itertools import combinations

import networkx as nx
import numpy as np

from algorithms.base import GraphDrawingAlgorithm
from metrics import count_crossings


def _same_side(pos, p, q, a, b) -> bool:
    dx = pos[p][0] - pos[q][0]
    dy = pos[p][1] - pos[q][1]
    dxa = pos[a][0] - pos[p][0]
    dya = pos[a][1] - pos[p][1]
    dxb = pos[b][0] - pos[p][0]
    dyb = pos[b][1] - pos[p][1]
    return (dy * dxa - dx * dya > 0) == (dy * dxb - dx * dyb > 0)


def _edges_cross(pos, e1, e2) -> bool:
    a, b = e1
    c, d = e2
    ax, ay = pos[a]; bx, by = pos[b]
    cx, cy = pos[c]; dx, dy = pos[d]
    if min(ax, bx) > max(cx, dx) or max(ax, bx) < min(cx, dx):
        return False
    if min(ay, by) > max(cy, dy) or max(ay, by) < min(cy, dy):
        return False
    if a == c or a == d or b == c or b == d:
        return False
    return not (_same_side(pos, a, b, c, d) or _same_side(pos, c, d, a, b))


def _crossing_participation(edges, pos) -> dict:
    counts = {e: 0 for e in edges}
    for e1, e2 in combinations(edges, 2):
        if _edges_cross(pos, e1, e2):
            counts[e1] += 1
            counts[e2] += 1
    return counts


class EdgeRelaxationAdaptive(GraphDrawingAlgorithm):
    """
    Adaptive edge relaxation: recomputes crossing participation every K steps (D1).
    """

    def __init__(
        self,
        k_r: float = 0.1,
        k_w: float = 0.05,
        max_iter: int = 100,
        patience: int = 20,
        seed: int = 42,
        initial_layout_iterations: int = 50,
        loop_spring_iters: int = 50,
        rescore_every: int = 10,
    ):
        self.k_r = k_r
        self.k_w = k_w
        self.max_iter = max_iter
        self.patience = patience
        self.seed = seed
        self.initial_layout_iterations = initial_layout_iterations
        self.loop_spring_iters = loop_spring_iters
        self.rescore_every = rescore_every

    @property
    def name(self) -> str:
        return f"edge_relaxation_adaptive(kr={self.k_r},kw={self.k_w},k={self.rescore_every})"

    def layout(self, G: nx.Graph) -> dict:
        G = G.copy()
        np.random.seed(self.seed)

        pos = self._initial_layout(G, self.seed, self.initial_layout_iterations)

        edges = list(G.edges())
        raw = _crossing_participation(edges, pos)
        base_score = {e: raw.get(e, raw.get((e[1], e[0]), 0)) for e in edges}

        scale  = {e: 1.0 for e in edges}
        weight = {e: 1.0 for e in edges}
        scores = {e: weight[e] * base_score[e] for e in edges}

        best_crossings    = np.inf
        best_crossings_it = -1
        best_pos          = None
        last_pos          = pos.copy()

        for it in range(self.max_iter):
            if it > 0 and it % self.rescore_every == 0:
                raw = _crossing_participation(edges, last_pos)
                base_score = {e: raw.get(e, raw.get((e[1], e[0]), 0)) for e in edges}
                for e in edges:
                    scores[e] = weight[e] * base_score[e]

            selected = max(scores, key=scores.get)

            scale[selected]  *= self.k_r
            weight[selected] *= self.k_w
            scores[selected]  = weight[selected] * base_score[selected]

            G[selected[0]][selected[1]]['relax'] = scale[selected]

            pos = nx.spring_layout(G, pos=last_pos.copy(), weight='relax',
                                   iterations=self.loop_spring_iters)
            crossings = count_crossings(G, pos)

            if crossings < best_crossings:
                best_crossings    = crossings
                best_crossings_it = it
                best_pos          = pos

            if best_crossings_it + self.patience < it:
                break

            last_pos = pos

        return best_pos if best_pos is not None else pos
