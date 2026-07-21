"""
algorithms/novel/community_collapse.py
=======================================
Collapse-Expand Layout (N1).

Idea: shrink the problem, solve the pieces, then put them back together.

1. COLLAPSE  — Detect communities (Louvain) and contract each one into a
               super-node, giving a small quotient graph.
2. SOLVE     — Lay out the quotient graph (global cluster placement) and each
               community subgraph INDEPENDENTLY. Because bridges are absent
               during the per-community layouts, they exert zero distortion —
               this sidesteps the paper's core problem instead of gradually
               relaxing it away.
3. EXPAND    — Place each community's internal layout at its super-node
               position, scaled by community size.
4. STITCH    — Short weighted spring refinement of the full graph with
               inter-community edges down-weighted, starting from the
               composed positions. Several stitch strengths are tried and
               the layout with fewest crossings wins (crossings are
               scale-invariant, so this selection is fair).

Cost: Louvain + a handful of small layouts + ~3 crossing counts.
Much cheaper than the relaxation loop (which needs ~30+ layout/count rounds).
"""

import math

import networkx as nx
import numpy as np

from algorithms.base import GraphDrawingAlgorithm
from metrics import count_crossings


class CommunityCollapse(GraphDrawingAlgorithm):
    """Collapse communities, lay out parts independently, expand and stitch (N1)."""

    def __init__(
        self,
        seed: int = 42,
        sub_layout_iterations: int = 50,
        stitch_iterations: int = 30,
        stitch_weights: tuple = (0.05, 0.2),
        community_scale: float = 0.6,
    ):
        self.seed = seed
        self.sub_layout_iterations = sub_layout_iterations
        self.stitch_iterations = stitch_iterations
        self.stitch_weights = stitch_weights
        self.community_scale = community_scale

    @property
    def name(self) -> str:
        return "community_collapse"

    def layout(self, G: nx.Graph) -> dict:
        G = G.copy()
        np.random.seed(self.seed)

        communities = nx.community.louvain_communities(G, seed=self.seed)
        communities = [list(c) for c in communities]

        # Degenerate case: one community — nothing to collapse
        if len(communities) < 2:
            return self._initial_layout(G, self.seed, self.sub_layout_iterations)

        node_comm = {}
        for cid, comm in enumerate(communities):
            for v in comm:
                node_comm[v] = cid

        # ── 1. Quotient graph: one node per community ───────────────────────
        Q = nx.Graph()
        Q.add_nodes_from(range(len(communities)))
        for u, v in G.edges():
            cu, cv = node_comm[u], node_comm[v]
            if cu != cv:
                Q.add_edge(cu, cv)

        # ── 2a. Global placement of communities ─────────────────────────────
        if Q.number_of_edges() > 0:
            qpos = self._initial_layout(Q, self.seed, self.sub_layout_iterations)
        else:
            qpos = nx.circular_layout(Q)

        # ── 2b. Independent per-community layouts (bridges absent) ──────────
        n_total = G.number_of_nodes()
        composed = {}
        for cid, comm in enumerate(communities):
            cx, cy = qpos[cid]
            if len(comm) == 1:
                composed[comm[0]] = (cx, cy)
                continue
            sub = G.subgraph(comm)
            if nx.is_connected(sub):
                sub_pos = self._initial_layout(sub, self.seed, self.sub_layout_iterations)
            else:
                sub_pos = nx.spring_layout(sub, seed=self.seed,
                                           iterations=self.sub_layout_iterations)
            # spring_layout output spans roughly [-1, 1]; shrink by size share
            radius = self.community_scale * math.sqrt(len(comm) / n_total)
            for v in comm:
                composed[v] = (cx + sub_pos[v][0] * radius,
                               cy + sub_pos[v][1] * radius)

        # ── 3/4. Stitch: weighted refinement, keep best by crossings ────────
        for u, v in G.edges():
            G[u][v]['stitch'] = 1.0  # intra-community default

        best_pos = composed
        best_crossings = count_crossings(G, composed)

        for w_inter in self.stitch_weights:
            for u, v in G.edges():
                G[u][v]['stitch'] = 1.0 if node_comm[u] == node_comm[v] else w_inter
            pos = nx.spring_layout(G, pos={k: np.array(p) for k, p in composed.items()},
                                   weight='stitch', iterations=self.stitch_iterations,
                                   seed=self.seed)
            crossings = count_crossings(G, pos)
            if crossings < best_crossings:
                best_crossings = crossings
                best_pos = pos

        return {v: tuple(p) for v, p in best_pos.items()}
