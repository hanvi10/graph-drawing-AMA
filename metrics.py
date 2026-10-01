"""
metrics.py
==========
Quality metrics for evaluating graph drawings.

All functions take:
    G   -- a NetworkX graph
    pos -- a dict {node: (x, y)} with 2D coordinates for each node

The three metrics used in the paper are:
    - Edge crossings      (lower is better)
    - Mean edge length    (lower is better)
    - Path continuity     (lower is better — means straighter paths)
plus one extra:
    - Edge length variance (lower is better — more uniform edge lengths)

The definitions follow the paper authors' code (src/graph_utils.py in
github.com/raulhigueras/graph-drawing-edge-relaxation), which reproduces the
paper's reported numbers (checked on urban_streets/new-york, Fig. 2a):
    - Length metrics first rescale the drawing with nx.rescale_layout_dict
      (centred, largest |coordinate| = 1). The paper's text says "scaled to a
      1×1 box", but its numbers only match this [-1, 1] rescale. Doing it
      inside the metric makes the result independent of the input scale.
    - Path continuity uses shortest paths of 3–5 NODES (2–4 edges).
"""

import math
from itertools import combinations

import networkx as nx
import numpy as np


# ============================================================================
# EDGE CROSSINGS
# ============================================================================

def _same_side(pos, p, q, a, b) -> bool:
    """Return True if points a and b are on the same side of the line through p and q."""
    dx = pos[p][0] - pos[q][0]
    dy = pos[p][1] - pos[q][1]
    dxa = pos[a][0] - pos[p][0]
    dya = pos[a][1] - pos[p][1]
    dxb = pos[b][0] - pos[p][0]
    dyb = pos[b][1] - pos[p][1]
    return (dy * dxa - dx * dya > 0) == (dy * dxb - dx * dyb > 0)


def _edges_cross(pos, e1, e2) -> bool:
    """Return True if edge e1 and edge e2 cross each other."""
    a, b = e1
    c, d = e2
    ax, ay = pos[a]; bx, by = pos[b]
    cx, cy = pos[c]; dx, dy = pos[d]

    # Fast rejection: if bounding boxes don't overlap, edges can't cross
    if min(ax, bx) > max(cx, dx) or max(ax, bx) < min(cx, dx):
        return False
    if min(ay, by) > max(cy, dy) or max(ay, by) < min(cy, dy):
        return False

    if a == c or a == d or b == c or b == d:
        return False

    return not (_same_side(pos, a, b, c, d) or _same_side(pos, c, d, a, b))


def count_crossings(G: nx.Graph, pos: dict) -> int:
    """Count the total number of edge crossings in the drawing."""
    edges = list(G.edges())
    return sum(1 for e1, e2 in combinations(edges, 2) if _edges_cross(pos, e1, e2))


# ============================================================================
# EDGE LENGTH
# ============================================================================

def _all_edge_lengths(G: nx.Graph, pos: dict) -> list[float]:
    """Return a list of lengths for all edges."""
    return [math.hypot(pos[v][0] - pos[u][0], pos[v][1] - pos[u][1])
            for u, v in G.edges()]


def _edge_length_stats(G: nx.Graph, pos: dict) -> tuple[float, float]:
    """Return (mean, variance) of edge lengths after rescaling to [-1, 1]."""
    pos = nx.rescale_layout_dict(pos)
    lengths = _all_edge_lengths(G, pos)
    if not lengths:
        return 0.0, 0.0
    arr = np.array(lengths)
    return float(arr.mean()), float(arr.var())


def mean_edge_length(G: nx.Graph, pos: dict) -> float:
    """Average edge length, measured on the drawing rescaled to [-1, 1]."""
    mean, _ = _edge_length_stats(G, pos)
    return mean


def edge_length_variance(G: nx.Graph, pos: dict) -> float:
    """Variance of edge lengths on the [-1, 1]-rescaled drawing (how uniform they are)."""
    _, var = _edge_length_stats(G, pos)
    return var


# ============================================================================
# PATH CONTINUITY
# ============================================================================
# Measures how "straight" paths look in the drawing.
# For every node pair whose shortest path has 3–5 nodes (2–4 edges), take
# one shortest path between them and compute the mean turning angle at its
# intermediate nodes (0° = straight, 90° = right-angle turn, 180° = doubling
# back).
# Lower is better.

def _angle_between_segments(p1, p2, p3) -> float:
    """
    Angle (in degrees) at node p2 along the path p1 → p2 → p3.
    0° = straight, 90° = right-angle turn.
    """
    ax, ay = p2[0] - p1[0], p2[1] - p1[1]   # incoming vector
    bx, by = p3[0] - p2[0], p3[1] - p2[1]   # outgoing vector
    len_a = math.hypot(ax, ay)
    len_b = math.hypot(bx, by)
    if len_a == 0 or len_b == 0:
        return 0.0
    cos_angle = (ax * bx + ay * by) / (len_a * len_b)
    return math.degrees(math.acos(max(-1.0, min(1.0, cos_angle))))


def path_continuity(G: nx.Graph, pos: dict,
                    min_nodes: int = 3, max_nodes: int = 5) -> float:
    """
    Mean path continuity over shortest paths with min_nodes–max_nodes nodes.

    Same as the paper's code (graph_utils.continuity): for each ordered pair
    (u, v) (so each pair counts twice), take the shortest path that
    nx.all_pairs_shortest_path returns, keep it if it has 3–5 nodes, compute
    the mean turning angle at its intermediate nodes, then average those
    per-path means. Angles are scale-independent, so no rescale is needed.
    Lower = straighter paths = better.
    """
    total, count = 0.0, 0
    # cutoff counts edges; paths it finds are the same as without a cutoff
    for u, paths in nx.all_pairs_shortest_path(G, cutoff=max_nodes - 1):
        for v, path in paths.items():
            if not (min_nodes <= len(path) <= max_nodes):
                continue
            angles = [
                _angle_between_segments(pos[path[k-1]], pos[path[k]], pos[path[k+1]])
                for k in range(1, len(path) - 1)
            ]
            total += float(np.mean(angles))
            count += 1

    return total / count if count else 0.0


# ============================================================================
# CONVENIENCE: compute all metrics at once
# ============================================================================

def all_metrics(G: nx.Graph, pos: dict) -> dict:
    """Return a dict with all four quality metrics."""
    mean_len, len_var = _edge_length_stats(G, pos)
    return {
        "crossings":          count_crossings(G, pos),
        "mean_edge_length":   mean_len,
        "edge_length_var":    len_var,
        "path_continuity":    path_continuity(G, pos),
    }
