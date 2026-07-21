"""
presentation/make_centrality_figure.py
=======================================
Slide figure: shortest-path edge betweenness (EBC, the paper's edge score)
vs edge current-flow betweenness (the score our champion pipeline uses).

The example graph is chosen to make the difference unmissable: two communities
joined by TWO competing routes —

    direct bridge   a0 —————————— b0      (crossing cost 1)
    detour bridge   a1 — x0 — x1 — b1     (crossing cost 3)

Every shortest path between the communities must use the DIRECT bridge, so EBC
gives the detour almost nothing — shortest-path betweenness is winner-take-all.
Current flow instead splits the load across both routes like resistors in
parallel, so the detour lights up as the genuine alternative route it is.

That is exactly the "soft bottleneck EBC misses" that motivates using
current-flow as the relaxation score in algorithms/edge_relaxation/currentflow.py.

HOW TO RUN:
    ./.venv/bin/python presentation/make_centrality_figure.py

OUTPUT:
    presentation/ebc_vs_currentflow.png
"""

import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_PATH = os.path.join(OUT_DIR, "ebc_vs_currentflow.png")

CLUSTER_SIZE = 8          # big enough that cross-community traffic dominates
INK = "#333333"
NODE_COLOR = "#3b3b3b"

# Reds bottoms out at white, which would make the low-centrality community
# edges vanish. Use the ramp truncated to its upper end instead. The colorbar
# uses this same map, so what you read off it is exactly what is drawn.
CMAP = LinearSegmentedColormap.from_list(
    "reds_upper", plt.cm.Reds(np.linspace(0.18, 1.0, 256)))

EDGE_WIDTH = 2.8          # uniform: colour alone encodes centrality


def _key(e):
    return tuple(sorted(e))


def build_graph():
    """Two cliques joined by a direct bridge and a longer detour bridge."""
    G = nx.Graph()
    A = [f"a{i}" for i in range(CLUSTER_SIZE)]
    B = [f"b{i}" for i in range(CLUSTER_SIZE)]
    G.add_edges_from((u, v) for i, u in enumerate(A) for v in A[i + 1:])
    G.add_edges_from((u, v) for i, u in enumerate(B) for v in B[i + 1:])

    G.add_edge("a0", "b0")                       # direct bridge  (cost 1)
    nx.add_path(G, ["a1", "x0", "x1", "b1"])     # detour bridge  (cost 3)
    return G, A, B


def _angular_gap(a, b):
    """Smallest absolute angle between two bearings, in degrees."""
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


def build_pos(A, B):
    """Each cluster is a REGULAR polygon; the bridges run between them.

    The polygon is rotated so one vertex faces the other cluster — that vertex
    anchors the direct bridge (node 0) — and the vertex nearest the bottom
    anchors the detour (node 1). The remaining nodes fill the other vertices in
    order, so the spacing stays perfectly even.
    """
    pos = {}
    radius = 1.3
    for nodes, center, facing in ((A, np.array([-3.4, 0.7]), 0.0),
                                  (B, np.array([3.4, 0.7]), 180.0)):
        n = len(nodes)
        slots = [(facing + k * 360.0 / n) % 360.0 for k in range(n)]
        bottom = min(range(n), key=lambda k: _angular_gap(slots[k], 270.0))
        order = [0, bottom] + [k for k in range(n) if k not in (0, bottom)]
        for v, k in zip(nodes, order):
            a = math.radians(slots[k])
            pos[v] = center + radius * np.array([math.cos(a), math.sin(a)])

    pos["x0"] = np.array([-1.35, -2.75])
    pos["x1"] = np.array([1.35, -2.75])
    return pos


def draw_panel(ax, G, pos, cent, title):
    edges = list(G.edges())
    vals = np.array([cent[_key(e)] for e in edges])
    # Normalize per panel so the busiest edge is full-saturation in BOTH panels.
    # The comparison is about WHERE the load concentrates, not absolute units.
    rel = vals / vals.max()

    nx.draw_networkx_edges(G, pos, ax=ax, edgelist=edges, edge_color=rel,
                           edge_cmap=CMAP, edge_vmin=0.0, edge_vmax=1.0,
                           width=EDGE_WIDTH)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=55,
                           node_color=NODE_COLOR, linewidths=0)
    ax.set_title(title, fontsize=14, color=INK, pad=14, fontweight="bold")
    ax.set_axis_off()
    ax.set_aspect("equal")
    return rel


def main():
    G, A, B = build_graph()
    pos = build_pos(A, B)

    ebc = {_key(e): v for e, v in
           nx.edge_betweenness_centrality(G, normalized=True).items()}
    cfb = {_key(e): v for e, v in
           nx.edge_current_flow_betweenness_centrality(G, normalized=True).items()}

    # Figure aspect is matched to the data extent so the equal-aspect axes do
    # not letterbox and leave dead whitespace.
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.2))
    draw_panel(axes[0], G, pos, ebc, "EBC")
    draw_panel(axes[1], G, pos, cfb, "Current flow")

    for ax in axes:
        ax.set_xlim(-5.2, 5.2)
        ax.set_ylim(-3.4, 2.5)

    sm = ScalarMappable(cmap=CMAP, norm=Normalize(vmin=0, vmax=1))
    cbar = fig.colorbar(sm, ax=axes, fraction=0.025, pad=0.02)
    cbar.set_label("edge centrality (relative to busiest edge)",
                   fontsize=10, color=INK)

    fig.savefig(OUT_PATH, dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)

    # Console summary — the numbers behind the slide.
    print(f"Saved: {OUT_PATH}\n")
    print(f"{'':22s} {'EBC':>8s} {'curr-flow':>11s}")
    for label, e in [("direct bridge a0-b0", ("a0", "b0")),
                     ("detour x0-x1", ("x0", "x1"))]:
        print(f"{label:22s} {ebc[_key(e)]:8.3f} {cfb[_key(e)]:11.3f}")
    r_ebc = ebc[_key(("x0", "x1"))] / ebc[_key(("a0", "b0"))]
    r_cfb = cfb[_key(("x0", "x1"))] / cfb[_key(("a0", "b0"))]
    print(f"\ndetour valued at {r_ebc:.0%} of the direct bridge by EBC, "
          f"but {r_cfb:.0%} by current flow.")


if __name__ == "__main__":
    main()
