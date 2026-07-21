"""
presentation/make_circuit_figure.py
====================================
Current flow, drawn as the physical circuit it actually is.

Three parallel routes from I (initial) to F (final), every edge a 1 ohm
resistor. 1 A is injected at I and drawn out at F. Node potentials come from
solving the graph Laplacian system  L v = b  (b = +1 at I, -1 at F); each
edge's current is its voltage drop (conductance = 1). Edge colour = current
carried; nothing is labelled, the colour is the message.

The routes have resistance 1, 2 and 3 ohm, so the current splits 6/11, 3/11,
2/11 = 0.545 / 0.273 / 0.182 A -- exactly the parallel-resistor rule. Every
route carries current, inversely to its resistance; none is discarded for being
longer, which is precisely what shortest-path betweenness does do. And every
edge of a series route carries the same current, since it has nowhere else to go.

HOW TO RUN:
    ./.venv/bin/python presentation/make_circuit_figure.py

OUTPUT:
    presentation/currentflow_circuit.png
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize

OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "currentflow_circuit.png")

# Reds bottoms out at white, which would make the weakest edge invisible, so
# use the ramp truncated to its upper end. The colorbar uses this same map, so
# what you read off it is exactly what is drawn.
CMAP = LinearSegmentedColormap.from_list(
    "reds_upper", plt.cm.Reds(np.linspace(0.20, 1.0, 256)))
INK = "#2f3640"

POS = {
    "I": np.array([0.0, 0.0]),
    "F": np.array([7.0, 0.0]),
    "a": np.array([3.5, 2.6]),
    "b": np.array([2.3, -2.6]),
    "c": np.array([4.7, -2.6]),
}


def build_circuit():
    G = nx.Graph()
    G.add_edge("I", "F")                   # 1 ohm route
    nx.add_path(G, ["I", "a", "F"])        # 2 ohm route
    nx.add_path(G, ["I", "b", "c", "F"])   # 3 ohm route
    return G


def solve_currents(G, source="I", sink="F"):
    """Node potentials from L v = b, then edge current = voltage drop."""
    nodes = list(G.nodes())
    idx = {v: k for k, v in enumerate(nodes)}
    L = nx.laplacian_matrix(G, nodelist=nodes).toarray().astype(float)

    b = np.zeros(len(nodes))
    b[idx[source]] = 1.0
    b[idx[sink]] = -1.0
    v = np.linalg.pinv(L) @ b

    potential = {n: v[idx[n]] for n in nodes}
    current = {}
    for u, w in G.edges():
        # orient each edge so current is positive: high potential -> low
        if potential[u] < potential[w]:
            u, w = w, u
        current[(u, w)] = potential[u] - potential[w]
    return current


def draw_resistor(ax, p1, p2, color, lw=3.0, n_zig=5, amp=0.22, span=1.5):
    """Edge drawn as two leads with a resistor zig-zag in the middle.

    `span` is an absolute length, not a fraction of the edge: every resistor is
    1 ohm, so they should all be drawn the same size.
    """
    v = p2 - p1
    length = np.linalg.norm(v)
    d = v / length
    n = np.array([-d[1], d[0]])
    mid = (p1 + p2) / 2.0
    half = min(span, 0.55 * length) / 2.0
    start, end = mid - d * half, mid + d * half

    for lead in ((p1, start), (end, p2)):
        ax.plot(*zip(*lead), color=color, lw=lw, solid_capstyle="round", zorder=2)

    ts = np.linspace(0.0, 1.0, 2 * n_zig + 1)
    pts = []
    for i, t in enumerate(ts):
        base = start + (end - start) * t
        off = 0.0 if i in (0, len(ts) - 1) else (amp if i % 2 else -amp)
        pts.append(base + n * off)
    pts = np.array(pts)
    ax.plot(pts[:, 0], pts[:, 1], color=color, lw=lw,
            solid_joinstyle="miter", zorder=2)


def draw_arrowhead(ax, p1, p2, color, at=0.74, size=0.30):
    """Small arrowhead along the edge showing the direction of current."""
    d = (p2 - p1) / np.linalg.norm(p2 - p1)
    tip = p1 + (p2 - p1) * at + d * size / 2
    tail = tip - d * size
    ax.annotate("", xy=tip, xytext=tail, zorder=4,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.6,
                                mutation_scale=15, shrinkA=0, shrinkB=0))


def main():
    G = build_circuit()
    current = solve_currents(G)
    imax = max(current.values())
    norm = Normalize(vmin=0.0, vmax=imax)

    fig, ax = plt.subplots(figsize=(11, 7.4))

    for (u, w), amps in current.items():
        # Colour alone carries the current; a uniform stroke keeps every 1 ohm
        # resistor drawn identically.
        draw_resistor(ax, POS[u], POS[w], CMAP(norm(amps)))
        draw_arrowhead(ax, POS[u], POS[w], INK)

    for n, p in POS.items():
        terminal = n in ("I", "F")
        ax.scatter(*p, s=760 if terminal else 300, color=INK, zorder=5,
                   linewidths=0)
        ax.text(*p, n, ha="center", va="center", zorder=6,
                fontsize=15 if terminal else 12, color="white",
                fontweight="bold")

    # terminal leads: current in at I, out at F
    ax.annotate("", xy=POS["I"], xytext=POS["I"] + np.array([-1.9, 0.0]),
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=2.6,
                                mutation_scale=22))
    ax.annotate("", xy=POS["F"] + np.array([1.9, 0.0]), xytext=POS["F"],
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=2.6,
                                mutation_scale=22))

    ax.set_xlim(-3.0, 10.0)
    ax.set_ylim(-4.2, 4.0)
    ax.set_aspect("equal")
    ax.set_axis_off()

    sm = ScalarMappable(cmap=CMAP, norm=norm)
    cbar = fig.colorbar(sm, ax=ax, fraction=0.030, pad=0.02)
    cbar.set_label("current through edge (A)", fontsize=11, color=INK)

    fig.savefig(OUT_PATH, dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {OUT_PATH}\n")
    for (u, w), amps in sorted(current.items(), key=lambda kv: -kv[1]):
        print(f"  {u}-{w:2s}  {amps:.4f} A")
    print(f"\ntotal out of I: {sum(a for (u, _), a in current.items() if u == 'I'):.4f} A")


if __name__ == "__main__":
    main()
