"""
algorithms2/render_compare.py
=============================
Three drawings of each graph, side by side, so the separation step can be
judged by eye and not only by the metrics table:

    old baseline        spectral + Fruchterman-Reingold (algorithms/baseline.py)
    FA2, no floor       spectral + ForceAtlas2, min_dist_frac=0
    FA2 + min distance  the same drawing with the floor applied

The middle and right panels differ ONLY by the separation step, so the visible
difference between them is exactly what `min_dist_frac` buys.

Nodes inside the floor are drawn in red in the middle panel: those are the ones
a reader cannot separate, and the right panel is what happens to them.

    ./.venv/bin/python -m algorithms2.render_compare
    ./.venv/bin/python -m algorithms2.render_compare --graphs lesmis,zebras --frac 0.4

Output: data/figures/algorithms2/<graph>.png
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.spatial import cKDTree

from algorithms.baseline import Baseline as OldBaseline
from algorithms2.baseline import Baseline
from algorithms2.separation import (enforce_min_distance, ideal_distance,
                                    separation_report)
from metrics import count_crossings

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GRAPHS_DIR = os.path.join(REPO, "data", "graphs")
OUT_DIR = os.path.join(REPO, "data", "figures", "algorithms2")

NODE_COLOR = "#2f5f9e"
FLAG_COLOR = "#d1495b"
EDGE_COLOR = "#9aa5b1"
INK = "#333333"

# graphs that show the effect most clearly: heavily clumped by FA2, plus two
# that are already clear so the floor is visibly a no-op on them
DEFAULT = ["celegans_interactomes__BPmaps", "urban_streets__irvine2",
           "london_transport", "mist__ppi_zebrafish", "lesmis",
           "urban_streets__new-york", "football", "zebras"]


def _clumped_mask(G, pos, frac):
    coords = np.array([pos[v] for v in G], dtype=float)
    if len(coords) < 2:
        return np.zeros(len(coords), dtype=bool)
    nearest = cKDTree(coords).query(coords, k=2)[0][:, 1]
    return nearest < frac * ideal_distance(coords)


def panel(ax, G, pos, title, frac, flag=False):
    node_size = max(4, 40 - G.number_of_nodes() // 15)
    nx.draw_networkx_edges(G, pos, ax=ax, width=0.6, edge_color=EDGE_COLOR)
    if flag:
        mask = _clumped_mask(G, pos, frac)
        colors = [FLAG_COLOR if m else NODE_COLOR for m in mask]
        title += f"\n{mask.sum()} nodes inside the floor"
    else:
        colors = NODE_COLOR
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_size,
                           node_color=colors, linewidths=0)
    ax.set_title(title, fontsize=10, color=INK)
    ax.set_axis_off()
    ax.set_aspect("equal")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graphs", default=",".join(DEFAULT))
    ap.add_argument("--frac", type=float, default=0.30)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    for name in args.graphs.split(","):
        path = os.path.join(GRAPHS_DIR, f"{name}.graphml")
        if not os.path.exists(path):
            print(f"  skipped (no such graph): {name}")
            continue
        G = nx.convert_node_labels_to_integers(nx.read_graphml(path))

        old = OldBaseline(seed=args.seed).layout(G)
        raw = Baseline(seed=args.seed, min_dist_frac=0.0).layout(G)
        sep = enforce_min_distance(raw, args.frac, seed=args.seed)
        rep = separation_report(sep, args.frac)

        fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.6))
        panel(axes[0], G, old,
              f"old baseline: spectral + Fruchterman-Reingold\n"
              f"{count_crossings(G, old)} crossings", args.frac)
        panel(axes[1], G, raw,
              f"spectral + ForceAtlas2, no floor\n"
              f"{count_crossings(G, raw)} crossings", args.frac, flag=True)
        panel(axes[2], G, sep,
              f"+ min distance (frac={args.frac})\n"
              f"{count_crossings(G, sep)} crossings, "
              f"closest pair {rep['min_dist_over_k']:.2f}k", args.frac)
        fig.suptitle(f"{name}  ({G.number_of_nodes()} nodes, "
                     f"{G.number_of_edges()} edges)", fontsize=12, color=INK)
        fig.tight_layout(rect=(0, 0, 1, 0.94))   # leave room for the suptitle
        out = os.path.join(OUT_DIR, f"{name}.png")
        fig.savefig(out, dpi=140, facecolor="white")
        plt.close(fig)
        print(f"  {out}")


if __name__ == "__main__":
    sys.exit(main())
