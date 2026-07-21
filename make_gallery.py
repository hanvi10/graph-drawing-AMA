"""
make_gallery.py
===============
Renders a 6-panel image per graph: random ("chaos") layout, baseline
(spectral+spring), EBC relaxation (the paper), currentflow relaxation,
currentflow+repair, and cf_cross_sep (the full guaranteed-separation pipeline).
Only graphs readable as a picture are included (<= MAX_EDGES edges).

HOW TO RUN:
    ./.venv/bin/python make_gallery.py            # all readable graphs
    ./.venv/bin/python make_gallery.py --workers 6

OUTPUT:
    data/figures/gallery/<safe_name>.png
"""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

from metrics import count_crossings
from layout_cache import cached_layout
from algorithms.baseline import Baseline
from algorithms.edge_relaxation.ebc import EdgeRelaxation
from algorithms.edge_relaxation.currentflow import EdgeRelaxationCurrentFlow
from algorithms.novel.crossing_repair import CrossingRepair
from algorithms.novel.cf_cross_sep import CFCrossSep

MAX_EDGES = 400
SEED = 42       # cache keys (baseline_s42 / ebc_s42 / currentflow_s42) shared
                # with make_renders.py, so the expensive layouts are reused

ROOT        = os.path.dirname(__file__)
GRAPHS_DIR  = os.path.join(ROOT, "data", "graphs")
OUT_DIR     = os.path.join(ROOT, "data", "figures", "gallery")

NODE_COLOR = "#2f5f9e"
EDGE_COLOR = "#9aa5b1"
INK        = "#333333"


def _draw_panel(ax, G, pos, title):
    n = G.number_of_nodes()
    node_size = max(4, 40 - n // 15)          # smaller dots on bigger graphs
    nx.draw_networkx_edges(G, pos, ax=ax, width=0.6, edge_color=EDGE_COLOR)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_size,
                           node_color=NODE_COLOR, linewidths=0)
    ax.set_title(title, fontsize=10, color=INK)
    ax.set_axis_off()
    ax.set_aspect("equal")


def render_graph(args):
    name, safe_name, dpi = args
    out_path = os.path.join(OUT_DIR, f"{safe_name}.png")

    G = nx.read_graphml(os.path.join(GRAPHS_DIR, f"{safe_name}.graphml"))
    G = nx.convert_node_labels_to_integers(G)

    # Shared layouts come from the disk cache (populated by make_renders.py);
    # repair + separation are cheap and reuse the cached currentflow layout.
    algo = CFCrossSep()
    cf_pos = cached_layout(f"currentflow_s{SEED}", G,
                           lambda: EdgeRelaxationCurrentFlow(seed=SEED).layout(G))
    rep_pos = algo._repair.repair(G, cf_pos)
    sep_pos = algo._ensure_floors(
        G, algo._polish.repair(G, algo._polish.anneal_expand(G, rep_pos)))

    layouts = [
        ("Random",           dict(nx.random_layout(G, seed=SEED))),
        ("Baseline",         cached_layout(f"baseline_s{SEED}", G,
                                           lambda: Baseline(seed=SEED).layout(G))),
        ("EBC (paper)",      cached_layout(f"ebc_s{SEED}", G,
                                           lambda: EdgeRelaxation(seed=SEED).layout(G))),
        ("CurrentFlow",      cf_pos),
        ("CF + Repair",      rep_pos),
        ("cf_cross_sep",     sep_pos),
    ]

    fig, axes = plt.subplots(1, 6, figsize=(26, 4.8))
    for ax, (label, pos) in zip(axes, layouts):
        crossings = count_crossings(G, pos)
        _draw_panel(ax, G, pos, f"{label} — {crossings} crossings")
    fig.suptitle(f"{name}   ({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)",
                 fontsize=12, color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=dpi, facecolor="white")
    plt.close(fig)
    return name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--dpi", type=int, default=300,
                        help="output resolution (110 was the old default)")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    md = pd.read_csv(os.path.join(ROOT, "data", "metadata.csv"))
    sel = md[md.edges <= MAX_EDGES]
    tasks = [(r["name"], r["safe_name"], args.dpi) for _, r in sel.iterrows()]
    print(f"Rendering {len(tasks)} graphs @ {args.dpi} dpi -> {OUT_DIR}")

    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(render_graph, t): t[0] for t in tasks}
        for fut in as_completed(futures):
            done += 1
            try:
                print(f"[{done}/{len(tasks)}] {fut.result()}")
            except Exception as exc:
                print(f"[{done}/{len(tasks)}] {futures[fut]} FAILED: {exc}")

    print("Gallery complete.")


if __name__ == "__main__":
    main()
