"""
make_renders.py
===============
Renders every readable graph (<= MAX_EDGES edges) once per algorithm, each
algorithm in its own folder:

    data/figures/renders/random/<safe_name>.png
    data/figures/renders/baseline/<safe_name>.png
    data/figures/renders/ebc/<safe_name>.png
    data/figures/renders/currentflow/<safe_name>.png
    data/figures/renders/currentflow_aesthetic/<safe_name>.png
    data/figures/renders/cf_cross_sep/<safe_name>.png

Gallery style (blue nodes, thin gray edges), title = graph name + crossings.
Baseline / EBC / currentflow layouts go through layout_cache, so anything
already computed (e.g. by a cf_cross_sep benchmark run) is reused, and this
script warms the cache for everyone else.

HOW TO RUN:
    ./.venv/bin/python make_renders.py --workers 6
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
from algorithms.novel.cf_cross_sep import CFCrossSep
from algorithms.novel.currentflow_aesthetic import CurrentFlowAesthetic

MAX_EDGES = 400
SEED = 42

ROOT       = os.path.dirname(__file__)
GRAPHS_DIR = os.path.join(ROOT, "data", "graphs")
OUT_ROOT   = os.path.join(ROOT, "data", "figures", "renders")

NODE_COLOR = "#2f5f9e"
EDGE_COLOR = "#9aa5b1"
INK        = "#333333"

ALGOS = ["random", "baseline", "ebc", "currentflow",
         "currentflow_aesthetic", "cf_cross_sep"]


def layout_for(algo, G):
    if algo == "random":
        return dict(nx.random_layout(G, seed=SEED))
    if algo == "baseline":
        return cached_layout(f"baseline_s{SEED}", G,
                             lambda: Baseline(seed=SEED).layout(G))
    if algo == "ebc":
        return cached_layout(f"ebc_s{SEED}", G,
                             lambda: EdgeRelaxation(seed=SEED).layout(G))
    if algo == "currentflow":
        return cached_layout(
            f"currentflow_s{SEED}", G,
            lambda: EdgeRelaxationCurrentFlow(seed=SEED).layout(G))
    if algo == "cf_cross_sep":
        return CFCrossSep(seed=SEED).layout(G)   # reuses the cf cache itself
    if algo == "currentflow_aesthetic":
        # its layout() computes currentflow internally; compose the same
        # stages from the cached cf layout instead
        cfa = CurrentFlowAesthetic(seed=SEED)
        pos = cached_layout(
            f"currentflow_s{SEED}", G,
            lambda: EdgeRelaxationCurrentFlow(seed=SEED).layout(G))
        pos = cfa._repair.repair(G, pos)
        pos = cfa._repair.expand(G, pos)
        return cfa._polish.repair(G, pos)
    raise ValueError(algo)


def render_graph(args):
    name, safe_name = args
    G = nx.read_graphml(os.path.join(GRAPHS_DIR, f"{safe_name}.graphml"))
    G = nx.convert_node_labels_to_integers(G)
    n = G.number_of_nodes()
    node_size = max(4, 40 - n // 15)

    for algo in ALGOS:
        out_path = os.path.join(OUT_ROOT, algo, f"{safe_name}.png")
        if os.path.exists(out_path):
            continue
        pos = layout_for(algo, G)
        fig, ax = plt.subplots(figsize=(6.4, 6.4))
        nx.draw_networkx_edges(G, pos, ax=ax, width=0.6, edge_color=EDGE_COLOR)
        nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_size,
                               node_color=NODE_COLOR, linewidths=0)
        ax.set_title(f"{name} — {count_crossings(G, pos)} crossings",
                     fontsize=11, color=INK)
        ax.set_axis_off()
        ax.set_aspect("equal")
        fig.tight_layout()
        fig.savefig(out_path, dpi=150, facecolor="white")
        plt.close(fig)
    return name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    for algo in ALGOS:
        os.makedirs(os.path.join(OUT_ROOT, algo), exist_ok=True)

    md = pd.read_csv(os.path.join(ROOT, "data", "metadata.csv"))
    sel = md[md.edges <= MAX_EDGES]
    tasks = [(r["name"], r["safe_name"]) for _, r in sel.iterrows()]
    print(f"Rendering {len(tasks)} graphs x {len(ALGOS)} algorithms -> {OUT_ROOT}")

    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(render_graph, t): t[0] for t in tasks}
        for fut in as_completed(futures):
            done += 1
            try:
                print(f"[{done}/{len(tasks)}] {fut.result()}")
            except Exception as exc:
                print(f"[{done}/{len(tasks)}] {futures[fut]} FAILED: {exc}")
    print("Renders complete.")


if __name__ == "__main__":
    main()
