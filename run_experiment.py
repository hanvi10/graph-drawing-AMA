"""
run_experiment.py
=================
Runs one or more graph drawing algorithms on every graph in the dataset
and saves the results to a CSV file.

HOW TO RUN:
    python3 run_experiment.py                  # uses all CPU cores (parallel)
    python3 run_experiment.py --workers 1      # sequential (easier to debug)

OUTPUT:
    data/results/<run_name>.csv
    One row per (graph, algorithm) combination with all quality metrics.

HOW TO ADD A NEW ALGORITHM:
    1. Create your algorithm in algorithms/edge_relaxation/ (relaxation
       variants) or algorithms/novel/ (everything else)
    2. Register it in run_single.py's ALGORITHM_REGISTRY (or import it below
       in the ALGORITHMS list)
    3. Re-run this script (or run_single.py --algorithm <name>)
"""

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import networkx as nx
import numpy as np
import pandas as pd

from metrics import count_crossings, mean_edge_length, edge_length_variance, path_continuity
from algorithms import Baseline, EdgeRelaxation

METRIC_KEYS = ["crossings", "mean_edge_length", "edge_length_var", "path_continuity", "runtime_s"]

# ============================================================================
# CONFIGURE: which algorithms to run
# ============================================================================
# Add or remove algorithms from this list to control what gets tested.
# Each entry is an instance of a GraphDrawingAlgorithm subclass.

ALGORITHMS = [
    Baseline(),
    EdgeRelaxation(k_r=0.1, k_w=0.05),   # best overall params from the paper
]

# ============================================================================
# PATHS
# ============================================================================

METADATA_PATH  = os.path.join(os.path.dirname(__file__), "data", "metadata.csv")
GRAPHS_DIR     = os.path.join(os.path.dirname(__file__), "data", "graphs")
RESULTS_DIR    = os.path.join(os.path.dirname(__file__), "data", "results")


# ============================================================================
# HELPERS
# ============================================================================

def scale_to_unit_box(pos: dict) -> dict:
    """Scale positions so the drawing fits in a 1×1 bounding box."""
    coords = np.array(list(pos.values()))
    min_xy = coords.min(axis=0)
    scale  = (coords.max(axis=0) - min_xy).max()
    if scale == 0:
        scale = 1.0   # all nodes coincide; avoid division by zero
    coords = (coords - min_xy) / scale
    return dict(zip(pos.keys(), map(tuple, coords)))


def run_on_graph(G: nx.Graph, algorithm) -> dict:
    """Run one algorithm on one graph and return all metrics."""
    t0 = time.time()
    pos = algorithm.layout(G)
    elapsed = time.time() - t0

    pos_scaled = scale_to_unit_box(pos)  # scale for fair edge length comparison

    return {
        "crossings":         count_crossings(G, pos_scaled),
        "mean_edge_length":  mean_edge_length(G, pos_scaled),
        "edge_length_var":   edge_length_variance(G, pos_scaled),
        "path_continuity":   path_continuity(G, pos_scaled),
        "runtime_s":         round(elapsed, 2),
    }


def _process_graph(args):
    """
    Run all algorithms on one graph. Designed to run in a subprocess.
    Returns (graph_name, n_nodes, n_edges, rows, log_lines).
    """
    graph_name, tags_json, graph_path, algorithms = args

    G = nx.read_graphml(graph_path)
    G = nx.convert_node_labels_to_integers(G)

    rows = []
    log_lines = []

    for algorithm in algorithms:
        try:
            metrics = run_on_graph(G, algorithm)
            status = "ok"
            c  = metrics["crossings"]
            el = metrics["mean_edge_length"]
            pc = metrics["path_continuity"]
            log_lines.append(
                f"    {algorithm.name}: crossings={c}  "
                f"edge_len={el:.4f}  continuity={pc:.2f}°  ({metrics['runtime_s']}s)"
            )
        except Exception as exc:
            metrics = {k: None for k in METRIC_KEYS}
            status = f"error: {exc}"
            log_lines.append(f"    {algorithm.name}: ERROR {exc}")

        rows.append({
            "graph":     graph_name,
            "tags":      tags_json,
            "nodes":     G.number_of_nodes(),
            "edges":     G.number_of_edges(),
            "algorithm": algorithm.name,
            **metrics,
            "status":    status,
        })

    return graph_name, G.number_of_nodes(), G.number_of_edges(), rows, log_lines


# ============================================================================
# MAIN
# ============================================================================

def main(run_name: str = "experiment", n_workers: int = None):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    output_path = os.path.join(RESULTS_DIR, f"{run_name}.csv")

    metadata = pd.read_csv(METADATA_PATH)
    n_graphs = len(metadata)

    workers = n_workers if n_workers is not None else os.cpu_count()
    print(f"Dataset:    {n_graphs} graphs")
    print(f"Algorithms: {[a.name for a in ALGORITHMS]}")
    print(f"Workers:    {workers}  ({'parallel' if workers > 1 else 'sequential'})")
    print()

    tasks = [
        (
            row.name,
            row.tags,
            os.path.join(GRAPHS_DIR, f"{row.safe_name}.graphml"),
            ALGORITHMS,
        )
        for row in metadata.itertuples()
    ]

    rows = []

    if workers == 1:
        # Sequential mode — easier to debug; progress is printed in order
        for idx, task in enumerate(tasks, 1):
            graph_name, n_v, n_e, graph_rows, log = _process_graph(task)
            print(f"[{idx}/{n_graphs}] {graph_name} ({n_v}v, {n_e}e)")
            for line in log:
                print(line)
            rows.extend(graph_rows)
            pd.DataFrame(rows).to_csv(output_path, index=False)

    else:
        # Parallel mode — graphs processed concurrently; save after each completes
        done = 0
        with ProcessPoolExecutor(max_workers=workers) as executor:
            future_map = {executor.submit(_process_graph, t): t[0] for t in tasks}
            for future in as_completed(future_map):
                done += 1
                graph_name, n_v, n_e, graph_rows, log = future.result()
                print(f"[{done}/{n_graphs}] {graph_name} ({n_v}v, {n_e}e)")
                for line in log:
                    print(line)
                rows.extend(graph_rows)
                pd.DataFrame(rows).to_csv(output_path, index=False)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default="experiment",
                        help="Name for the output CSV (default: experiment)")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of parallel workers (default: all CPU cores)")
    args = parser.parse_args()
    main(run_name=args.run_name, n_workers=args.workers)
