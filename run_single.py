"""
run_single.py
=============
Run a single graph drawing algorithm on all graphs in the dataset and save results.

Usage:
    python run_single.py --algorithm edge_relaxation_crossing
    python run_single.py --algorithm edge_relaxation_fiedler --run-name fiedler_v1
    python run_single.py --algorithm edge_relaxation_currentflow --workers 4

Available algorithms:
    edge_relaxation_crossing      (A1) crossing participation count
    edge_relaxation_stress        (A2) geometric stress score
    edge_relaxation_angular       (A3) inverse angular resolution
    edge_relaxation_currentflow   (B1) edge current flow betweenness
    edge_relaxation_embeddedness  (B2) inverse neighborhood overlap
    edge_relaxation_community     (C1) community membership distance
    edge_relaxation_fiedler       (C2) Fiedler vector partition score
    edge_relaxation_adaptive      (D1) adaptive crossing re-scoring
    community_collapse            (N1) collapse-expand community layout
    backbone_restore              (N2) delete bridges, layout, restore
    crossing_repair               (N3) crossing-guided node relocation
"""

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import networkx as nx
import numpy as np
import pandas as pd

from metrics import count_crossings, mean_edge_length, edge_length_variance, path_continuity

METRIC_KEYS = ["crossings", "mean_edge_length", "edge_length_var", "path_continuity", "runtime_s"]

METADATA_PATH = os.path.join(os.path.dirname(__file__), "data", "metadata.csv")
GRAPHS_DIR    = os.path.join(os.path.dirname(__file__), "data", "graphs")
RESULTS_DIR   = os.path.join(os.path.dirname(__file__), "data", "results")


# Registry: algorithm name -> (module path, class name)
ALGORITHM_REGISTRY = {
    "edge_relaxation_crossing":     ("algorithms.edge_relaxation.crossing",     "EdgeRelaxationCrossing"),
    "edge_relaxation_stress":       ("algorithms.edge_relaxation.stress",       "EdgeRelaxationStress"),
    "edge_relaxation_angular":      ("algorithms.edge_relaxation.angular",      "EdgeRelaxationAngular"),
    "edge_relaxation_currentflow":  ("algorithms.edge_relaxation.currentflow",  "EdgeRelaxationCurrentFlow"),
    "edge_relaxation_embeddedness": ("algorithms.edge_relaxation.embeddedness", "EdgeRelaxationEmbeddedness"),
    "edge_relaxation_community":    ("algorithms.edge_relaxation.community",    "EdgeRelaxationCommunity"),
    "edge_relaxation_fiedler":      ("algorithms.edge_relaxation.fiedler",      "EdgeRelaxationFiedler"),
    "edge_relaxation_adaptive":     ("algorithms.edge_relaxation.adaptive",     "EdgeRelaxationAdaptive"),
    "community_collapse":           ("algorithms.novel.community_collapse",     "CommunityCollapse"),
    "backbone_restore":             ("algorithms.novel.backbone_restore",       "BackboneRestore"),
    "crossing_repair":              ("algorithms.novel.crossing_repair",        "CrossingRepair"),
    "collapse_repair":              ("algorithms.novel.collapse_repair",        "CollapseRepair"),
    "currentflow_repair":           ("algorithms.novel.currentflow_repair",     "CurrentFlowRepair"),
    "currentflow_repair_expand":    ("algorithms.novel.currentflow_repair_expand", "CurrentFlowRepairExpand"),
    "aesthetic_repair":             ("algorithms.novel.aesthetic_repair",       "AestheticRepair"),
    "currentflow_aesthetic":        ("algorithms.novel.currentflow_aesthetic",  "CurrentFlowAesthetic"),
    "cf_cross_sep":                 ("algorithms.novel.cf_cross_sep",           "CFCrossSep"),
    "cf_cross_sep_min":             ("algorithms.novel.cf_cross_sep_min",       "CFCrossSepMin"),
    "cf_cross_sep_prism":           ("algorithms.novel.cf_cross_sep_prism",     "CFCrossSepPrism"),
    "cf_cross_sep_vpsc":            ("algorithms.novel.cf_cross_sep_vpsc",      "CFCrossSepVpsc"),
    "cross_sep":                    ("algorithms.novel.cross_sep",              "CrossSep"),
}


def _get_algorithm(name):
    if name not in ALGORITHM_REGISTRY:
        raise ValueError(f"Unknown algorithm: {name}")
    module_path, class_name = ALGORITHM_REGISTRY[name]
    import importlib
    module = importlib.import_module(module_path)
    return getattr(module, class_name)()


ALGORITHM_NAMES = list(ALGORITHM_REGISTRY)


def scale_to_unit_box(pos: dict) -> dict:
    coords = np.array(list(pos.values()))
    min_xy = coords.min(axis=0)
    scale  = (coords.max(axis=0) - min_xy).max()
    if scale == 0:
        scale = 1.0
    coords = (coords - min_xy) / scale
    return dict(zip(pos.keys(), map(tuple, coords)))


def _process_graph(args):
    graph_name, tags, graph_path, algorithm_name = args
    algorithm = _get_algorithm(algorithm_name)

    G = nx.read_graphml(graph_path)
    G = nx.convert_node_labels_to_integers(G)

    try:
        t0 = time.time()
        pos = algorithm.layout(G)
        elapsed = time.time() - t0

        pos_scaled = scale_to_unit_box(pos)
        metrics = {
            "crossings":        count_crossings(G, pos_scaled),
            "mean_edge_length": mean_edge_length(G, pos_scaled),
            "edge_length_var":  edge_length_variance(G, pos_scaled),
            "path_continuity":  path_continuity(G, pos_scaled),
            "runtime_s":        round(elapsed, 2),
        }
        status = "ok"
        log = (f"    crossings={metrics['crossings']}  "
               f"edge_len={metrics['mean_edge_length']:.4f}  "
               f"continuity={metrics['path_continuity']:.2f}°  "
               f"({metrics['runtime_s']}s)")
    except Exception as exc:
        metrics = {k: None for k in METRIC_KEYS}
        status = f"error: {exc}"
        log = f"    ERROR: {exc}"

    row = {
        "graph":     graph_name,
        "tags":      tags,
        "nodes":     G.number_of_nodes(),
        "edges":     G.number_of_edges(),
        "algorithm": algorithm.name,
        **metrics,
        "status":    status,
    }
    return graph_name, G.number_of_nodes(), G.number_of_edges(), row, log


def main():
    parser = argparse.ArgumentParser(description="Run one graph drawing algorithm on the full dataset.")
    parser.add_argument("--algorithm", required=True, choices=ALGORITHM_NAMES,
                        help="Algorithm to run")
    parser.add_argument("--run-name", default=None,
                        help="Output CSV filename (default: same as --algorithm)")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel workers (default: 1 = sequential)")
    parser.add_argument("--metadata", default=METADATA_PATH,
                        help="Metadata CSV to use (default: full dataset)")
    args = parser.parse_args()

    run_name = args.run_name or args.algorithm
    os.makedirs(RESULTS_DIR, exist_ok=True)
    output_path = os.path.join(RESULTS_DIR, f"{run_name}.csv")

    metadata = pd.read_csv(args.metadata)
    n_graphs = len(metadata)

    sample_alg = _get_algorithm(args.algorithm)
    print(f"Algorithm:  {sample_alg.name}")
    print(f"Dataset:    {n_graphs} graphs")
    print(f"Workers:    {args.workers}  ({'parallel' if args.workers > 1 else 'sequential'})")
    print(f"Output:     {output_path}")
    print()

    tasks = [
        (row.name, row.tags, os.path.join(GRAPHS_DIR, f"{row.safe_name}.graphml"), args.algorithm)
        for row in metadata.itertuples()
    ]

    rows = []

    if args.workers == 1:
        for idx, task in enumerate(tasks, 1):
            graph_name, n_v, n_e, row, log = _process_graph(task)
            print(f"[{idx}/{n_graphs}] {graph_name} ({n_v}v, {n_e}e)")
            print(log)
            rows.append(row)
            pd.DataFrame(rows).to_csv(output_path, index=False)
    else:
        done = 0
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            future_map = {executor.submit(_process_graph, t): t[0] for t in tasks}
            for future in as_completed(future_map):
                done += 1
                graph_name, n_v, n_e, row, log = future.result()
                print(f"[{done}/{n_graphs}] {graph_name} ({n_v}v, {n_e}e)")
                print(log)
                rows.append(row)
                pd.DataFrame(rows).to_csv(output_path, index=False)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
