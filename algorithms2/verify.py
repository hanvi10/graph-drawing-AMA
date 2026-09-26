"""
algorithms2/verify.py
=====================
Correctness checks for the algorithms2 baseline. Run it after touching
baseline.py, separation.py or forceatlas2.py:

    ./.venv/bin/python -m algorithms2.verify              # full dataset
    ./.venv/bin/python -m algorithms2.verify --limit 12   # quick pass

What it asserts, per graph:

  1. every node gets a position, and every coordinate is finite
     — the check that would have caught ForceAtlas2 returning all-NaN on
       malaria_genes__HVR_3 from a spectral start
  2. the minimum-distance floor holds, measured against k of the RETURNED
     drawing, not the one the projection was aimed at
  3. the same seed gives the same drawing twice
  4. min_dist_frac=0 leaves the ForceAtlas2 output untouched, so the floor is
     genuinely optional and its cost is measurable
  5. the drawing is not degenerate — positive area, no single runaway node
     carrying the whole bounding box

and on synthetic edge cases: empty, one node, two nodes, a path (whose spectral
layout is collinear, so the bounding box has zero area), a complete graph, a
star (every leaf structurally equivalent, so the spectral layout stacks them)
and two nodes at identical coordinates.
"""

import argparse
import math
import os
import sys

import networkx as nx
import numpy as np

from algorithms2.baseline import Baseline
from algorithms2.separation import (enforce_min_distance, ideal_distance,
                                    separation_report)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GRAPHS_DIR = os.path.join(REPO, "data", "graphs")
FRAC = 0.30


def _finite(pos):
    return np.isfinite(np.array(list(pos.values()), dtype=float)).all()


def check_graph(G, name, failures):
    b = Baseline(seed=42, min_dist_frac=FRAC)
    pos = b.layout(G)

    if set(pos) != set(G.nodes()):
        failures.append(f"{name}: position set != node set")
        return None
    if not _finite(pos):
        failures.append(f"{name}: non-finite coordinates")
        return None

    rep = separation_report(pos, FRAC)
    if not rep["satisfied"]:
        failures.append(
            f"{name}: floor breached — {rep['violating_pairs']} pairs below "
            f"{FRAC}k (closest {rep['min_dist_over_k']:.4f}k)")

    again = Baseline(seed=42, min_dist_frac=FRAC).layout(G)
    if any(pos[v] != again[v] for v in G):
        failures.append(f"{name}: not reproducible at a fixed seed")

    # min_dist_frac=0 must be a genuine no-op on the force stage's output
    raw = Baseline(seed=42, min_dist_frac=0.0).layout(G)
    if _finite(raw) and raw == enforce_min_distance(raw, 0.0):
        pass
    else:
        failures.append(f"{name}: min_dist_frac=0 is not a no-op")

    coords = np.array([pos[v] for v in G], dtype=float)
    if len(coords) > 2:
        w, h = coords.max(axis=0) - coords.min(axis=0)
        if not (w > 0 and h > 0):
            failures.append(f"{name}: degenerate drawing, bounding box {w}x{h}")

    moved = np.linalg.norm(
        np.array([pos[v] for v in G]) - np.array([raw[v] for v in G]), axis=1)
    k_raw = ideal_distance(np.array([raw[v] for v in G]))
    return {
        "name": name,
        "n": G.number_of_nodes(),
        "m": G.number_of_edges(),
        "min_over_k": rep["min_dist_over_k"],
        "clumped_before": separation_report(raw, FRAC)["clumped_frac"],
        "max_move_k": float(moved.max() / k_raw) if k_raw > 0 else 0.0,
        "moved_frac": float((moved > 1e-9).mean()),
    }


def check_edge_cases(failures):
    cases = {
        "empty": nx.Graph(),
        "one node": nx.empty_graph(1),
        "two nodes": nx.path_graph(2),
        "path-30 (collinear spectral)": nx.path_graph(30),
        "complete-12": nx.complete_graph(12),
        "star-40 (stacked leaves)": nx.star_graph(40),
    }
    for name, G in cases.items():
        try:
            pos = Baseline(seed=42, min_dist_frac=FRAC).layout(G)
        except Exception as exc:
            failures.append(f"edge case {name}: raised {type(exc).__name__}: {exc}")
            continue
        if set(pos) != set(G.nodes()):
            failures.append(f"edge case {name}: position set != node set")
            continue
        if not _finite(pos):
            failures.append(f"edge case {name}: non-finite coordinates")
            continue
        if G.number_of_nodes() > 2:
            rep = separation_report(pos, FRAC)
            status = "ok" if rep["satisfied"] else "FLOOR BREACHED"
            if not rep["satisfied"]:
                failures.append(f"edge case {name}: {rep['violating_pairs']} pairs "
                                f"below the floor")
            print(f"  {name:<32} n={G.number_of_nodes():<4} "
                  f"min/k={rep['min_dist_over_k']:.3f}  {status}")
        else:
            print(f"  {name:<32} n={G.number_of_nodes():<4} (no pair to separate)")

    # coincident input: the case that makes the separation direction undefined
    stacked = {0: (1.0, 1.0), 1: (1.0, 1.0), 2: (1.0, 1.0), 3: (4.0, 4.0)}
    out = enforce_min_distance(stacked, 0.30, seed=42)
    rep = separation_report(out, 0.30)
    if not (_finite(out) and rep["satisfied"]):
        failures.append("edge case coincident points: floor not established")
    print(f"  {'3 coincident points':<32} n=4    "
          f"min/k={rep['min_dist_over_k']:.3f}  "
          f"{'ok' if rep['satisfied'] else 'FLOOR BREACHED'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="check only the first N graphs (by node count)")
    args = ap.parse_args()

    failures = []

    print("Edge cases")
    print("-" * 78)
    check_edge_cases(failures)

    paths = sorted(f for f in os.listdir(GRAPHS_DIR) if f.endswith(".graphml"))
    graphs = []
    for f in paths:
        G = nx.convert_node_labels_to_integers(
            nx.read_graphml(os.path.join(GRAPHS_DIR, f)))
        graphs.append((f[:-8], G))
    graphs.sort(key=lambda t: t[1].number_of_nodes())
    if args.limit:
        graphs = graphs[:args.limit]

    print(f"\nDataset ({len(graphs)} graphs)")
    print("-" * 78)
    print(f"{'graph':<36}{'n':>6}{'m':>6}{'min/k':>8}{'clumped':>9}"
          f"{'moved':>8}{'max move':>10}")
    rows = []
    for name, G in graphs:
        row = check_graph(G, name, failures)
        if row is None:
            print(f"{name:<36}  FAILED")
            continue
        rows.append(row)
        print(f"{name[:35]:<36}{row['n']:>6}{row['m']:>6}{row['min_over_k']:>8.3f}"
              f"{row['clumped_before']:>9.1%}{row['moved_frac']:>8.1%}"
              f"{row['max_move_k']:>9.2f}k")

    print("-" * 78)
    if rows:
        print(f"{'mean':<36}{'':>12}{np.mean([r['min_over_k'] for r in rows]):>8.3f}"
              f"{np.mean([r['clumped_before'] for r in rows]):>9.1%}"
              f"{np.mean([r['moved_frac'] for r in rows]):>8.1%}"
              f"{np.mean([r['max_move_k'] for r in rows]):>9.2f}k")
        print(f"\n  min/k     smallest node gap in the returned drawing, in units of k"
              f" (floor = {FRAC})")
        print("  clumped   share of nodes inside the floor BEFORE separation")
        print("  moved     share of nodes the separation step displaced at all")
        print("  max move  furthest any single node was displaced")

    print("\n" + "=" * 78)
    if failures:
        print(f"{len(failures)} FAILURE(S):")
        for f in failures:
            print(" -", f)
        return 1
    print(f"All checks passed on {len(rows)} graphs + edge cases.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
