"""
algorithms2/sweep_min_dist.py
=============================
What does raising `min_dist_frac` buy, and what does it cost?

    ./.venv/bin/python -m algorithms2.sweep_min_dist
    ./.venv/bin/python -m algorithms2.sweep_min_dist --fracs 0,0.2,0.3,0.4 --limit 30

The separation step is pure post-processing: it never changes the ForceAtlas2
output except where the floor is breached. So the whole cost/benefit is visible
by re-running the same FA2 layouts through different floors — the force stage
runs once per graph and every floor reuses it, which is also why this sweep is
cheap.

Columns
-------
  cross/elen/cont   mean per-graph % change against the SAME drawing with no
                    floor (min_dist_frac=0), so the numbers isolate the
                    separation step rather than the switch to ForceAtlas2
  stacked           share of nodes whose nearest neighbour is inside 0.05k —
                    close enough to read as one blob. This is the number the
                    floor exists to drive to zero
  clumped           share inside the floor being tested, after separation
                    (should be 0: the floor held)
  moved             share of nodes the step displaced at all
  held              graphs where the floor was satisfied on the returned drawing
"""

import argparse
import os
import sys

import networkx as nx
import numpy as np

from algorithms2.baseline import Baseline
from algorithms2.separation import enforce_min_distance, separation_report
from metrics import count_crossings, mean_edge_length, path_continuity
from run_single import scale_to_unit_box

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GRAPHS_DIR = os.path.join(REPO, "data", "graphs")

# "stacked" = visually a single blob, whatever floor is being tested
STACKED_FRAC = 0.05


def metrics_of(G, pos):
    p = scale_to_unit_box(pos)
    return (count_crossings(G, p), mean_edge_length(G, p), path_continuity(G, p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fracs", default="0,0.1,0.2,0.25,0.3,0.35,0.4,0.5")
    ap.add_argument("--limit", type=int, default=None,
                    help="use only the N smallest graphs (this sweep is O(n^2) "
                         "in the continuity metric, so the full set is slow)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    fracs = [float(f) for f in args.fracs.split(",")]

    names = sorted(f[:-8] for f in os.listdir(GRAPHS_DIR) if f.endswith(".graphml"))
    graphs = []
    for name in names:
        G = nx.convert_node_labels_to_integers(
            nx.read_graphml(os.path.join(GRAPHS_DIR, f"{name}.graphml")))
        graphs.append((name, G))
    graphs.sort(key=lambda t: t[1].number_of_nodes())
    if args.limit:
        graphs = graphs[:args.limit]

    print(f"Sweeping {len(fracs)} floors over {len(graphs)} graphs "
          f"(ForceAtlas2 runs once per graph and is shared)\n")

    unsep, base_metrics = {}, {}
    for name, G in graphs:
        pos = Baseline(seed=args.seed, min_dist_frac=0.0).layout(G)
        unsep[name] = pos
        base_metrics[name] = metrics_of(G, pos)

    print(f"{'floor':>7}{'cross':>9}{'elen':>9}{'cont':>9}"
          f"{'stacked':>10}{'clumped':>9}{'moved':>8}{'held':>10}")
    print("-" * 71)
    for frac in fracs:
        dc, de, dp, stacked, clumped, moved, held = [], [], [], [], [], [], 0
        for name, G in graphs:
            raw = unsep[name]
            pos = (dict(raw) if frac <= 0
                   else enforce_min_distance(raw, frac, seed=args.seed))
            c, e, p = metrics_of(G, pos)
            bc, be, bp = base_metrics[name]
            if bc: dc.append(100 * (c - bc) / bc)
            if be: de.append(100 * (e - be) / be)
            if bp: dp.append(100 * (p - bp) / bp)

            stacked.append(separation_report(pos, STACKED_FRAC)["clumped_frac"])
            rep = separation_report(pos, frac) if frac > 0 else None
            clumped.append(rep["clumped_frac"] if rep else float("nan"))
            held += bool(rep["satisfied"]) if rep else 0
            delta = np.linalg.norm(
                np.array([pos[v] for v in G]) - np.array([raw[v] for v in G]), axis=1)
            moved.append(float((delta > 1e-9).mean()))

        held_str = f"{held}/{len(graphs)}" if frac > 0 else "—"
        clump_str = "—" if frac <= 0 else f"{np.nanmean(clumped):.1%}"
        print(f"{frac:>7.2f}{np.mean(dc):>+8.1f}%{np.mean(de):>+8.1f}%"
              f"{np.mean(dp):>+8.1f}%{np.mean(stacked):>10.1%}{clump_str:>9}"
              f"{np.mean(moved):>8.1%}{held_str:>10}")

    print("\n  % columns are vs the same ForceAtlas2 drawing with no floor.")
    print(f"  stacked = nodes with a neighbour inside {STACKED_FRAC}k "
          f"(unreadable overlap); the floor exists to drive this to 0.")


if __name__ == "__main__":
    sys.exit(main())
