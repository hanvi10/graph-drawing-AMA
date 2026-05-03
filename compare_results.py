"""
compare_results.py
==================
Analyses the experiment CSV and compares our results with the paper's
reported numbers (Table 4, Figure 7).

HOW TO RUN:
    python3 compare_results.py
"""

import os
import pandas as pd
import numpy as np

RESULTS_PATH = os.path.join(os.path.dirname(__file__), "data", "results", "experiment.csv")

# Paper's reported averages (Table 4, k_r=0.1, k_s=0.05, all graphs)
PAPER = {
    "crossings_reduction":    -16.8,   # %
    "edge_length_reduction":  -13.1,   # %
    "continuity_change":      +0.1,    # % (negligible)
}


def main():
    df = pd.read_csv(RESULTS_PATH)

    # Pivot so each graph has one row with baseline and relaxation side-by-side
    baseline = df[df["algorithm"] == "baseline"].set_index("graph")
    relaxed  = df[df["algorithm"].str.startswith("edge_relaxation")].set_index("graph")

    # Only keep graphs present in both
    common = baseline.index.intersection(relaxed.index)
    baseline = baseline.loc[common]
    relaxed  = relaxed.loc[common]

    print(f"Graphs analysed: {len(common)}")
    print()

    # ── Per-graph percentage changes ────────────────────────────────────────
    cross_pct = 100 * (relaxed["crossings"]        - baseline["crossings"])        / baseline["crossings"]
    elen_pct  = 100 * (relaxed["mean_edge_length"] - baseline["mean_edge_length"]) / baseline["mean_edge_length"]
    cont_pct  = 100 * (relaxed["path_continuity"]  - baseline["path_continuity"])  / baseline["path_continuity"]

    # ── Summary table ───────────────────────────────────────────────────────
    print("=" * 60)
    print(f"{'Metric':<30} {'Ours':>10} {'Paper':>10}")
    print("=" * 60)
    print(f"{'Crossings change (mean %)' :<30} {cross_pct.mean():>+10.1f} {PAPER['crossings_reduction']:>+10.1f}")
    print(f"{'Edge length change (mean %)':<30} {elen_pct.mean():>+10.1f} {PAPER['edge_length_reduction']:>+10.1f}")
    print(f"{'Path continuity change (%)':<30} {cont_pct.mean():>+10.1f} {PAPER['continuity_change']:>+10.1f}")
    print("=" * 60)
    print()

    # ── Improvement rate ────────────────────────────────────────────────────
    n_improved = (cross_pct < 0).sum()
    n_worsened = (cross_pct > 0).sum()
    n_same     = (cross_pct == 0).sum()
    print(f"Graphs improved: {n_improved}/{len(common)}  "
          f"worsened: {n_worsened}  unchanged: {n_same}")
    print(f"(Paper: improved {len(common)-4}/{len(common)}, 4 worsened)")
    print()

    # ── Per-graph breakdown ─────────────────────────────────────────────────
    print("Per-graph results (sorted by crossing reduction):")
    print(f"{'Graph':<45} {'Before':>7} {'After':>7} {'Δ%':>7}")
    print("-" * 70)
    details = pd.DataFrame({
        "before": baseline["crossings"],
        "after":  relaxed["crossings"],
        "pct":    cross_pct,
    }).sort_values("pct")

    for graph, row in details.iterrows():
        marker = " ✓" if row["pct"] < 0 else (" ✗" if row["pct"] > 0 else "")
        print(f"{graph:<45} {int(row['before']):>7} {int(row['after']):>7} {row['pct']:>+7.1f}%{marker}")


if __name__ == "__main__":
    main()
