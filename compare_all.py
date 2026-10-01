"""
compare_all.py
==============
Comprehensive comparison of all graph drawing algorithms.

Produces two CSV files:
  data/results/comparison_summary.csv   — one row per algorithm,
      mean metric values + % change vs baseline + % change vs EBC.

  data/results/comparison_per_graph.csv — one row per (graph, algorithm),
      absolute values + % change vs baseline + % change vs EBC for every graph.

HOW TO RUN:
    python3 compare_all.py
"""

import os
import pandas as pd
import numpy as np

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "data", "results")

METRICS = ["crossings", "mean_edge_length", "path_continuity"]

# Map display name -> CSV file (or None to load from experiment.csv)
ALGORITHMS = {
    "baseline":     ("experiment.csv",                      "baseline"),
    "EBC":          ("experiment.csv",                      "edge_relaxation(kr=0.1,kw=0.05)"),
    "crossing":     ("edge_relaxation_crossing.csv",        None),
    "fiedler":      ("edge_relaxation_fiedler.csv",         None),
    "community":    ("edge_relaxation_community.csv",       None),
    "embeddedness": ("edge_relaxation_embeddedness.csv",    None),
    "stress":       ("edge_relaxation_stress.csv",          None),
    "angular":      ("edge_relaxation_angular.csv",         None),
    "adaptive":     ("edge_relaxation_adaptive.csv",        None),
    "currentflow":  ("edge_relaxation_currentflow.csv",     None),
    # Novel (non-relaxation) algorithms
    "collapse":     ("community_collapse.csv",              None),
    "backbone":     ("backbone_restore.csv",                None),
    "repair":       ("crossing_repair.csv",                 None),
    "collapse+rep": ("collapse_repair.csv",                 None),
    "cflow+rep":    ("currentflow_repair.csv",              None),
    "cf+rep+exp":   ("currentflow_repair_expand.csv",       None),
    "cf+aesthetic": ("currentflow_aesthetic.csv",           None),
    "cf+cross+sep": ("cf_cross_sep.csv",                    None),
    "cf+sep+prism": ("cf_cross_sep_prism.csv",             None),
    "cf+sep+vpsc":  ("cf_cross_sep_vpsc.csv",              None),
}


def load_all() -> dict[str, pd.DataFrame]:
    """Load each algorithm's results into a dict keyed by display name."""
    frames = {}
    for name, (filename, algo_filter) in ALGORITHMS.items():
        path = os.path.join(RESULTS_DIR, filename)
        df = pd.read_csv(path)
        if algo_filter is not None:
            df = df[df["algorithm"] == algo_filter]
        if "status" in df.columns:
            df = df[df["status"] == "ok"]
        frames[name] = df.set_index("graph")
    return frames


def per_graph_pct(algo_series: pd.Series, ref_series: pd.Series) -> float:
    """Mean of per-graph % changes — matches how the paper reports results."""
    common = algo_series.index.intersection(ref_series.index)
    a = algo_series.loc[common]
    r = ref_series.loc[common]
    return (100.0 * (a - r) / r.replace(0, np.nan)).mean()


def build_summary(frames: dict) -> pd.DataFrame:
    """One row per algorithm: mean metric values + % vs baseline + % vs EBC."""
    base = frames["baseline"]
    ebc  = frames["EBC"]

    rows = []
    for name, df in frames.items():
        common_base = df.index.intersection(base.index)
        common_ebc  = df.index.intersection(ebc.index)

        row = {"algorithm": name}

        # Mean absolute values
        for m in METRICS:
            row[f"mean_{m}"] = df.loc[common_base, m].mean()

        row["mean_runtime_s"] = df["runtime_s"].mean() if "runtime_s" in df.columns else np.nan

        # % vs baseline — mean of per-graph % changes (same method as paper)
        for m in METRICS:
            row[f"{m}_pct_vs_baseline"] = per_graph_pct(df[m], base[m])

        # % vs EBC — mean of per-graph % changes
        for m in METRICS:
            row[f"{m}_pct_vs_ebc"] = per_graph_pct(df[m], ebc[m])

        # Head-to-head win/tie/loss vs EBC on crossings
        idx = df.index.intersection(ebc.index)
        algo_c = df.loc[idx, "crossings"]
        ebc_c  = ebc.loc[idx, "crossings"]
        row["wins_vs_ebc"]   = int((algo_c < ebc_c).sum())
        row["ties_vs_ebc"]   = int((algo_c == ebc_c).sum())
        row["losses_vs_ebc"] = int((algo_c > ebc_c).sum())
        row["n_graphs"]      = len(idx)

        rows.append(row)

    return pd.DataFrame(rows).set_index("algorithm")


def build_per_graph(frames: dict) -> pd.DataFrame:
    """One row per (graph, algorithm): absolute values + % vs baseline + % vs EBC."""
    base = frames["baseline"]
    ebc  = frames["EBC"]

    rows = []
    for name, df in frames.items():
        for graph in df.index:
            row = {"graph": graph, "algorithm": name}

            for m in METRICS:
                val = df.loc[graph, m]
                row[m] = val

                # % vs baseline (per-graph, matches paper's reporting method)
                if graph in base.index:
                    ref = base.loc[graph, m]
                    row[f"{m}_pct_vs_baseline"] = 100.0 * (val - ref) / ref if ref != 0 else np.nan
                else:
                    row[f"{m}_pct_vs_baseline"] = np.nan

                # % vs EBC (per-graph)
                if graph in ebc.index:
                    ref = ebc.loc[graph, m]
                    row[f"{m}_pct_vs_ebc"] = 100.0 * (val - ref) / ref if ref != 0 else np.nan
                else:
                    row[f"{m}_pct_vs_ebc"] = np.nan

            if "runtime_s" in df.columns:
                row["runtime_s"] = df.loc[graph, "runtime_s"]

            rows.append(row)

    return pd.DataFrame(rows)


def print_summary(summary: pd.DataFrame):
    algos = [a for a in summary.index if a != "baseline"]

    W = 16 + 3 * 12 + 3 * 10 + 3 * 10 + 12 + 4
    sep = "=" * W

    # Header spanning all column groups
    print()
    print(sep)
    print(
        f"{'':16}"
        f"{'── Absolute values ──':>36}"
        f"{'── % vs Baseline ──':>32}"
        f"{'── % vs EBC ──':>32}"
        f"{'':>12}"
    )
    print(
        f"{'Algorithm':<16}"
        f"{'Crossings':>12}{'Edge Len':>12}{'Path Cont':>12}"
        f"{'Cross%':>10}{'ELen%':>10}{'PCont%':>10}"
        f"{'Cross%':>10}{'ELen%':>10}{'PCont%':>10}"
        f"{'Runtime(s)':>12}"
    )
    print("-" * W)

    # Baseline row (no % change columns — it is the reference)
    r = summary.loc["baseline"]
    print(
        f"{'baseline':<16}"
        f"{r['mean_crossings']:>12.1f}{r['mean_mean_edge_length']:>12.4f}{r['mean_path_continuity']:>12.2f}"
        f"{'—':>10}{'—':>10}{'—':>10}"
        f"{'—':>10}{'—':>10}{'—':>10}"
        f"{r['mean_runtime_s']:>12.2f}"
    )
    print("-" * W)

    for algo in algos:
        r = summary.loc[algo]
        ebc_str = "—" if algo == "EBC" else f"{r['crossings_pct_vs_ebc']:>+.1f}%"
        elen_ebc = "—" if algo == "EBC" else f"{r['mean_edge_length_pct_vs_ebc']:>+.1f}%"
        pc_ebc   = "—" if algo == "EBC" else f"{r['path_continuity_pct_vs_ebc']:>+.1f}%"
        print(
            f"{algo:<16}"
            f"{r['mean_crossings']:>12.1f}{r['mean_mean_edge_length']:>12.4f}{r['mean_path_continuity']:>12.2f}"
            f"{r['crossings_pct_vs_baseline']:>+10.1f}%{r['mean_edge_length_pct_vs_baseline']:>+10.1f}%{r['path_continuity_pct_vs_baseline']:>+10.1f}%"
            f"{ebc_str:>10}{elen_ebc:>10}{pc_ebc:>10}"
            f"{r['mean_runtime_s']:>12.2f}"
        )

    print(sep)
    print("  % vs Baseline: mean of per-graph % changes (same method as the paper).")
    print("  % vs EBC:      negative = beats the paper's edge betweenness method.")
    print()


def main():
    frames = load_all()

    summary = build_summary(frames)
    per_graph = build_per_graph(frames)

    out_summary   = os.path.join(RESULTS_DIR, "comparison_summary.csv")
    out_per_graph = os.path.join(RESULTS_DIR, "comparison_per_graph.csv")

    summary.to_csv(out_summary)
    per_graph.to_csv(out_per_graph, index=False)

    print_summary(summary)

    print(f"Saved: {out_summary}")
    print(f"Saved: {out_per_graph}")


if __name__ == "__main__":
    main()
