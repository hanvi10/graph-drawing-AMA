"""
compare.py
==========
Compares each relaxation algorithm against the baseline the way the paper
does (Fig. 7 and Table 4): for every graph, the % change of the relaxed
layout's metric relative to the baseline's,

    % change = 100 * (relaxed - baseline) / baseline,

then the mean of those per-graph % changes over all graphs. Every metric is
"lower is better", so a negative % is an improvement.

The paper's mean is easily dominated by graphs with tiny baselines (1 -> 2
crossings is +100%, while 5000 -> 2500 is only -50%), so three robust
summaries are reported next to it, plus one absolute measure:
    - median % change: what the typical graph does.
    - win/tie/loss counts, like Fig. 7's upgrade/downgrade/no change
      (no change: |% change| < 0.05, i.e. 0.0% when rounded to one decimal).
    - geometric mean of ratios, as a %: 100 * (exp(mean(log(relaxed /
      baseline))) - 1). Averaging logs makes halving and doubling cancel
      (log 2 + log 0.5 = 0) instead of counting as +100% and -50%.
      Crossings are counts that can reach 0 (log 0 = -inf), so for crossings
      the ratio is (relaxed + 1) / (baseline + 1); the other metrics are
      never 0 and use the plain ratio.
    - change in crossings per edge: (relaxed - baseline crossings) / edges,
      averaged (mean and median) over graphs. Unlike the relative measures
      above, it weighs HOW MANY crossings were removed, scaled by graph size:
      100 -> 50 on 100 edges is -0.5, while 1 -> 2 on 300 edges is only
      +0.003. Very dense graphs (tens of crossings per edge) can still sway
      the mean, so the median is reported too.

INPUT (from run_single.py):
    data/results/baseline.csv
    data/results/edge_relaxation_ebc.csv
    data/results/edge_relaxation_currentflow.csv
    data/results/cf_cross_sep.csv

OUTPUT:
    data/results/comparison_summary.csv    -- one row per (algorithm, metric)
    data/results/comparison_per_graph.csv  -- one row per (algorithm, graph)

HOW TO RUN:
    python compare.py
"""

import os

import numpy as np
import pandas as pd

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "data", "results")

BASELINE = "baseline"
ALGORITHMS = {                      # display name -> results CSV (without .csv)
    "EBC":         "edge_relaxation_ebc",
    "currentflow": "edge_relaxation_currentflow",
    "cf_cross_sep": "cf_cross_sep",
}
METRICS = ["crossings", "mean_edge_length", "edge_length_var", "path_continuity"]

# The paper's Table 4 "All graphs" row (k_r = 0.1, k_s = 0.05); the variance
# figure is from Section 5.2. Shown for reference only: the paper's baseline
# is spectral + spring, ours is spectral + ForceAtlas2 + min separation.
PAPER = {"crossings": -16.8, "mean_edge_length": -13.1,
         "edge_length_var": 51.0, "path_continuity": 0.1}

NO_CHANGE = 0.05   # |% change| below this counts as "no change"
COUNT_METRICS = {"crossings"}   # can be 0: geometric mean uses +1 on both sides


def geometric_mean_pct(baseline: pd.Series, relaxed: pd.Series, metric: str) -> float:
    """Geometric mean of relaxed/baseline over graphs, as a % change."""
    if metric in COUNT_METRICS:
        baseline, relaxed = baseline + 1, relaxed + 1
    log_ratio = np.log(relaxed / baseline)
    return 100.0 * (np.exp(log_ratio.mean()) - 1.0)


def load(name: str) -> pd.DataFrame:
    df = pd.read_csv(os.path.join(RESULTS_DIR, f"{name}.csv"))
    bad = df[df["status"] != "ok"]
    if len(bad):
        print(f"  {name}: skipping {len(bad)} failed graph(s): {list(bad['graph'])}")
    return df[df["status"] == "ok"].set_index("graph")


def main():
    base = load(BASELINE)
    per_graph, summary = [], []

    for alg, csv in ALGORITHMS.items():
        relaxed = load(csv)
        graphs = base.index.intersection(relaxed.index)
        pct = pd.DataFrame(index=graphs)
        for m in METRICS:
            b, r = base.loc[graphs, m], relaxed.loc[graphs, m]
            pct[m] = (100.0 * (r - b) / b).where(b != 0)   # undefined when baseline is 0

            p = pct[m].dropna()
            excluded = sorted(set(graphs) - set(p.index))
            row = {
                "algorithm":     alg,
                "metric":        m,
                "graphs":        len(p),
                "mean_pct":      round(p.mean(), 2),
                "median_pct":    round(p.median(), 2),
                "geomean_pct":   round(geometric_mean_pct(b, r, m), 2),
                "upgrade":       int((p <= -NO_CHANGE).sum()),
                "downgrade":     int((p >= NO_CHANGE).sum()),
                "no_change":     int((p.abs() < NO_CHANGE).sum()),
                "paper_mean_pct": PAPER[m],
                "excluded":      "; ".join(excluded),
            }
            summary.append(row)
            if m == "crossings":
                crossing_row = row

        # Absolute measure: change in crossings per edge, per graph
        edges = base.loc[graphs, "edges"]
        delta_per_edge = (relaxed.loc[graphs, "crossings"] - base.loc[graphs, "crossings"]) / edges
        crossing_row["baseline_crossings_per_edge"] = round((base.loc[graphs, "crossings"] / edges).mean(), 4)
        crossing_row["mean_delta_per_edge"] = round(delta_per_edge.mean(), 4)
        crossing_row["median_delta_per_edge"] = round(delta_per_edge.median(), 4)

        rows = pct.add_suffix("_pct")
        rows["crossings_delta_per_edge"] = delta_per_edge
        for m in METRICS:
            rows[f"{m}_baseline"] = base.loc[graphs, m]
            rows[f"{m}_relaxed"] = relaxed.loc[graphs, m]
        rows.insert(0, "algorithm", alg)
        per_graph.append(rows.reset_index())

    summary = pd.DataFrame(summary)
    per_graph = pd.concat(per_graph, ignore_index=True)
    summary.to_csv(os.path.join(RESULTS_DIR, "comparison_summary.csv"), index=False)
    per_graph.to_csv(os.path.join(RESULTS_DIR, "comparison_per_graph.csv"), index=False)

    def show(title, column, paper=False):
        print(f"\n{title}\n")
        table = summary.pivot(index="algorithm", columns="metric", values=column)
        table = table.loc[list(ALGORITHMS), METRICS]   # keep ALGORITHMS order
        if paper:
            table.loc["paper (EBC, spring)"] = [PAPER[m] for m in METRICS]
        print(table.to_string(float_format=lambda v: f"{v:+.1f}%"))

    show("Mean % change vs baseline -- the paper's measure (negative = better)", "mean_pct", paper=True)
    show("Median % change vs baseline (the typical graph)", "median_pct")
    show("Geometric mean of relaxed/baseline ratios, as % change", "geomean_pct")

    print("\nChange in crossings per edge: (relaxed - baseline) / edges (negative = crossings removed)\n")
    print(f"  {'':12} {'mean':>8} {'median':>8}   (baseline average: crossings per edge)")
    for alg in ALGORITHMS:
        c = summary[(summary.algorithm == alg) & (summary.metric == "crossings")].iloc[0]
        print(f"  {alg:12} {c.mean_delta_per_edge:+8.3f} {c.median_delta_per_edge:+8.3f}   "
              f"({c.baseline_crossings_per_edge:.3f})")

    print("\nWin / loss / tie: graphs improved / worse / unchanged (Fig. 7)\n")
    for alg in ALGORITHMS:
        s = summary[summary.algorithm == alg].set_index("metric")
        print(f"  {alg:12} " + "   ".join(
            f"{m}: {s.loc[m, 'upgrade']}/{s.loc[m, 'downgrade']}/{s.loc[m, 'no_change']}"
            for m in METRICS))
    excl = summary[summary.excluded != ""]
    for _, r in excl.iterrows():
        print(f"  note: {r.algorithm} {r.metric}: excluded (baseline = 0): {r.excluded}")

    print(f"\nSaved: data/results/comparison_summary.csv, data/results/comparison_per_graph.csv")


if __name__ == "__main__":
    main()
