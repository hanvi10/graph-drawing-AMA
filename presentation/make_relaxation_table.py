"""
presentation/make_relaxation_table.py
======================================
Summary table of every edge-relaxation SCORING method we tried, ranked best to
worst by crossing reduction, in the same boxed style as the other presentation
tables (ebc_vs_currentflow_table.png etc.).

Columns: Algorithm | Edge crossings | Edge length | Path continuity |
         Time complexity

The three metric columns are the mean per-graph % change vs. the spectral+spring
baseline over the 75 Netzschleuder graphs (lower is better on all three). The
"time complexity" column is the one-time cost of computing each method's edge
scores (V = nodes, E = edges, d = mean degree); every method then shares the
same iterative spring + crossing-count relaxation loop on top, so that column is
what actually distinguishes them.

HOW TO RUN:
    ./.venv/bin/python presentation/make_relaxation_table.py

OUTPUT:
    presentation/relaxation_methods_table.png
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams["font.family"] = "serif"

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUMMARY = os.path.join(ROOT, "data", "results", "comparison_summary.csv")
OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "relaxation_methods_table.png")

INK = "#222222"
GRID = "#111111"
NODE_COLOR = "#2f5f9e"

# CSV key -> (display name, edge-scoring time complexity)
METHODS = [
    ("currentflow",  "Current-Flow",  r"$O(V^{3})$"),
    ("EBC",          "EBC (paper)",   r"$O(V\cdot E)$"),
    ("community",    "Community",     r"$O(V\cdot E)$"),
    ("stress",       "Stress",        r"$O(E)$"),
    ("fiedler",      "Fiedler",       r"$O(V^{3})$"),
    ("embeddedness", "Embeddedness",  r"$O(E\cdot \bar{d})$"),
    ("angular",      "Angular",       r"$O(E\cdot \bar{d})$"),
    ("adaptive",     "Adaptive",      r"$O(E^{2})$"),
    ("crossing",     "Crossing-count", r"$O(E^{2})$"),
]

COLS = ["Algorithm", "Edge crossings", "Edge length",
        "Path continuity", "Time complexity"]


def pct(v):
    """Signed percent with a typographic minus sign."""
    return f"{v:+.1f}%".replace("-", "−")


def main():
    df = pd.read_csv(SUMMARY).set_index("algorithm")
    # already ranked best -> worst in METHODS (by crossing reduction)

    rows = []
    for key, disp, cxty in METHODS:
        r = df.loc[key]
        rows.append([
            disp,
            pct(r["crossings_pct_vs_baseline"]),
            pct(r["mean_edge_length_pct_vs_baseline"]),
            pct(r["path_continuity_pct_vs_baseline"]),
            cxty,
        ])

    n_rows = len(rows)
    fig, ax = plt.subplots(figsize=(13.5, 8.2))
    ax.set_axis_off()

    tbl = ax.table(cellText=rows, colLabels=COLS, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(15)
    tbl.scale(1.0, 3.0)

    col_w = [0.24, 0.19, 0.16, 0.19, 0.22]
    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor(GRID)
        cell.set_linewidth(1.4)
        cell.set_facecolor("white")
        cell.set_width(col_w[col])
        txt = cell.get_text()
        txt.set_color(INK)
        if row == 0:                                   # header
            txt.set_fontsize(16)
        elif row == 1:                                 # champion row: highlight
            cell.set_facecolor("#eaf0f7")
            if col == 0:
                txt.set_color(NODE_COLOR)
                txt.set_fontweight("bold")

    # captions (bottom-left, gray) — same voice as the other tables
    fig.text(0.055, 0.075,
             "Mean per-graph % change vs. spectral+spring baseline, "
             "75 Netzschleuder graphs  •  lower is better  •  "
             "ranked by crossing reduction",
             fontsize=12.5, color="#6b7075", ha="left", va="center")
    fig.text(0.055, 0.038,
             "Time complexity = one-time edge-scoring cost "
             "(V nodes, E edges, d̄ mean degree); all methods then share the "
             "same iterative spring + crossing-count loop.",
             fontsize=11, color="#8a8f96", ha="left", va="center")

    fig.savefig(OUT_PATH, dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUT_PATH}  ({n_rows} methods)")


if __name__ == "__main__":
    main()
