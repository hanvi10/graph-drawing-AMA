"""
presentation/make_separation_table.py
======================================
The three separation methods bolted onto the same cf → crossing-repair layout,
ranked by crossing reduction, in the boxed style of the other tables.

  cf_cross_sep  — our minimal isotropic projection + crossing-reclaim polish
  cf + PRISM    — Gansner & Hu stress-based overlap removal
  cf + VPSC     — Dwyer/Marriott/Stuckey minimal-displacement overlap removal

Metric columns are mean per-graph % change vs the spectral+spring baseline over
the 75 Netzschleuder graphs (lower is better). Numbers are read live from
comparison_summary.csv.

HOW TO RUN:
    ./.venv/bin/python presentation/make_separation_table.py
OUTPUT:
    presentation/separation_methods_table.png
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
                        "separation_methods_table.png")

INK, GRID, WIN = "#222222", "#111111", "#2f5f9e"

# summary key -> (display, separation description)
ROWS = [
    ("cf+cross+sep", "cf_cross_sep",  "projection + reclaim"),
    ("cf+sep+prism", "cf + PRISM",    "stress-based"),
    ("cf+sep+vpsc",  "cf + VPSC",     "min-displacement"),
]
COLS = ["Algorithm", "Separation", "Edge crossings", "Edge length", "Path continuity"]


def pct(v):
    return f"{v:+.1f}%".replace("-", "−")


def main():
    df = pd.read_csv(SUMMARY).set_index("algorithm")
    rows = []
    for key, disp, sep in ROWS:
        r = df.loc[key]
        rows.append([disp, sep,
                     pct(r["crossings_pct_vs_baseline"]),
                     pct(r["mean_edge_length_pct_vs_baseline"]),
                     pct(r["path_continuity_pct_vs_baseline"])])

    fig, ax = plt.subplots(figsize=(13.5, 4.2))
    ax.set_axis_off()
    tbl = ax.table(cellText=rows, colLabels=COLS, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(15)
    tbl.scale(1.0, 3.2)

    col_w = [0.19, 0.27, 0.18, 0.16, 0.20]
    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor(GRID)
        cell.set_linewidth(1.4)
        cell.set_facecolor("white")
        cell.set_width(col_w[col])
        cell.get_text().set_color(INK)
        if row == 0:
            cell.get_text().set_fontsize(16)
        elif row == 1:                       # champion row
            cell.set_facecolor("#eaf0f7")
            if col == 0:
                cell.get_text().set_color(WIN)
                cell.get_text().set_fontweight("bold")

    fig.text(0.06, 0.12,
             "Same cf → crossing-repair layout; only the separation stage differs.  "
             "Mean per-graph % vs baseline, 75 graphs  ·  lower is better.",
             fontsize=12.5, color="#6b7075", ha="left", va="center")
    fig.text(0.06, 0.06,
             "VPSC's +456% is inflated by near-planar graphs (e.g. 2→278 crossings): "
             "the standard overlap removers destroy a crossing-optimised layout.",
             fontsize=11, color="#8a8f96", ha="left", va="center")

    fig.savefig(OUT_PATH, dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
