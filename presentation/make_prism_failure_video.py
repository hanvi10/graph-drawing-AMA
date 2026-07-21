"""
presentation/make_prism_failure_video.py
=========================================
Why a standard literature overlap remover (PRISM, Gansner & Hu, "Efficient Node
Overlap Removal", 2009) degrades a near-planar layout — the PRISM counterpart of
make_vpsc_failure_video.py.

Starts from the exact layout cf_cross_sep feeds into its separation stage:
current-flow relaxation → crossing repair. On `revolution` (141 nodes) that
layout is near-planar — only 4 crossings — but crossing repair has packed the
graph so hard that 134 of the 141 nodes are stacked (the layout collapses to 33
distinct positions). Separation is unavoidable; the question is HOW.

PRISM removes overlap by a stress objective: build a Delaunay proximity graph,
grow only the overlapping edges to the floor, and re-settle by SMACOF stress
majorisation, then re-triangulate and repeat. That objective matches proximity
distances, not crossings — so re-settling drags edges across each other and the
counter climbs 4 → 15. The right theory (minimal proximity distortion) with the
wrong objective for radial graph structure — the same evidence VPSC gives, that
cf_cross_sep's crossing-guarded, isotropic projection is what preserves the
low-crossing layout crossing repair built.

Faithful to algorithms/novel/cf_cross_sep_prism.py._prism: same Delaunay
proximity edges, same grow-to-sep + SMACOF majorisation, same floor sep_frac·k
recomputed each round, same node-edge projection cleanup. The first round's
SMACOF iterations are shown individually so the stress inflation is visible;
later rounds and the cleanup are morphed to the true algorithm output.

HOW TO RUN:
    ./.venv/bin/python presentation/make_prism_failure_video.py            # revolution
    ./.venv/bin/python presentation/make_prism_failure_video.py --graph mist/ppi_zebrafish

OUTPUT:
    presentation/videos/prism_failure_<graph>.gif
"""

import argparse
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.animation import PillowWriter
from matplotlib.collections import LineCollection
from scipy.spatial import Delaunay, QhullError

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import layout_cache
from algorithms.baseline import Baseline
from algorithms.edge_relaxation.currentflow import EdgeRelaxationCurrentFlow
from algorithms.novel.cf_cross_sep_prism import CFCrossSepPrism
from algorithms.novel.cf_cross_sep import _project_floors, _floor_violations
from algorithms.novel.crossing_repair import _strict_cross

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "videos")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

NODE_COLOR = "#2f5f9e"
EDGE_GRAY = "#9aa5b1"
CROSS_RED = "#c0392b"
STACK_ORANGE = "#e08a1e"
INK = "#333333"
SEED = 42

FPS = 30
HOLD_START = 55       # sit on the near-planar input, highlight the stacked nodes
MORPH_SMACOF = 24     # one shown SMACOF re-settle — slow, this is the mechanism
SETTLE_SMACOF = 6
MORPH_ROUND = 22      # a later whole PRISM round (re-triangulate + re-settle)
MORPH_CLEAN = 24      # node-edge cleanup to the true algorithm output
HOLD_END = 70


# ── geometry for drawing crossings (verbatim from make_vpsc_failure_video) ───
def seg_point(p1, p2, q1, q2):
    r, s = p2 - p1, q2 - q1
    rxs = r[0] * s[1] - r[1] * s[0]
    if abs(rxs) < 1e-12:
        return None
    t = ((q1 - p1)[0] * s[1] - (q1 - p1)[1] * s[0]) / rxs
    return p1 + t * r


def crossing_points(coords, E):
    pts = []
    for a in range(len(E)):
        e1 = E[a]
        for b in range(a + 1, len(E)):
            e2 = E[b]
            if len(set(e1) | set(e2)) < 4:
                continue
            pt = seg_point(coords[e1[0]], coords[e1[1]],
                           coords[e2[0]], coords[e2[1]])
            if pt is not None and _strict_cross(coords[e1[0]], coords[e1[1]],
                                                coords[e2[0]], coords[e2[1]]):
                pts.append(pt)
    return pts


def _delaunay_edges(coords):
    tri = Delaunay(coords)
    edges = set()
    for s in tri.simplices:
        for a, b in ((0, 1), (1, 2), (2, 0)):
            i, j = int(s[a]), int(s[b])
            edges.add((min(i, j), max(i, j)))
    return np.array(sorted(edges))


def _smacof_step(coords, Et, L, w, n):
    """One SMACOF stress-majorisation update (verbatim from _prism's inner loop)."""
    diff = coords[Et[:, 0]] - coords[Et[:, 1]]
    dist = np.maximum(np.linalg.norm(diff, axis=1), 1e-9)
    unit = diff / dist[:, None]
    num = np.zeros((n, 2))
    den = np.zeros(n)
    np.add.at(num, Et[:, 0], w[:, None] * (coords[Et[:, 1]] + L[:, None] * unit))
    np.add.at(num, Et[:, 1], w[:, None] * (coords[Et[:, 0]] - L[:, None] * unit))
    np.add.at(den, Et[:, 0], w)
    np.add.at(den, Et[:, 1], w)
    m = den > 0
    coords[m] = num[m] / den[m, None]
    return coords


def build_keyframes(G, coords0, E):
    """Replay cf_cross_sep_prism._separate, capturing the trajectory.

    Round 0's SMACOF iterations are captured individually so the stress
    inflation is visible; later rounds and the node-edge cleanup are captured
    whole. Faithful to _prism (same Delaunay + grow-to-sep + majorisation)."""
    algo = CFCrossSepPrism(seed=SEED)
    sep_frac, overlap_frac = algo.sep_frac, algo.overlap_frac
    n = len(coords0)
    end_mask = np.zeros((len(E), n), dtype=bool)
    end_mask[np.arange(len(E)), E[:, 0]] = True
    end_mask[np.arange(len(E)), E[:, 1]] = True
    rng = np.random.default_rng(SEED)

    frames = [(coords0.copy(), "current-flow  →  crossing repair")]
    coords = coords0.copy()

    for r in range(25):                          # matches _prism max_outer
        w_, h_ = coords.max(0) - coords.min(0)
        k = math.sqrt(max(w_ * h_, 1e-12) / n)
        sep = sep_frac * k
        try:
            Et = _delaunay_edges(coords)
        except (QhullError, ValueError):
            break
        d = np.linalg.norm(coords[Et[:, 0]] - coords[Et[:, 1]], axis=1)
        if (d >= sep - 1e-12).all():
            break
        L = np.where(d < sep, sep, np.maximum(d, 1e-9))
        w = 1.0 / (L * L)
        for it in range(12):                     # matches _prism smacof_iters
            coords = _smacof_step(coords, Et, L, w, n)
            if r == 0 and it in (2, 5, 11):      # show the first inflation
                frames.append((coords.copy(),
                               "PRISM  ·  grow overlapping edges, re-settle by stress"))
        if 0 < r <= 6:                           # subsequent whole rounds
            frames.append((coords.copy(), "PRISM  ·  re-triangulate, repeat"))

    frames.append((coords.copy(), "PRISM  ·  repeat until the floor is met"))

    # residual + node-edge cleanup (identical to _separate's second loop)
    for _ in range(6):
        w_, h_ = coords.max(0) - coords.min(0)
        k = math.sqrt(max(w_ * h_, 1e-12) / n)
        sep, clear = sep_frac * k, overlap_frac * k
        if _floor_violations(coords, E, end_mask, sep, clear) == (0, 0):
            break
        coords = _project_floors(coords, E, end_mask, sep, clear, rng, rounds=6)
    frames.append((coords.copy(), "PRISM separation  ·  finished"))
    return frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", default="revolution")
    args = parser.parse_args()
    safe = args.graph
    tag = safe.replace("/", "_")
    out_path = os.path.join(OUT_DIR, f"prism_failure_{tag}.gif")
    os.makedirs(OUT_DIR, exist_ok=True)

    G = nx.convert_node_labels_to_integers(
        nx.read_graphml(os.path.join(ROOT, "data", "graphs", f"{safe}.graphml")))
    n = G.number_of_nodes()
    nodes = list(G.nodes())
    idx = {v: i for i, v in enumerate(nodes)}
    E = np.array([(idx[u], idx[v]) for u, v in G.edges()])

    # the exact layout cf_cross_sep_prism feeds into separation: cf → repair
    cf = layout_cache.load(f"currentflow_s{SEED}", G)
    if cf is None:
        Baseline(seed=SEED).layout(G)
        cf = EdgeRelaxationCurrentFlow(seed=SEED).layout(G)
        layout_cache.save(f"currentflow_s{SEED}", G, cf)
    algo = CFCrossSepPrism(seed=SEED)
    repaired = algo._repair.repair(G, {v: tuple(cf[v]) for v in nodes})
    coords0 = np.array([repaired[v] for v in nodes], float)

    # which nodes are stacked on top of another (why separation is unavoidable)
    D0 = np.linalg.norm(coords0[:, None, :] - coords0[None, :, :], axis=2)
    np.fill_diagonal(D0, np.inf)
    span0 = float((coords0.max(0) - coords0.min(0)).max()) or 1.0
    stacked = D0.min(1) < 0.01 * span0
    n_stacked = int(stacked.sum())

    frames = build_keyframes(G, coords0, E)
    start_cross = len(crossing_points(coords0, E))
    final_cross = len(crossing_points(frames[-1][0], E))

    fig = plt.figure(figsize=(11.0, 6.4))
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0.34, 0.03, 0.64, 0.94])
    ax.set_aspect("equal")
    ax.set_axis_off()

    fig.text(0.055, 0.88, safe, fontsize=17, fontweight="bold", color=INK,
             ha="left", va="center")
    fig.text(0.055, 0.82, "cf + PRISM  (literature separation)", fontsize=12,
             color="#8a8f96", ha="left", va="center")
    count_text = fig.text(0.055, 0.60, str(start_cross), fontsize=54,
                          fontweight="bold", color=CROSS_RED, ha="left", va="center")
    fig.text(0.055, 0.50, "crossings", fontsize=17, color=INK, ha="left", va="center")
    stage_text = fig.text(0.055, 0.40, frames[0][1], fontsize=13.5,
                          color="#6b7075", ha="left", va="center")
    note_text = fig.text(0.055, 0.145, "", fontsize=12, color=STACK_ORANGE,
                         ha="left", va="center")

    cam = {"c": (coords0.min(0) + coords0.max(0)) / 2,
           "r": (coords0.max(0) - coords0.min(0)).max() / 2 * 1.18 + 1e-6}

    def draw(coords, show_stacked=False, ease=0.28):
        ax.clear()
        tc = (coords.min(0) + coords.max(0)) / 2
        tr = (coords.max(0) - coords.min(0)).max() / 2 * 1.18 + 1e-6
        cam["c"] = cam["c"] + ease * (tc - cam["c"])
        cam["r"] = cam["r"] + ease * (tr - cam["r"])
        ax.set_xlim(cam["c"][0] - cam["r"], cam["c"][0] + cam["r"])
        ax.set_ylim(cam["c"][1] - cam["r"], cam["c"][1] + cam["r"])
        ax.set_aspect("equal")
        ax.set_axis_off()
        ax.add_collection(LineCollection([(coords[a], coords[b]) for a, b in E],
                                         colors=EDGE_GRAY, linewidths=1.2, zorder=1))
        cp = crossing_points(coords, E)
        if cp:
            P = np.array(cp)
            ax.scatter(P[:, 0], P[:, 1], s=42, color=CROSS_RED, alpha=0.5,
                       zorder=2, linewidths=0)
        if show_stacked:
            ax.scatter(coords[stacked, 0], coords[stacked, 1], s=200,
                       facecolors="none", edgecolors=STACK_ORANGE,
                       linewidths=1.8, zorder=4)
        ax.scatter(coords[:, 0], coords[:, 1], s=90, color=NODE_COLOR,
                   linewidths=0, zorder=3)

    writer = PillowWriter(fps=FPS)
    with writer.saving(fig, out_path, dpi=100):
        # ── phase 1: the near-planar input, with stacked nodes ringed ───────
        count_text.set_text(str(start_cross))
        stage_text.set_text(frames[0][1])
        note_text.set_text(f"{n_stacked} of {n} nodes sit stacked on another")
        for f in range(HOLD_START):
            draw(coords0, show_stacked=(f > HOLD_START // 3))
            writer.grab_frame()
        note_text.set_text("")

        # ── phase 2: the PRISM trajectory, keyframe to keyframe ─────────────
        for j in range(1, len(frames)):
            a, _ = frames[j - 1]
            b, label = frames[j]
            stage_text.set_text(label)
            if "re-settle by stress" in label:
                m, hold = MORPH_SMACOF, SETTLE_SMACOF
            elif "re-triangulate" in label or "repeat until" in label:
                m, hold = MORPH_ROUND, 2
            else:
                m, hold = MORPH_CLEAN, 2
            for f in range(1, m + 1):
                t = f / m
                t = t * t * (3 - 2 * t)
                coords = (1 - t) * a + t * b
                count_text.set_text(str(len(crossing_points(coords, E))))
                draw(coords)
                writer.grab_frame()
            coords = b.copy()
            count_text.set_text(str(len(crossing_points(coords, E))))
            for _ in range(hold):
                draw(coords)
                writer.grab_frame()

        # ── phase 3: hold on the wreckage ───────────────────────────────────
        stage_text.set_text(f"PRISM (stress-based):  {start_cross} → {final_cross} crossings")
        for _ in range(HOLD_END):
            draw(frames[-1][0])
            writer.grab_frame()
    plt.close(fig)

    print(f"Saved: {out_path}")
    print(f"{n} nodes, {len(E)} edges, {n_stacked} stacked, {len(frames)} keyframes")
    print(f"crossings: {start_cross} (cf+repair) -> {final_cross} (PRISM)")
    print(f"{os.path.getsize(out_path) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
