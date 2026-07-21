"""
presentation/make_vpsc_failure_video.py
=========================================
Why a PROVEN literature overlap remover (VPSC / Fast Node Overlap Removal,
Dwyer, Marriott & Stuckey 2005) destroys a near-planar layout.

The video starts from the exact layout cf_cross_sep feeds into its separation
stage: current-flow relaxation → crossing repair. On `revolution` (141 nodes)
that layout is near-planar — only 4 crossings — but crossing repair has packed
the graph so hard that 134 of the 141 nodes are stacked on top of each other
(1700 coincident pairs). Separation is unavoidable; the question is HOW.

VPSC separates one axis at a time: solve X-overlaps, then Y-overlaps. So it fans
a coincident stack of a hub's neighbours onto a rectangular GRID, and grid-
arranged spokes cross everything. You watch the counter climb 4 → 63 in the
first round alone, settling at 84. The right theory (minimal displacement) with
the wrong geometry for radial graph structure — this is the evidence that
cf_cross_sep's ISOTROPIC projection (push apart along the connecting line, so
stacks fan radially and spokes stay non-crossing) is what preserves crossings.

Faithful to algorithms/novel/cf_cross_sep_vpsc.py._separate: same floor
sep_frac·k recomputed each round, same VPSC solver (algorithms/novel/vpsc.py),
same iterate-to-floor loop. The first round is broken into its X-pass and
Y-pass so the axis-aligned snapping into a grid is visible; later rounds and the
node-edge cleanup are morphed to the true algorithm output.

HOW TO RUN:
    ./.venv/bin/python presentation/make_vpsc_failure_video.py            # revolution
    ./.venv/bin/python presentation/make_vpsc_failure_video.py --graph mist/ppi_zebrafish

OUTPUT:
    presentation/videos/vpsc_failure_<graph>.gif
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import layout_cache
from metrics import count_crossings
from algorithms.baseline import Baseline
from algorithms.edge_relaxation.currentflow import EdgeRelaxationCurrentFlow
from algorithms.novel.cf_cross_sep import CFCrossSep, _project_floors, _floor_violations
from algorithms.novel.crossing_repair import _strict_cross
from algorithms.novel.vpsc import solve_vpsc

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "videos")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

NODE_COLOR = "#2f5f9e"
EDGE_GRAY = "#9aa5b1"
CROSS_RED = "#c0392b"
STACK_ORANGE = "#e08a1e"
INK = "#333333"
SEED = 42
MAX_CROSS_DOTS = 150   # above this the red dots are unreadable clutter — hide them

FPS = 30
HOLD_START = 55       # sit on the near-planar input, highlight the stacked nodes
MORPH_AXIS = 26       # one VPSC axis solve (X or Y) — slow, this is the mechanism
SETTLE_AXIS = 6
MORPH_ROUND = 24      # a later whole VPSC round
MORPH_CLEAN = 26      # node-edge cleanup to the true algorithm output
HOLD_END = 70


# ── VPSC axis constraints (verbatim from algorithms/novel/vpsc.remove_overlaps)
def _axis_constraints(coords, axis, sep):
    n = len(coords)
    other = 1 - axis
    a, o = coords[:, axis], coords[:, other]
    cons = []
    order = np.argsort(a)
    for ii in range(n):
        i = order[ii]
        for jj in range(ii + 1, n):
            j = order[jj]
            if a[j] - a[i] >= sep:
                break
            if abs(o[i] - o[j]) < sep:
                cons.append((i, j, sep))
    return cons


def _vpsc_axis(coords, axis, sep):
    """One VPSC solve along `axis` (the atomic step FNOR repeats)."""
    cons = _axis_constraints(coords, axis, sep)
    if not cons:
        return coords, False
    out = coords.copy()
    out[:, axis] = solve_vpsc(coords[:, axis], np.ones(len(coords)), cons)
    return out, not np.allclose(out[:, axis], coords[:, axis])


# ── geometry for drawing crossings ──────────────────────────────────────────
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


def build_keyframes(G, coords0, E):
    """Replay cf_cross_sep_vpsc._separate, capturing the trajectory.

    Returns a list of (coords, stage_label). The first VPSC round is split into
    its X-pass and Y-pass so the axis-aligned grid fan-out is visible; the
    remaining rounds and the projection cleanup are captured whole."""
    algo = CFCrossSep(seed=SEED)
    sep_frac, overlap_frac = algo.sep_frac, algo.overlap_frac
    n = len(coords0)
    end_mask = np.zeros((len(E), n), dtype=bool)
    end_mask[np.arange(len(E)), E[:, 0]] = True
    end_mask[np.arange(len(E)), E[:, 1]] = True
    rng = np.random.default_rng(SEED)

    frames = [(coords0.copy(), "current-flow  →  crossing repair")]
    coords = coords0.copy()

    for r in range(12):
        w_, h_ = coords.max(0) - coords.min(0)
        k = math.sqrt(max(w_ * h_, 1e-12) / n)
        D = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)
        np.fill_diagonal(D, np.inf)
        floor = sep_frac * k
        if D.min() >= floor - 1e-9:
            break

        if r == 0:
            # split the first, most dramatic round into its axis passes
            for p in range(3):
                for axis in (0, 1):
                    coords, changed = _vpsc_axis(coords, axis, floor)
                    if changed:
                        lbl = ("VPSC  ·  separate overlaps on the X axis"
                               if axis == 0 else
                               "VPSC  ·  then separate on the Y axis")
                        frames.append((coords.copy(), lbl))
        else:
            # faithful whole-round FNOR (X→Y, three internal passes)
            for _ in range(3):
                for axis in (0, 1):
                    coords, _ = _vpsc_axis(coords, axis, floor)
            frames.append((coords.copy(),
                           "VPSC  ·  repeat until the floor is met"))

    # residual + node-edge cleanup (identical to _separate's second loop)
    for _ in range(6):
        w_, h_ = coords.max(0) - coords.min(0)
        k = math.sqrt(max(w_ * h_, 1e-12) / n)
        sep, clear = sep_frac * k, overlap_frac * k
        if _floor_violations(coords, E, end_mask, sep, clear) == (0, 0):
            break
        coords = _project_floors(coords, E, end_mask, sep, clear, rng, rounds=6)
    frames.append((coords.copy(), "VPSC separation  ·  finished"))
    return frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", default="revolution")
    args = parser.parse_args()
    safe = args.graph
    tag = safe.replace("/", "_")
    out_path = os.path.join(OUT_DIR, f"vpsc_failure_{tag}.gif")
    os.makedirs(OUT_DIR, exist_ok=True)

    G = nx.convert_node_labels_to_integers(
        nx.read_graphml(os.path.join(ROOT, "data", "graphs", f"{safe}.graphml")))
    n = G.number_of_nodes()
    nodes = list(G.nodes())
    idx = {v: i for i, v in enumerate(nodes)}
    E = np.array([(idx[u], idx[v]) for u, v in G.edges()])

    # the exact layout cf_cross_sep_vpsc feeds into separation: cf → repair
    cf = layout_cache.load(f"currentflow_s{SEED}", G)
    if cf is None:
        Baseline(seed=SEED).layout(G)
        cf = EdgeRelaxationCurrentFlow(seed=SEED).layout(G)
        layout_cache.save(f"currentflow_s{SEED}", G, cf)
    algo = CFCrossSep(seed=SEED)
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
    fig.text(0.055, 0.82, "cf + VPSC  (literature separation)", fontsize=12,
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
        if cp and len(cp) <= MAX_CROSS_DOTS:
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
            # pulse the stacked-node rings on for the second half of the hold
            draw(coords0, show_stacked=(f > HOLD_START // 3))
            writer.grab_frame()
        note_text.set_text("")

        # ── phase 2: the VPSC trajectory, keyframe to keyframe ──────────────
        for j in range(1, len(frames)):
            a, _ = frames[j - 1]
            b, label = frames[j]
            stage_text.set_text(label)
            if j == 1 or "X axis" in label or "Y axis" in label:
                m, hold = MORPH_AXIS, SETTLE_AXIS
            elif "repeat" in label:
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
        stage_text.set_text(f"VPSC (axis-aligned):  {start_cross} → {final_cross} crossings")
        for _ in range(HOLD_END):
            draw(frames[-1][0])
            writer.grab_frame()
    plt.close(fig)

    print(f"Saved: {out_path}")
    print(f"{n} nodes, {len(E)} edges, {n_stacked} stacked, {len(frames)} keyframes")
    print(f"crossings: {start_cross} (cf+repair) -> {final_cross} (VPSC)")
    print(f"{os.path.getsize(out_path) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
