"""
presentation/make_cf_sep_video.py
==================================
The counterpart to make_vpsc_failure_video.py: the SAME near-planar input, the
SAME graph, separated by OUR method (cf_cross_sep) instead of VPSC — so the two
videos placed side by side explain why crossing-preserving separation matters.

Starting layout is identical: current-flow relaxation → crossing repair on
`revolution` (141 nodes, 4 crossings, but 134 of 141 nodes stacked). Where VPSC
fans that stack onto an axis-aligned grid and explodes to 84 crossings, our
separation:

  1. PROJECTS isotropically — every too-close pair is pushed apart *along the
     line connecting them* (algorithms/novel/cf_cross_sep._project_floors), so a
     hub's stacked neighbours fan out into a RADIAL burst and the spokes stay
     radial (non-crossing) instead of gridded.
  2. ANNEALS — the floor ramps up over the rounds, so the layout expands a
     little at a time and edges reroute as the pressure builds.
  3. RECLAIMS — after each push, a gated relocate pass moves the worst-crossing
     nodes to positions that remove crossings while respecting the floor, so the
     handful of crossings the push introduces are won straight back.

Watch round 1: the push nudges crossings 4 → 13, then the reclaim pass drops
them back to 5, and they stay ~5 for the rest of the anneal. VPSC has no reclaim
step, which is the whole difference.

Faithful to CFCrossSep.layout(): the annealed rounds are replayed exactly (the
replay is asserted equal to _polish.anneal_expand), then the verified reclaim
pass and the final floor-lock (_ensure_floors) are the true algorithm outputs —
the video ends on the exact gallery layout.

HOW TO RUN:
    ./.venv/bin/python presentation/make_cf_sep_video.py                 # revolution
    ./.venv/bin/python presentation/make_cf_sep_video.py --graph kangaroo

OUTPUT:
    presentation/videos/cf_sep_<graph>.gif
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
from algorithms.novel.cf_cross_sep import (CFCrossSep, _project_floors,
                                           _floor_violations)
from algorithms.novel.crossing_repair import _strict_cross

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "videos")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

NODE_COLOR = "#2f5f9e"
EDGE_GRAY = "#9aa5b1"
CROSS_RED = "#c0392b"
STACK_ORANGE = "#e08a1e"
GOOD_GREEN = "#2e7d46"
INK = "#333333"
SEED = 42
MAX_CROSS_DOTS = 150   # above this the red dots are unreadable clutter — hide them

FPS = 30
HOLD_START = 60        # sit on the near-planar input, ring the stacked nodes
MORPH_KEY = 50         # the round-1 push and reclaim — the mechanism, slow
SETTLE_KEY = 8
MORPH_ANNEAL = 28      # a later anneal round (small growth, crossings held)
SETTLE_ANNEAL = 3
MORPH_FINAL = 34       # verified reclaim / floor-lock
HOLD_END = 85


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


def build_keyframes(G, repaired_dict, coords0, E):
    """Replay CFCrossSep's separation, capturing (coords, label) keyframes.

    Round 1's push and reclaim are captured separately (the mechanism); rounds
    2-8 are captured at each round's end; then the true verified-reclaim and
    floor-lock outputs finish on the exact gallery layout."""
    algo = CFCrossSep(seed=SEED)
    polish = algo._polish
    nodes, coords, E2, share_mask, node_data = polish._prepare(G, repaired_dict)
    n = len(nodes)
    end_mask = np.zeros((len(E2), n), dtype=bool)
    end_mask[np.arange(len(E2)), E2[:, 0]] = True
    end_mask[np.arange(len(E2)), E2[:, 1]] = True
    rng = np.random.default_rng(polish.seed)
    anneal = max(1, polish.max_rounds - 2)
    slack = polish.angle_slack_deg

    frames = [(coords0.copy(), "current-flow  →  crossing repair")]

    for t in range(polish.max_rounds):
        ramp = min(1.0, (t + 1) / anneal)
        w, h = coords.max(0) - coords.min(0)
        span = float(max(w, h)) or 1.0
        k = math.sqrt(max(w * h, 1e-12) / n)
        sep = ramp * polish.sep_frac * k
        clear = ramp * polish.overlap_frac * k
        thresh = 2.0 * clear

        coords = _project_floors(coords, E2, end_mask, sep, clear, rng)
        if t == 0:
            frames.append((coords.copy(),
                           "our push  ·  spread each pair along its own line (radial)"))
        moved = polish._relocate_pass(coords, E2, share_mask, node_data,
                                      sep, thresh, slack, span, rng)
        for _ in range(2):
            polish._smooth_sep_pass(coords, node_data, sep, thresh, slack, span)
        if t == 0:
            frames.append((coords.copy(),
                           "our reclaim  ·  relocate nodes to win the crossings back"))
        else:
            frames.append((coords.copy(),
                           "anneal  ·  ramp the floor up, crossings held"))

        if ramp >= 1.0 and moved == 0:
            if _floor_violations(coords, E2, end_mask, sep, clear) == (0, 0):
                break

    # assert the replay is the real anneal_expand output, then finish on the
    # true verified reclaim + floor-lock (the exact gallery layout)
    exp = polish.anneal_expand(G, repaired_dict)
    assert np.allclose(coords, np.array([exp[v] for v in nodes], float)), \
        "anneal replay diverged from anneal_expand"
    rep2 = polish.repair(G, exp)
    frames.append((np.array([rep2[v] for v in nodes], float),
                   "verified reclaim  ·  full floors, continuity checked"))
    fin = algo._ensure_floors(G, rep2)
    frames.append((np.array([fin[v] for v in nodes], float),
                   "lock the separation floors"))
    return frames, nodes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", default="revolution")
    args = parser.parse_args()
    safe = args.graph
    tag = safe.replace("/", "_")
    out_path = os.path.join(OUT_DIR, f"cf_sep_{tag}.gif")
    os.makedirs(OUT_DIR, exist_ok=True)

    G = nx.convert_node_labels_to_integers(
        nx.read_graphml(os.path.join(ROOT, "data", "graphs", f"{safe}.graphml")))
    n = G.number_of_nodes()
    nodes = list(G.nodes())
    idx = {v: i for i, v in enumerate(nodes)}
    E = np.array([(idx[u], idx[v]) for u, v in G.edges()])

    cf = layout_cache.load(f"currentflow_s{SEED}", G)
    if cf is None:
        Baseline(seed=SEED).layout(G)
        cf = EdgeRelaxationCurrentFlow(seed=SEED).layout(G)
        layout_cache.save(f"currentflow_s{SEED}", G, cf)
    algo = CFCrossSep(seed=SEED)
    repaired_dict = algo._repair.repair(G, {v: tuple(cf[v]) for v in nodes})
    coords0 = np.array([repaired_dict[v] for v in nodes], float)

    D0 = np.linalg.norm(coords0[:, None, :] - coords0[None, :, :], axis=2)
    np.fill_diagonal(D0, np.inf)
    span0 = float((coords0.max(0) - coords0.min(0)).max()) or 1.0
    stacked = D0.min(1) < 0.01 * span0
    n_stacked = int(stacked.sum())

    frames, nodes_k = build_keyframes(G, repaired_dict, coords0, E)
    start_cross = len(crossing_points(coords0, E))
    final_cross = len(crossing_points(frames[-1][0], E))

    fig = plt.figure(figsize=(11.0, 6.4))
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0.34, 0.03, 0.64, 0.94])
    ax.set_aspect("equal")
    ax.set_axis_off()

    fig.text(0.055, 0.88, safe, fontsize=17, fontweight="bold", color=INK,
             ha="left", va="center")
    fig.text(0.055, 0.82, "cf_cross_sep  (our separation)", fontsize=12,
             color="#8a8f96", ha="left", va="center")
    count_text = fig.text(0.055, 0.60, str(start_cross), fontsize=54,
                          fontweight="bold", color=NODE_COLOR, ha="left", va="center")
    fig.text(0.055, 0.50, "crossings", fontsize=17, color=INK, ha="left", va="center")
    stage_text = fig.text(0.055, 0.40, frames[0][1], fontsize=13.5,
                          color="#6b7075", ha="left", va="center")
    note_text = fig.text(0.055, 0.145, "", fontsize=12, color=STACK_ORANGE,
                         ha="left", va="center")

    cam = {"c": (coords0.min(0) + coords0.max(0)) / 2,
           "r": (coords0.max(0) - coords0.min(0)).max() / 2 * 1.18 + 1e-6}

    def draw(coords, show_stacked=False, ease=0.22):
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
        # ── phase 1: the near-planar input, stacked nodes ringed ────────────
        count_text.set_text(str(start_cross))
        stage_text.set_text(frames[0][1])
        note_text.set_text(f"{n_stacked} of {n} nodes sit stacked on another")
        for f in range(HOLD_START):
            draw(coords0, show_stacked=(f > HOLD_START // 3))
            writer.grab_frame()
        note_text.set_text("")

        # ── phase 2: our separation, keyframe to keyframe ───────────────────
        for j in range(1, len(frames)):
            a, _ = frames[j - 1]
            b, label = frames[j]
            stage_text.set_text(label)
            if "our push" in label or "our reclaim" in label:
                m, hold = MORPH_KEY, SETTLE_KEY
                note_text.set_text("radial spread — spokes stay radial, so crossings don't multiply"
                                   if "push" in label else "")
            elif "anneal" in label:
                m, hold = MORPH_ANNEAL, SETTLE_ANNEAL
                note_text.set_text("")
            else:
                m, hold = MORPH_FINAL, 3
                note_text.set_text("")
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

        # ── phase 3: hold on the clean, separated layout ────────────────────
        count_text.set_color(GOOD_GREEN)
        stage_text.set_text(f"cf_cross_sep:  {start_cross} → {final_cross} crossings, 0 stacked")
        for _ in range(HOLD_END):
            draw(frames[-1][0])
            writer.grab_frame()
    plt.close(fig)

    gallery = count_crossings(G, algo.layout(G))
    print(f"Saved: {out_path}")
    print(f"{n} nodes, {len(E)} edges, {n_stacked} stacked, {len(frames)} keyframes")
    print(f"crossings: {start_cross} (cf+repair) -> {final_cross} (cf_cross_sep)")
    print(f"gallery match: {final_cross == gallery} (gallery={gallery})")
    print(f"{os.path.getsize(out_path) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
