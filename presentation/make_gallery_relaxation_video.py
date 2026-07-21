"""
presentation/make_gallery_relaxation_video.py
=============================================
Animated current-flow edge-relaxation on a REAL gallery graph, in the exact
gallery figure style (blue nodes, thin gray edges, white background).

Graph: dutch_criticism (35 nodes, 80 edges) — one of the strongest
baseline -> current-flow improvements in the benchmark: 185 -> 123 crossings
(-33.5%). Faithful to algorithms/edge_relaxation/currentflow.py:

  * baseline = spectral -> spring (the algorithm's initial layout)
  * edge current-flow betweenness computed ONCE up front
  * each iteration relaxes the highest-scoring edge (weight cut by k_r, its
    re-selection score cut by k_w) and re-runs spring_layout warm-started from
    the previous positions.

We replay that exact loop, capture the settled layout after each step, and
animate: the edge being relaxed flashes RED, then the layout morphs to its new
settled positions. Runs up to the iteration that produced the best (fewest
crossings) layout — the same layout shown in the gallery's "CurrentFlow" panel.

Consecutive spring layouts are rescaled to a common size and orthogonally
aligned (Procrustes) to the baseline so the drawing settles in place rather
than spinning/flipping — display only; the algorithm's own coordinates are
untouched.

HOW TO RUN:
    ./.venv/bin/python presentation/make_gallery_relaxation_video.py

OUTPUT:
    presentation/videos/dutch_criticism_relaxation.gif
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.animation import PillowWriter
from matplotlib.collections import LineCollection
from scipy.linalg import orthogonal_procrustes

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from metrics import count_crossings

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GRAPHS_DIR = os.path.join(ROOT, "data", "graphs")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "videos")

SAFE_NAME = "dutch_criticism"
OUT_PATH = os.path.join(OUT_DIR, f"{SAFE_NAME}_relaxation.gif")

# Exact gallery style (make_gallery.py) — blue nodes, gray edges
NODE_COLOR = "#2f5f9e"
EDGE_GRAY = np.array([154, 165, 177]) / 255.0   # "#9aa5b1"
EDGE_RED = np.array([192, 57, 43]) / 255.0       # an edge while being relaxed
INK = "#333333"
LW = 1.2                       # uniform edge width (gallery feel)
HL_W = 3.4                     # transient width of the red "spotted" flash
NODE_SIZE = 55

# Algorithm hyper-parameters (currentflow.py defaults)
SEED = 42
K_R, K_W = 0.1, 0.05
MAX_ITER, PATIENCE = 100, 20
INIT_ITERS, LOOP_ITERS = 50, 50

FPS = 30
HOLD_START = 45                # hold on the baseline before relaxing
SPOT_FRAMES = 13               # red flash on the selected edge before it moves
MORPH_FRAMES = 30              # settle to the new layout (slower)
HOLD_AFTER = 10                # pause on each settled step
HOLD_END = 80                  # hold on the best layout


def fix_scale(P, target_rms):
    """Center at origin and scale to a constant RMS radius (display only)."""
    Q = P - P.mean(axis=0)
    r = float(np.sqrt((Q * Q).sum(axis=1).mean()))
    return Q * (target_rms / r) if r > 1e-9 else Q


def align_to(P, ref):
    """Orthogonally rotate/reflect centered P onto ref (no scaling)."""
    R, _ = orthogonal_procrustes(P, ref)
    return P @ R


def smoothstep(u):
    return u * u * (3.0 - 2.0 * u)


def run_relaxation(G):
    """Replay currentflow.layout(), capturing the settled layout each step.

    Returns (positions, picks, crossings) up to and including the best step.
      positions[0] = baseline; positions[k+1] = layout after relaxing picks[k]
      crossings[k] = crossings of positions[k]
    """
    try:
        p0 = nx.spectral_layout(G, weight=None)
    except Exception:
        p0 = nx.random_layout(G, seed=SEED)
    pos = nx.spring_layout(G, pos=p0, weight=None, seed=SEED, iterations=INIT_ITERS)

    raw = nx.edge_current_flow_betweenness_centrality(G, normalized=False)
    ecfb = {e: raw.get(e, raw.get((e[1], e[0]), 0.0)) for e in G.edges()}
    scale = {e: 1.0 for e in G.edges()}
    weight = {e: 1.0 for e in G.edges()}
    scores = {e: weight[e] * ecfb[e] for e in G.edges()}

    n = G.number_of_nodes()
    positions = [np.array([pos[i] for i in range(n)])]
    crossings = [count_crossings(G, pos)]
    picks = []

    best, best_it, last = np.inf, -1, dict(pos)
    step_pos, step_pick, step_cross = [], [], []
    for it in range(MAX_ITER):
        sel = max(scores, key=scores.get)
        scale[sel] *= K_R
        weight[sel] *= K_W
        scores[sel] = weight[sel] * ecfb[sel]
        G[sel[0]][sel[1]]["relax"] = scale[sel]

        pos = nx.spring_layout(G, pos=last.copy(), weight="relax", iterations=LOOP_ITERS)
        c = count_crossings(G, pos)

        step_pick.append(sel)
        step_pos.append(np.array([pos[i] for i in range(n)]))
        step_cross.append(c)

        if c < best:
            best, best_it = c, it
        if best_it + PATIENCE < it:
            break
        last = pos

    # keep only through the best iteration (the layout the gallery shows)
    for k in range(best_it + 1):
        picks.append(step_pick[k])
        positions.append(step_pos[k])
        crossings.append(step_cross[k])
    return positions, picks, crossings


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    G = nx.read_graphml(os.path.join(GRAPHS_DIR, f"{SAFE_NAME}.graphml"))
    G = nx.convert_node_labels_to_integers(G)
    n = G.number_of_nodes()
    edges = np.array([tuple(sorted(e)) for e in G.edges()])
    # index of each edge for red-highlighting
    edge_index = {tuple(e): i for i, e in enumerate(edges)}

    positions, picks, crossings = run_relaxation(G)

    # normalize scale + align every captured layout to the baseline frame
    target_rms = float(np.sqrt(((positions[0] - positions[0].mean(0)) ** 2).sum(1).mean()))
    P = [fix_scale(positions[0], target_rms)]
    for k in range(1, len(positions)):
        Q = fix_scale(positions[k], target_rms)
        P.append(align_to(Q, P[0]))

    n_steps = len(picks)  # == len(P) - 1

    # ── build the frame list: (positions, hot_edge_index, heat, crossings) ──
    frames = []
    frames += [(P[0], -1, 0.0, crossings[0])] * HOLD_START
    for k in range(n_steps):
        hot = edge_index[tuple(sorted(picks[k]))]
        # spotlight: fade the edge to red on the current layout
        for f in range(1, SPOT_FRAMES + 1):
            frames.append((P[k], hot, smoothstep(f / SPOT_FRAMES), crossings[k]))
        # morph: settle to the new layout, edge held red then fading back
        for f in range(1, MORPH_FRAMES + 1):
            s = smoothstep(f / MORPH_FRAMES)
            pos = (1 - s) * P[k] + s * P[k + 1]
            heat = 1.0 - 0.85 * s
            frames.append((pos, hot, heat, crossings[k + 1]))
        # brief pause on the settled layout, edge back to gray
        frames += [(P[k + 1], -1, 0.0, crossings[k + 1])] * HOLD_AFTER
    frames += [(P[-1], -1, 0.0, crossings[-1])] * HOLD_END

    # ── render (horizontal frame: text panel on the left, graph on the right) ──
    all_pts = np.vstack([f[0] for f in frames])
    lo, hi = all_pts.min(axis=0), all_pts.max(axis=0)
    pad = 0.10 * (hi - lo).max()
    baseline_cross = crossings[0]

    fig = plt.figure(figsize=(11.0, 6.2))            # 16:9-ish landscape
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0.35, 0.03, 0.63, 0.94])      # graph on the right
    ax.set_xlim(lo[0] - pad, hi[0] + pad)
    ax.set_ylim(lo[1] - pad, hi[1] + pad)
    ax.set_aspect("equal")
    ax.set_axis_off()

    # left side caption
    fig.text(0.055, 0.70, "dutch_criticism", fontsize=20, fontweight="bold",
             color=INK, ha="left", va="center")
    count_text = fig.text(0.055, 0.52, str(baseline_cross), fontsize=52,
                          fontweight="bold", color=NODE_COLOR, ha="left", va="center")
    fig.text(0.055, 0.42, "crossings", fontsize=17, color=INK,
             ha="left", va="center")
    fig.text(0.055, 0.33, f"baseline: {baseline_cross}", fontsize=13.5,
             color="#8a8f96", ha="left", va="center")

    lc = LineCollection(P[0][edges], colors=EDGE_GRAY, linewidths=LW, zorder=1)
    ax.add_collection(lc)
    dots = ax.scatter(P[0][:, 0], P[0][:, 1], s=NODE_SIZE,
                      color=NODE_COLOR, linewidths=0, zorder=2)

    n_edges = len(edges)
    writer = PillowWriter(fps=FPS)
    with writer.saving(fig, OUT_PATH, dpi=100):
        for pos, hot, heat, cross in frames:
            colors = np.tile(np.append(EDGE_GRAY, 1.0), (n_edges, 1))
            widths = np.full(n_edges, LW)
            if hot >= 0:
                colors[hot, :3] = (1 - heat) * EDGE_GRAY + heat * EDGE_RED
                widths[hot] = LW + (HL_W - LW) * heat
            lc.set_segments(pos[edges])
            lc.set_color(colors)
            lc.set_linewidths(widths)
            lc.set_zorder(1)
            dots.set_offsets(pos)
            count_text.set_text(str(int(cross)))
            writer.grab_frame()
    plt.close(fig)

    print(f"Saved: {OUT_PATH}")
    print(f"{n} nodes, {n_edges} edges, {n_steps} relaxation steps, "
          f"{len(frames)} frames @ {FPS} fps = {len(frames)/FPS:.1f} s, "
          f"{os.path.getsize(OUT_PATH)/1e6:.1f} MB")
    print(f"crossings: {crossings[0]} (baseline) -> {crossings[-1]} (best)")


if __name__ == "__main__":
    main()
