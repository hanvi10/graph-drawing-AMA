"""
presentation/make_relaxation_video.py
======================================
Animated visualization of the current-flow edge-relaxation algorithm, in the
exact gallery figure style (blue nodes, thin gray edges).

Timeline:
  1. the perfect grid, drawn at its natural lattice coordinates
  2. a few random long-range edges (drawn GREEN so they stay identifiable) fade
     in across the middle -- these wreck it
  3. spectral + spring is run; the green edges pull their distant endpoints
     together and fold the grid into a tangle
  4. relaxation (shown slowly): current-flow betweenness has already flagged the
     green edges as the highest-flow ones, so the loop picks exactly them. Each
     turns red ("spotted"), its spring weight is cut once, and the layout
     re-settles. Motion happens only during these per-edge settles -- the
     highlight pauses and the final hold freeze the layout, so nothing drifts or
     expands once relaxing is done.

Faithful to algorithms/edge_relaxation/currentflow.py: betweenness computed once
up front, highest-scoring edge relaxed each iteration. Edge THICKNESS is left
uniform throughout (gallery style) -- a relaxed edge is shown only by the red
flash and by the grid relaxing, and it returns to normal weight afterwards. The
spring phase is a damped mass-spring approximation of Fruchterman-Reingold, as
in the other videos.

HOW TO RUN:
    ./.venv/bin/python presentation/make_relaxation_video.py

OUTPUT:
    presentation/videos/currentflow_relaxation.gif
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from metrics import count_crossings

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "videos")
OUT_PATH = os.path.join(OUT_DIR, "currentflow_relaxation.gif")

# Exact gallery style (make_gallery.py)
NODE_COLOR = "#2f5f9e"
EDGE_GRAY = np.array([154, 165, 177]) / 255.0
EDGE_GREEN = np.array([39, 174, 96]) / 255.0    # the added long edges
EDGE_RED = np.array([192, 57, 43]) / 255.0      # an edge while being relaxed
INK = "#333333"
LW = 1.2                      # uniform edge width, kept constant throughout
HL_W = 3.0                    # transient width of the red "spotted" flash
NODE_SIZE = 55

SEED = 47
GRID = 6
N_SHORTCUTS = 4
RELAX_ROUNDS = 1              # one relaxation per edge
K_R, K_W = 0.1, 0.05          # relaxation / re-selection decay (paper defaults)

FPS = 30
# phase lengths (frames)
HOLD_GRID, FADE_EDGES, HOLD_ADDED = 20, 18, 14
MORPH_SPECTRAL, HOLD_SPECTRAL = 38, 10
SETTLE_STEPS, HOLD_TANGLED = 60, 16
HIGHLIGHT_FRAMES, RELAX_STEPS = 10, 50
HOLD_END = 40

# Damped spring integrator (matches make_layout_video.py)
DAMPING = 0.90
FORCE_CAP = 3.0
DT = 0.05          # spectral -> spring settle
DT_RELAX = 0.028   # slower steps during relaxation, so the motion reads calmly


def normalized(P):
    P = P - P.mean(axis=0)
    return P / np.abs(P).max()


def build_graph():
    """6x6 grid + N random long-range edges spanning the middle.

    The span requirement keeps them long, so they carry the highest current-flow
    betweenness and relaxation picks them first (SEED chosen so all N are the
    top-N flow edges).
    """
    G0 = nx.grid_2d_graph(GRID, GRID)
    G = nx.convert_node_labels_to_integers(G0, label_attribute="coord")
    coord = {i: np.array(G.nodes[i]["coord"], float) for i in G}

    rng = np.random.default_rng(SEED)
    nodes = list(G)
    shortcuts = []
    while len(shortcuts) < N_SHORTCUTS:
        u, v = (int(x) for x in rng.choice(nodes, 2, replace=False))
        e = tuple(sorted((u, v)))
        if u == v or G.has_edge(u, v) or e in shortcuts:
            continue
        if np.linalg.norm(coord[u] - coord[v]) < 3.0:
            continue
        G.add_edge(u, v)
        shortcuts.append(e)
    return G, coord, shortcuts


def weighted_A(edges, scale, n):
    A = np.zeros((n, n))
    for (u, v) in edges:
        A[u, v] = A[v, u] = scale[(u, v)]
    return A


def spring_sim(A, pos, steps, dt=DT):
    """Fruchterman-Reingold forces integrated with momentum and damping."""
    n = A.shape[0]
    k = np.sqrt(1.0 / n)
    pos = pos.astype(float).copy()
    vel = np.zeros_like(pos)
    out = []
    for _ in range(steps):
        delta = pos[:, None, :] - pos[None, :, :]
        dist = np.linalg.norm(delta, axis=-1)
        np.clip(dist, 0.01, None, out=dist)
        force = np.einsum("ijk,ij->ik", delta, (k * k / dist**2 - A * dist / k))
        mag = np.linalg.norm(force, axis=1, keepdims=True)
        np.clip(mag, 1e-12, None, out=mag)
        force = np.where(mag > FORCE_CAP, force / mag * FORCE_CAP, force)
        vel = (vel + force * dt) * DAMPING
        pos = pos + vel * dt
        out.append(pos.copy())
    return out


def smoothstep(u):
    return u * u * (3.0 - 2.0 * u)


def main():
    G, coord, shortcuts = build_graph()
    n = G.number_of_nodes()
    grid_edges = [tuple(sorted(e)) for e in G.edges() if tuple(sorted(e)) not in shortcuts]
    grid_E = np.array(grid_edges)
    short_E = np.array(shortcuts)

    ecfb_raw = nx.edge_current_flow_betweenness_centrality(G, normalized=False)
    ecfb = {tuple(sorted(e)): v for e, v in ecfb_raw.items()}
    scale = {tuple(sorted(e)): 1.0 for e in G.edges()}
    weight = {tuple(sorted(e)): 1.0 for e in G.edges()}
    score = {e: ecfb[e] for e in ecfb}

    P_grid = normalized(np.array([coord[i] for i in range(n)]))
    spec = nx.spectral_layout(G)
    P_spec = normalized(np.array([spec[i] for i in range(n)]))

    # frame = (pos, shortcut_alpha, hot_shortcut_index or -1, heat)
    frames = []
    frames += [(P_grid, 0.0, -1, 0.0)] * HOLD_GRID
    for f in range(1, FADE_EDGES + 1):                      # long edges fade in
        frames.append((P_grid, smoothstep(f / FADE_EDGES), -1, 0.0))
    frames += [(P_grid, 1.0, -1, 0.0)] * HOLD_ADDED
    for f in range(1, MORPH_SPECTRAL + 1):                  # straight to spectral
        s = smoothstep(f / MORPH_SPECTRAL)
        frames.append(((1 - s) * P_grid + s * P_spec, 1.0, -1, 0.0))
    frames += [(P_spec, 1.0, -1, 0.0)] * HOLD_SPECTRAL

    settle = spring_sim(weighted_A(list(G.edges()), scale, n), P_spec, SETTLE_STEPS)
    pos = settle[-1]
    for p in settle[::2]:                                   # spring folds the grid
        frames.append((p, 1.0, -1, 0.0))
    frames += [(pos, 1.0, -1, 0.0)] * HOLD_TANGLED
    tangle_pos = pos.copy()          # baseline reference for the counter

    # ── relaxation: the long middle edges, in two passes ─────────────────────
    # current-flow betweenness ranks the long edges highest, so we relax them in
    # that order; two passes weaken each edge further (scale -> ~0.01).
    order = sorted(shortcuts, key=lambda e: ecfb[e], reverse=True)
    picks = []
    for rnd in range(RELAX_ROUNDS):
        for sel in order:
            picks.append(sel)
            hot = shortcuts.index(sel)

            for _ in range(HIGHLIGHT_FRAMES):
                frames.append((pos, 1.0, hot, 1.0))

            scale[sel] *= K_R
            weight[sel] *= K_W
            score[sel] = weight[sel] * ecfb[sel]

            relax = spring_sim(weighted_A(list(G.edges()), scale, n), pos,
                               RELAX_STEPS, dt=DT_RELAX)
            for j, p in enumerate(relax):
                frames.append((p, 1.0, hot, 1.0 - (j + 1) / len(relax)))
            pos = relax[-1]

    # freeze here: no motion once the last edge has been relaxed
    frames += [(pos, 1.0, -1, 0.0)] * HOLD_END

    # ── crossings counter: crossings among the ORIGINAL grid edges only ───────
    # (the added green edges always cross the grid, so counting them would swamp
    # the signal; the story is the grid folding then being untangled)
    H = G.copy()
    H.remove_edges_from([tuple(e) for e in shortcuts])

    def cross_count(P):
        return count_crossings(
            H, {i: (float(P[i, 0]), float(P[i, 1])) for i in range(n)})

    baseline_cross = cross_count(tangle_pos)

    # ── render: dutch format — text panel on the left, graph on the right ─────
    all_pts = np.vstack([f[0] for f in frames])
    lo, hi = all_pts.min(axis=0), all_pts.max(axis=0)
    pad = 0.10 * (hi - lo).max()

    fig = plt.figure(figsize=(11.0, 6.2))            # 16:9-ish landscape
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0.35, 0.03, 0.63, 0.94])      # graph on the right
    ax.set_xlim(lo[0] - pad, hi[0] + pad)
    ax.set_ylim(lo[1] - pad, hi[1] + pad)
    ax.set_aspect("equal")
    ax.set_axis_off()

    # left caption (no graph name): live crossings + fixed baseline
    count_text = fig.text(0.055, 0.57, str(baseline_cross), fontsize=52,
                          fontweight="bold", color=NODE_COLOR, ha="left", va="center")
    fig.text(0.055, 0.47, "crossings", fontsize=17, color=INK,
             ha="left", va="center")
    fig.text(0.055, 0.38, f"baseline: {baseline_cross}", fontsize=13.5,
             color="#8a8f96", ha="left", va="center")

    grid_lc = LineCollection(P_grid[grid_E], colors=EDGE_GRAY, linewidths=LW, zorder=1)
    short_lc = LineCollection([], zorder=3)
    ax.add_collection(grid_lc)
    ax.add_collection(short_lc)
    dots = ax.scatter(P_grid[:, 0], P_grid[:, 1], s=NODE_SIZE,
                      color=NODE_COLOR, linewidths=0, zorder=2)

    writer = PillowWriter(fps=FPS)
    with writer.saving(fig, OUT_PATH, dpi=100):
        for P, alpha, hot, heat in frames:
            grid_lc.set_segments(P[grid_E])
            short_lc.set_segments(P[short_E])
            colors = np.tile(np.append(EDGE_GREEN, alpha), (len(shortcuts), 1))
            widths = np.full(len(shortcuts), LW)
            if hot >= 0:
                colors[hot, :3] = (1 - heat) * EDGE_GREEN + heat * EDGE_RED
                colors[hot, 3] = 1.0
                widths[hot] = LW + (HL_W - LW) * heat
            short_lc.set_color(colors)
            short_lc.set_linewidths(widths)
            dots.set_offsets(P)
            count_text.set_text(str(cross_count(P)))
            writer.grab_frame()
    plt.close(fig)

    final_pos = {i: tuple(frames[-1][0][i]) for i in range(n)}
    print(f"Saved: {OUT_PATH}")
    print(f"{n} nodes, {G.number_of_edges()} edges ({N_SHORTCUTS} shortcuts), "
          f"{len(frames)} frames @ {FPS} fps = {len(frames)/FPS:.1f} s, "
          f"{os.path.getsize(OUT_PATH)/1e6:.1f} MB")
    print(f"relaxed: {['SHORTCUT' if e in shortcuts else 'grid' for e in picks]}")
    print(f"final crossings: {count_crossings(G, final_pos)}")


if __name__ == "__main__":
    main()
