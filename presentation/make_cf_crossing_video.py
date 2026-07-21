"""
presentation/make_cf_crossing_video.py
=======================================
Visualization of the crossing-repair stage of the current-flow + crossing
pipeline (cf_cross_sep / currentflow_aesthetic share this stage), on a real
graph, in the gallery style.

The video STARTS from the layout current-flow relaxation already produced
(loaded from the layout cache) and then shows crossing repair at work: it
visits the node whose edges cause the most crossings, glides it to the
position that removes the most crossings, and repeats. Crossings are drawn as
red dots, so you watch them vanish as nodes move; the counter ticks down.

Graph: new_guinea_tribes (16 nodes, 58 edges). Faithful to
algorithms/novel/crossing_repair.py — worst-crossing node first, each move is
the algorithm's best candidate under the turning-angle guard. For the video we
animate only the moves that STRICTLY lower the crossing count (the algorithm
also makes equal-crossing moves that just shorten edges, and a smoothing pass
that straightens paths; both are skipped so every animated move visibly drops
a crossing).

HOW TO RUN:
    ./.venv/bin/python presentation/make_cf_crossing_video.py                  # new_guinea_tribes
    ./.venv/bin/python presentation/make_cf_crossing_video.py --graph kangaroo

OUTPUT:
    presentation/videos/cf_crossing_<graph>.gif
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
from algorithms.edge_relaxation.currentflow import EdgeRelaxationCurrentFlow
from algorithms.novel.crossing_repair import CrossingRepair, _strict_cross
from algorithms.novel.cf_cross_sep import CFCrossSep

SEP_FRAC = 0.4     # the cf_cross_sep node-node separation floor (fraction of k)

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "videos")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

NODE_COLOR = "#2f5f9e"
EDGE_GRAY = "#9aa5b1"
CROSS_RED = "#c0392b"
INK = "#333333"
SEED = 42

FPS = 30
HOLD_START = 30
# crossing repair — snappier: one node at a time, so it reads even when quick
RING = 4            # focus on the node before it moves
MORPH = 12          # glide to the chosen position (continuous)
SETTLE = 3          # brief pause after each move
# everything after crossing moves ALL nodes at once, so it needs more time to
# follow — slowed down deliberately (see the explanation for why it's a burst)
TIDY = 34           # settle onto the true repair output (edge-shorten + smooth)
HOLD_MID = 34       # pause on the repaired layout before expanding
SPREAD = 105        # the expansion spread (repaired -> expanded), slow
POLISH = 58         # the polish + floor-enforce settle, slow
HOLD_END = 55


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


def build_trace(G, pos):
    """Replay the crossing-repair relocation, capturing each accepted move as
    (node_index, chosen_position). Mirrors CrossingRepair._repair_once but
    keeps only the relocation passes (no smoothing)."""
    rep = CrossingRepair(seed=SEED)
    nodes, coords, E, share_mask, node_data = rep._prepare(G, pos)
    rng = np.random.default_rng(SEED)
    trace = []
    for _ in range(rep.max_rounds):
        edge_cross = rep._edge_crossing_counts(coords, E, share_mask)
        node_cross = np.zeros(len(nodes))
        for e_i, (a, b) in enumerate(E):
            node_cross[a] += edge_cross[e_i]
            node_cross[b] += edge_cross[e_i]
        order = np.argsort(-node_cross)
        span = float((coords.max(0) - coords.min(0)).max()) or 1.0

        moved = 0
        for i in order:
            if node_cross[i] == 0 or i not in node_data:
                continue
            _, u_other, other_E, mask, nb_u, nb_w = node_data[i]
            cur = coords[i].copy()
            cen = coords[u_other].mean(axis=0)
            c_cur = rep._node_incident_crossings(coords, cur, u_other, other_E, mask)
            if c_cur == 0:
                continue
            len_cur = np.linalg.norm(coords[u_other] - cur, axis=1).sum()
            pen_cur = rep._angle_penalty(coords, cur, u_other, nb_u, nb_w)

            cands = [cen, (cur + cen) / 2, 2 * cen - cur]
            cands += [cen + rng.normal(0, rep.jitter_scale * span, 2)
                      for _ in range(rep.n_jitter)]

            best_c, best_len, best_cand = c_cur, len_cur, None
            for cand in cands:
                if (np.linalg.norm(coords[u_other] - cand, axis=1).min()
                        < 1e-3 * span):
                    continue
                c_new = rep._node_incident_crossings(coords, cand, u_other,
                                                     other_E, mask)
                if c_new > best_c:
                    continue
                if (rep._angle_penalty(coords, cand, u_other, nb_u, nb_w)
                        > pen_cur + rep.angle_slack_deg):
                    continue
                len_new = np.linalg.norm(coords[u_other] - cand, axis=1).sum()
                if c_new < best_c or len_new < best_len - 1e-12:
                    best_c, best_len, best_cand = c_new, len_new, cand

            # animate only moves that strictly reduce this node's crossings
            if best_cand is not None and best_c == c_cur:
                coords[i] = best_cand          # apply silently (edge tidying)
                continue
            if best_cand is not None:
                trace.append((int(i), best_cand.copy()))
                coords[i] = best_cand
                moved += 1
        if moved == 0:
            break
    return trace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", default="new_guinea_tribes")
    args = parser.parse_args()
    safe = args.graph
    out_path = os.path.join(OUT_DIR, f"cf_crossing_{safe}.gif")

    os.makedirs(OUT_DIR, exist_ok=True)
    G = nx.convert_node_labels_to_integers(
        nx.read_graphml(os.path.join(ROOT, "data", "graphs", f"{safe}.graphml")))
    n = G.number_of_nodes()
    nodes = list(G.nodes())
    idx = {v: i for i, v in enumerate(nodes)}
    E = np.array([(idx[u], idx[v]) for u, v in G.edges()])

    cf = layout_cache.load(f"currentflow_s{SEED}", G)
    if cf is None:
        cf = EdgeRelaxationCurrentFlow(seed=SEED).layout(G)
        layout_cache.save(f"currentflow_s{SEED}", G, cf)
    # RAW cached currentflow — no pre-normalization, so the pipeline below
    # reproduces CFCrossSep.layout() exactly (the gallery layout). The view box
    # is computed from every keyframe at the end, so scale doesn't matter.
    coords0 = np.array([cf[v] for v in nodes], float)

    cf_dict = {v: tuple(coords0[i]) for i, v in enumerate(nodes)}
    trace = build_trace(G, cf_dict)
    start_cross = len(crossing_points(coords0, E))

    # phase 2 animation: strict crossing-reducing moves (the clear part)
    seq = [coords0.copy()]
    c = coords0.copy()
    for i, newp in trace:
        c = c.copy(); c[i] = newp
        seq.append(c.copy())
    repaired_strict = seq[-1].copy()

    # The real cf_cross_sep stages, so the video ENDS at the gallery layout.
    # Its crossing repair also makes equal-crossing (edge-shortening) and
    # smoothing moves the strict animation skips, so we morph the animated
    # strict layout to the true repair output before separating.
    algo = CFCrossSep(seed=SEED)
    repaired_dict = algo._repair.repair(G, cf_dict)                 # true repair
    exp_dict = algo._polish.anneal_expand(G, repaired_dict)         # expansion
    fin_dict = algo._ensure_floors(G, algo._polish.repair(G, exp_dict))  # = gallery
    repaired = np.array([repaired_dict[v] for v in nodes], float)
    expanded = np.array([exp_dict[v] for v in nodes], float)
    final_arr = np.array([fin_dict[v] for v in nodes], float)

    fig = plt.figure(figsize=(11.0, 6.4))
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0.34, 0.03, 0.64, 0.94])
    ax.set_aspect("equal")
    ax.set_axis_off()

    fig.text(0.055, 0.86, safe, fontsize=17, fontweight="bold", color=INK,
             ha="left", va="center")
    fig.text(0.055, 0.80, "cf_cross_sep", fontsize=12, color="#8a8f96",
             ha="left", va="center")
    count_text = fig.text(0.055, 0.58, str(start_cross), fontsize=54,
                          fontweight="bold", color=NODE_COLOR, ha="left", va="center")
    fig.text(0.055, 0.48, "crossings", fontsize=17, color=INK, ha="left", va="center")
    stage_text = fig.text(0.055, 0.38, "current-flow layout", fontsize=14,
                          color="#8a8f96", ha="left", va="center")

    # a camera that smoothly reframes to the current layout, so every stage
    # (raw current-flow, expansion, and the compact final) fills the panel
    cam = {"c": (coords0.min(0) + coords0.max(0)) / 2,
           "r": (coords0.max(0) - coords0.min(0)).max() / 2 * 1.18 + 1e-6}

    def draw(coords, active=None, moving_pos=None, ease=0.28):
        ax.clear()
        disp = coords.copy()
        if active is not None and moving_pos is not None:
            disp = coords.copy(); disp[active] = moving_pos
        tc = (disp.min(0) + disp.max(0)) / 2
        tr = (disp.max(0) - disp.min(0)).max() / 2 * 1.18 + 1e-6
        cam["c"] = cam["c"] + ease * (tc - cam["c"])
        cam["r"] = cam["r"] + ease * (tr - cam["r"])
        ax.set_xlim(cam["c"][0] - cam["r"], cam["c"][0] + cam["r"])
        ax.set_ylim(cam["c"][1] - cam["r"], cam["c"][1] + cam["r"])
        ax.set_aspect("equal")
        ax.set_axis_off()
        ax.add_collection(LineCollection([(disp[a], disp[b]) for a, b in E],
                                         colors=EDGE_GRAY, linewidths=1.3, zorder=1))
        cp = crossing_points(disp, E)
        if cp:
            P = np.array(cp)
            ax.scatter(P[:, 0], P[:, 1], s=55, color=CROSS_RED, alpha=0.55,
                       zorder=2, linewidths=0)
        ax.scatter(disp[:, 0], disp[:, 1], s=150, color=NODE_COLOR,
                   linewidths=0, zorder=3)
        if active is not None:
            p = moving_pos if moving_pos is not None else coords[active]
            ax.scatter([p[0]], [p[1]], s=430, facecolors="none",
                       edgecolors=INK, linewidths=2.2, zorder=4)
        # red dots follow the node live; the counter (below) updates only when a
        # move commits, so it drops monotonically instead of bouncing mid-glide

    writer = PillowWriter(fps=FPS)
    with writer.saving(fig, out_path, dpi=100):
        # ── phase 1: current-flow layout ────────────────────────────────────
        coords = coords0.copy()
        count_text.set_text(str(start_cross))
        stage_text.set_text("current-flow layout")
        for _ in range(HOLD_START):
            draw(coords); writer.grab_frame()

        # ── phase 2: crossing repair (counter drops per committed move) ─────
        stage_text.set_text("crossing repair")
        for i, newp in trace:
            start = coords[i].copy()
            for _ in range(RING):
                draw(coords, active=i, moving_pos=start); writer.grab_frame()
            for f in range(1, MORPH + 1):
                t = f / MORPH
                t = t * t * (3 - 2 * t)
                draw(coords, active=i, moving_pos=(1 - t) * start + t * newp)
                writer.grab_frame()
            coords[i] = newp
            count_text.set_text(str(len(crossing_points(coords, E))))  # commit
            for _ in range(SETTLE):
                draw(coords); writer.grab_frame()

        # settle the animated strict layout onto the true repair output
        # (the equal-crossing edge-shortening + smoothing moves)
        strict = coords.copy()
        for f in range(1, TIDY + 1):
            t = f / TIDY
            t = t * t * (3 - 2 * t)
            coords = (1 - t) * strict + t * repaired
            count_text.set_text(str(len(crossing_points(coords, E))))
            draw(coords); writer.grab_frame()
        coords = repaired.copy()

        for _ in range(HOLD_MID):
            draw(coords); writer.grab_frame()

        # ── phase 3a: expansion — spread to the separation floor ────────────
        stage_text.set_text("separation  ·  spreading nodes apart")
        for f in range(1, SPREAD + 1):
            t = f / SPREAD
            t = t * t * (3 - 2 * t)
            coords = (1 - t) * repaired + t * expanded
            count_text.set_text(str(len(crossing_points(coords, E))))
            draw(coords); writer.grab_frame()

        # ── phase 3b: polish + floor enforce — reclaim crossings, lock floors
        stage_text.set_text("separation  ·  polish reclaims crossings")
        for f in range(1, POLISH + 1):
            t = f / POLISH
            t = t * t * (3 - 2 * t)
            coords = (1 - t) * expanded + t * final_arr
            count_text.set_text(str(len(crossing_points(coords, E))))
            draw(coords); writer.grab_frame()
        coords = final_arr.copy()

        stage_text.set_text("cf_cross_sep: current-flow → cross → sep")
        for _ in range(HOLD_END):
            draw(coords); writer.grab_frame()
    plt.close(fig)

    final = {v: tuple(coords[i]) for i, v in enumerate(nodes)}
    gallery = count_crossings(G, CFCrossSep(seed=SEED).layout(G))
    rep_x = count_crossings(G, {v: tuple(repaired[i]) for i, v in enumerate(nodes)})
    print(f"Saved: {out_path}")
    print(f"{n} nodes, {len(E)} edges, {len(trace)} relocation moves")
    print(f"crossings: {start_cross} (cf) -> {rep_x} (repair) -> "
          f"{count_crossings(G, final)} (video final)")
    print(f"gallery CFCrossSep.layout crossings: {gallery}  "
          f"(match: {count_crossings(G, final) == gallery})")
    print(f"{os.path.getsize(out_path)/1e6:.1f} MB")


if __name__ == "__main__":
    main()
