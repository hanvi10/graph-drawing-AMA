"""
presentation/make_crossing_repair_video.py
===========================================
A super-simple teaching animation of the crossing-repair idea, in the gallery
figure style (blue nodes, thin gray edges).

The algorithm, in one sentence: visit the node whose edges cause the most
crossings, TRY it at a few candidate positions, and move it to whichever gives
the fewest crossings. This video shows exactly that — for each repaired node it
ghosts the node to each candidate position, marks the resulting crossings with
red dots and prints the crossing count, then turns the winner green and moves
the node there.

Candidate positions shown (from crossing_repair.py) — all relative to the
node's current spot `cur` and its neighbours' centroid `cen`:
    1. cen              the neighbour centroid
    2. (cur + cen) / 2  halfway toward it
    3. 2*cen - cur      reflected through it
(The full algorithm also samples a few random points near the centroid; those
are omitted here so the counts stay readable.)

HOW TO RUN:
    ./.venv/bin/python presentation/make_crossing_repair_video.py

OUTPUT:
    presentation/videos/crossing_repair_demo.gif
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
from algorithms.novel.crossing_repair import _strict_cross

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "videos")
OUT_PATH = os.path.join(OUT_DIR, "crossing_repair_demo.gif")

NODE_COLOR = "#2f5f9e"
EDGE_GRAY = "#9aa5b1"
INCIDENT = "#e67e22"     # the moving node's own edges, while testing
WIN = "#27ae60"          # winning candidate
GHOST = "#c3c9d0"        # untested candidate marker
CROSS_RED = "#c0392b"    # crossing markers
INK = "#333333"

FPS = 30
OPEN = 10                 # brief look at the tangle before moving
MORPH = 18                # frames to glide between two positions (continuous)
DWELL = 8                 # short pause at each candidate to read its count
HOLD_AFTER = 12           # settle on the winner
HOLD_END = 45             # final held frame


def build_example():
    """A wheel: hexagon rim (0..5) + a hub (6) joined to every rim node, with
    the hub flung outside so its spokes cross the rim. Moving the hub to the
    centre (its neighbours' centroid) removes every crossing."""
    G = nx.cycle_graph(6)                 # rim 0-1-2-3-4-5-0
    G.add_node(6)
    for r in range(6):
        G.add_edge(6, r)                  # hub connected to all rim nodes
    pos = {r: (np.cos(r * np.pi / 3), np.sin(r * np.pi / 3)) for r in range(6)}
    pos[6] = np.array([2.7, 0.35])        # hub flung outside the rim
    return G, {k: np.array(v, float) for k, v in pos.items()}


def seg_point(p1, p2, q1, q2):
    """Intersection point of segments p1p2 and q1q2 (assumed to cross)."""
    r, s = p2 - p1, q2 - q1
    rxs = r[0] * s[1] - r[1] * s[0]
    if abs(rxs) < 1e-12:
        return None
    t = ((q1 - p1)[0] * s[1] - (q1 - p1)[1] * s[0]) / rxs
    return p1 + t * r


def crossing_points(coords, E):
    """All crossing intersection points in the drawing (for the red dots)."""
    pts = []
    for a in range(len(E)):
        for b in range(a + 1, len(E)):
            e1, e2 = E[a], E[b]
            if len(set(e1) | set(e2)) < 4:       # share an endpoint
                continue
            p1, p2 = coords[e1[0]], coords[e1[1]]
            q1, q2 = coords[e2[0]], coords[e2[1]]
            if _strict_cross(p1, p2, q1, q2):
                pt = seg_point(p1, p2, q1, q2)
                if pt is not None:
                    pts.append(pt)
    return pts


def total_crossings(coords, E):
    return len(crossing_points(coords, E))


def build_trace(G, pos):
    """Replay the relocation sweep, capturing candidate tests per node.

    Only the 3 deterministic candidates are used (n_jitter = 0) so the demo
    stays readable. Returns a list of steps; each step is a dict with the node,
    the coords before the move, the candidate list (pos, total-crossings), the
    winner index, and the coords after the move.
    """
    nodes = list(G.nodes())
    idx = {v: i for i, v in enumerate(nodes)}
    coords = np.array([pos[v] for v in nodes], float)
    E = [(idx[u], idx[v]) for u, v in G.edges()]
    adj = {i: [] for i in range(len(nodes))}
    for a, b in E:
        adj[a].append(b)
        adj[b].append(a)

    def incident_cross(coords, i):
        c = 0
        for u in adj[i]:
            for (a, b) in E:
                if i in (a, b) or u in (a, b):
                    continue
                if _strict_cross(coords[i], coords[u], coords[a], coords[b]):
                    c += 1
        return c

    trace = []
    for _ in range(8):                                  # rounds
        node_cross = np.array([incident_cross(coords, i) for i in range(len(nodes))])
        order = np.argsort(-node_cross)
        cur_total = total_crossings(coords, E)
        moved = False
        for i in order:                                 # worst-crossing node first
            if node_cross[i] == 0:
                continue
            cur = coords[i].copy()
            cen = np.mean([coords[u] for u in adj[i]], axis=0)
            cands = [cen, (cur + cen) / 2.0, 2.0 * cen - cur]

            evals = []
            for c in cands:
                trial = coords.copy()
                trial[i] = c
                evals.append((c, total_crossings(trial, E)))

            # winner: fewest total crossings, tie-break by shorter incident length
            def keyf(k):
                c, tc = evals[k]
                return (tc, sum(np.linalg.norm(coords[u] - c) for u in adj[i]))
            win = min(range(len(evals)), key=keyf)

            # only move if the best candidate actually beats staying put
            if evals[win][1] >= cur_total:
                continue
            before = coords.copy()
            coords[i] = evals[win][0]
            trace.append(dict(node=i, before=before, cands=evals, win=win,
                              after=coords.copy()))
            moved = True
            break            # one node per step, so the video is sequential
        if not moved:
            break
    return nodes, E, trace


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    G, pos = build_example()
    nodes, E, trace = build_trace(G, pos)
    E = np.array(E)

    coords0 = np.array([pos[v] for v in nodes], float)
    # global view box from every layout we will show
    allc = [coords0] + [s["before"] for s in trace] + [s["after"] for s in trace]
    allc += [s["cands"][k][0] for s in trace for k in range(len(s["cands"]))]
    allpts = np.vstack([np.atleast_2d(a) for a in allc])
    lo, hi = allpts.min(0), allpts.max(0)
    pad = 0.13 * (hi - lo).max()

    fig, ax = plt.subplots(figsize=(7.2, 7.2))
    ax.set_xlim(lo[0] - pad, hi[0] + pad)
    ax.set_ylim(lo[1] - pad, hi[1] + pad)
    ax.set_aspect("equal")
    ax.set_axis_off()
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.92)

    title = ax.set_title("", fontsize=15, color=INK, pad=10)

    def draw(coords, active=None, incident_to=None, cand_markers=None,
             cross_pts=None, moving_node_pos=None):
        ax.clear()
        ax.set_xlim(lo[0] - pad, hi[0] + pad)
        ax.set_ylim(lo[1] - pad, hi[1] + pad)
        ax.set_aspect("equal")
        ax.set_axis_off()

        # base edges (gray), minus the active node's incident edges if we are
        # drawing those separately in orange
        segs = [(coords[a], coords[b]) for a, b in E]
        ax.add_collection(LineCollection(segs, colors=EDGE_GRAY, linewidths=1.6,
                                         zorder=1))

        if incident_to is not None and moving_node_pos is not None:
            i = incident_to
            inc = [(moving_node_pos, coords[b if a == i else a])
                   for a, b in E if i in (a, b)]
            ax.add_collection(LineCollection(inc, colors=INCIDENT, linewidths=2.6,
                                             zorder=3))

        # candidate markers with their counts
        if cand_markers:
            for (cx, cy), count, state in cand_markers:
                fc = {"win": WIN}.get(state, "none")
                ec = {"idle": GHOST, "test": INK, "done": INK, "win": WIN}[state]
                ax.scatter([cx], [cy], s=260, facecolors=fc, edgecolors=ec,
                           linewidths=2.0, zorder=4)
                if count is not None:
                    col = WIN if state == "win" else INK
                    ax.text(cx, cy + 0.42, str(count), ha="center", va="bottom",
                            fontsize=14, fontweight="bold", color=col, zorder=6)

        # crossing dots
        if cross_pts:
            P = np.array(cross_pts)
            ax.scatter(P[:, 0], P[:, 1], s=95, color=CROSS_RED, zorder=5,
                       edgecolors="white", linewidths=0.8)

        # nodes
        ax.scatter(coords[:, 0], coords[:, 1], s=150, color=NODE_COLOR,
                   linewidths=0, zorder=4)
        if moving_node_pos is not None:
            ax.scatter([moving_node_pos[0]], [moving_node_pos[1]], s=180,
                       color=NODE_COLOR, linewidths=0, zorder=6)
        if active is not None:
            ax.scatter([coords[active, 0]], [coords[active, 1]], s=430,
                       facecolors="none", edgecolors=INK, linewidths=2.2, zorder=6)

    cand_names = ["centroid", "halfway", "reflection"]
    writer = PillowWriter(fps=FPS)
    with writer.saving(fig, OUT_PATH, dpi=100):
        def title_cross(k, settled=False):
            ax.set_title(f"{k} crossings", fontsize=16,
                         color=(WIN if (settled and k == 0) else INK))

        def emit(base, i, mp, markers, settled=False):
            mv = base.copy()
            mv[i] = mp
            title_cross(total_crossings(mv, E), settled)
            draw(coords=(base if not settled else mv),
                 active=(None if settled else i),
                 incident_to=(None if settled else i),
                 moving_node_pos=(None if settled else mp),
                 cand_markers=markers, cross_pts=crossing_points(mv, E))
            writer.grab_frame()

        # brief look at the tangle
        title_cross(total_crossings(coords0, E))
        for _ in range(OPEN):
            draw(coords=coords0, cross_pts=crossing_points(coords0, E))
            writer.grab_frame()

        n_frames = OPEN + HOLD_END
        for step in trace:
            i = step["node"]
            base = step["before"]
            cand_pos = [c for c, _ in step["cands"]]
            win = step["win"]
            counts = [None, None, None]        # scoreboard, revealed as visited

            def markers(settled=False):
                out = []
                for k in range(3):
                    if settled and k == win:
                        st = "win"
                    elif counts[k] is not None:
                        st = "done"
                    else:
                        st = "idle"
                    out.append((tuple(cand_pos[k]), counts[k], st))
                return out

            # one continuous path: start -> each candidate in turn -> winner
            waypoints = [base[i]] + cand_pos + [cand_pos[win]]
            wp_cand = [None, 0, 1, 2, win]
            for seg in range(len(waypoints) - 1):
                a, b = waypoints[seg], waypoints[seg + 1]
                for f in range(1, MORPH + 1):
                    t = f / MORPH
                    t = t * t * (3 - 2 * t)               # ease in/out
                    emit(base, i, (1 - t) * a + t * b, markers())
                    n_frames += 1
                ci = wp_cand[seg + 1]
                if ci is not None and counts[ci] is None:
                    counts[ci] = step["cands"][ci][1]      # reveal count on arrival
                for _ in range(DWELL):
                    emit(base, i, b, markers())
                    n_frames += 1

            # settle on the winner (green), node now part of the graph
            after = step["after"]
            for _ in range(HOLD_AFTER):
                emit(after, i, after[i], markers(settled=True), settled=True)
                n_frames += 1

        # closing: solved
        final = trace[-1]["after"] if trace else coords0
        title_cross(total_crossings(final, E), settled=True)
        for _ in range(HOLD_END):
            draw(coords=final, cross_pts=crossing_points(final, E))
            writer.grab_frame()

    plt.close(fig)

    print(f"Saved: {OUT_PATH}")
    print(f"{len(nodes)} nodes, {len(E)} edges, {len(trace)} repair steps, "
          f"~{n_frames} frames @ {FPS} fps, {os.path.getsize(OUT_PATH)/1e6:.1f} MB")
    print(f"crossings: {total_crossings(coords0, E)} -> "
          f"{total_crossings(trace[-1]['after'] if trace else coords0, E)}")


if __name__ == "__main__":
    main()
