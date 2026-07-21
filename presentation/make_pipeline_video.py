"""
presentation/make_pipeline_video.py
====================================
The WHOLE cf_cross_sep pipeline, end to end, on one graph, as a single long
continuous video — so you can watch a random scatter of nodes become the final
drawing, one stage at a time:

    random  →  spectral layout  →  spring forces        (the baseline)
            →  current-flow edge relaxation
            →  crossing repair
            →  separation                               (the final layout)

Every stage is the REAL thing, not a mock-up:
  * spectral is nx.spectral_layout; the spring phase is a damped
    Fruchterman-Reingold integrator (smooth momentum version of the same force
    law) settling onto the true baseline layout;
  * the current-flow stage replays the actual relaxation loop
    (algorithms/edge_relaxation/currentflow.py), one edge relaxed per step,
    settling on the layout the algorithm keeps (its minimum-crossing iterate);
  * crossing repair replays the real node relocations
    (algorithms/novel/crossing_repair.py);
  * separation replays the real annealed projection + reclaim rounds
    (algorithms/novel/cf_cross_sep.py) and ends on the exact gallery layout.

Deliberately clean: nodes and edges only, no crossing dots and no node
highlights — just the graph flowing through the pipeline, with a stage label.
A smoothly reframing camera keeps every stage filling the panel.

No ffmpeg on this machine, so the output is a GIF (the default graph is small,
so even a ~40 s clip stays a few MB). Pass --graph to change the graph.

HOW TO RUN:
    ./.venv/bin/python presentation/make_pipeline_video.py                       # malaria_genes__HVR_3
    ./.venv/bin/python presentation/make_pipeline_video.py --graph revolution

OUTPUT:
    presentation/videos/pipeline_<graph>.gif
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.animation import PillowWriter
from matplotlib.collections import LineCollection

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import layout_cache
from metrics import count_crossings
from algorithms.baseline import Baseline
from algorithms.edge_relaxation.currentflow import EdgeRelaxationCurrentFlow
from algorithms.novel.cf_cross_sep import CFCrossSep
from make_cf_sep_video import build_keyframes

OUT_DIR = os.path.join(HERE, "videos")

NODE_COLOR = "#2f5f9e"
EDGE_GRAY = "#9aa5b1"
INK = "#333333"
MUTE = "#8a8f96"
SEED = 42
FPS = 30

# damped Fruchterman-Reingold integrator (from make_layout_video.py) — a smooth
# momentum version of the same force law, for the spring phase
DT, DAMPING, FORCE_CAP = 0.055, 0.90, 3.0


def normalized(P):
    P = P - P.mean(axis=0)
    return P / max(np.abs(P).max(), 1e-9)


def spring_sim(A, pos, steps):
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
        vel = (vel + force * DT) * DAMPING
        pos = pos + vel * DT
        out.append(pos.copy())
    return out


def smoothstep(u):
    return u * u * (3.0 - 2.0 * u)


def capture_currentflow(G, nodes):
    """Replay currentflow.layout()'s relaxation loop, returning the trajectory
    of positions [baseline, after edge 1, after edge 2, ...] truncated at the
    minimum-crossing iterate (the layout the algorithm keeps)."""
    alg = EdgeRelaxationCurrentFlow(seed=SEED)
    Gc = G.copy()
    np.random.seed(SEED)
    pos = alg._initial_layout(Gc, SEED, alg.initial_layout_iterations)
    raw = nx.edge_current_flow_betweenness_centrality(Gc, normalized=False)
    ecfb = {e: raw.get(e, raw.get((e[1], e[0]), 0.0)) for e in Gc.edges()}
    scale = {e: 1.0 for e in Gc.edges()}
    weight = {e: 1.0 for e in Gc.edges()}
    scores = {e: weight[e] * ecfb[e] for e in Gc.edges()}

    def arr(p):
        return np.array([p[v] for v in nodes], float)

    traj = [arr(pos)]
    best, best_it, best_idx = np.inf, -1, 0
    last = pos.copy()
    for it in range(alg.max_iter):
        sel = max(scores, key=scores.get)
        scale[sel] *= alg.k_r
        weight[sel] *= alg.k_w
        scores[sel] = weight[sel] * ecfb[sel]
        Gc[sel[0]][sel[1]]["relax"] = scale[sel]
        pos = nx.spring_layout(Gc, pos=last.copy(), weight="relax",
                               iterations=alg.loop_spring_iters)
        traj.append(arr(pos))
        cr = count_crossings(Gc, pos)
        if cr < best:
            best, best_it, best_idx = cr, it, len(traj) - 1
        if best_it + alg.patience < it:
            break
        last = pos
    return traj[:best_idx + 1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", default="malaria_genes__HVR_3")
    args = parser.parse_args()
    safe = args.graph
    tag = safe.replace("/", "_")
    out_path = os.path.join(OUT_DIR, f"pipeline_{tag}.gif")
    os.makedirs(OUT_DIR, exist_ok=True)

    G = nx.convert_node_labels_to_integers(
        nx.read_graphml(os.path.join(ROOT, "data", "graphs", f"{safe}.graphml")))
    nodes = list(G.nodes())
    idx = {v: i for i, v in enumerate(nodes)}
    E = np.array([(idx[u], idx[v]) for u, v in G.edges()])
    A = nx.to_numpy_array(G, nodelist=nodes)

    # ── the real anchor layouts of every stage ──────────────────────────────
    rand = normalized(np.array([v for v in
                                nx.random_layout(G, seed=SEED).values()]))
    spec = normalized(np.array([nx.spectral_layout(G)[v] for v in nodes]))
    baseline = layout_cache.load(f"baseline_s{SEED}", G)
    if baseline is None:
        baseline = Baseline(seed=SEED).layout(G)
        layout_cache.save(f"baseline_s{SEED}", G, baseline)
    B = np.array([baseline[v] for v in nodes], float)

    cf = layout_cache.load(f"currentflow_s{SEED}", G)
    if cf is None:
        cf = EdgeRelaxationCurrentFlow(seed=SEED).layout(G)
        layout_cache.save(f"currentflow_s{SEED}", G, cf)
    C = np.array([cf[v] for v in nodes], float)

    algo = CFCrossSep(seed=SEED)
    repaired = algo._repair.repair(G, {v: tuple(cf[v]) for v in nodes})
    Rep = np.array([repaired[v] for v in nodes], float)

    # captured real trajectories
    spring = spring_sim(A, spec, steps=170)                 # spec -> ~baseline
    cf_traj = capture_currentflow(G, nodes)                 # B -> C (real)
    sep_frames, _ = build_keyframes(G, repaired, Rep, E)    # Rep -> F keyframes

    xin = count_crossings(G, {v: tuple(C[i]) for i, v in enumerate(nodes)})
    xrep = count_crossings(G, {v: tuple(Rep[i]) for i, v in enumerate(nodes)})
    F = sep_frames[-1][0]
    xfin = count_crossings(G, {v: tuple(F[i]) for i, v in enumerate(nodes)})

    # ── figure ──────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(11.0, 6.4))
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0.34, 0.03, 0.64, 0.94])
    ax.set_aspect("equal")
    ax.set_axis_off()

    fig.text(0.055, 0.90, safe, fontsize=17, fontweight="bold", color=INK,
             ha="left", va="center")
    fig.text(0.055, 0.845, "cf_cross_sep  ·  full pipeline", fontsize=12,
             color=MUTE, ha="left", va="center")
    phase_text = fig.text(0.055, 0.55, "", fontsize=20, fontweight="bold",
                          color=NODE_COLOR, ha="left", va="center")
    sub_text = fig.text(0.055, 0.47, "", fontsize=13, color=MUTE,
                        ha="left", va="center")
    cross_text = fig.text(0.055, 0.14, "", fontsize=12.5, color=MUTE,
                          ha="left", va="center")

    cam = {"c": (rand.min(0) + rand.max(0)) / 2,
           "r": (rand.max(0) - rand.min(0)).max() / 2 * 1.18 + 1e-6,
           "init": False}
    n = G.number_of_nodes()
    node_s = max(28, 150 - n)          # big dots for small graphs, small for large
    edge_lw = 1.3 if n < 60 else 1.0

    # live crossing count of whatever layout is on screen — the SAME metric the
    # paper/gallery report, so the number on screen always matches the algorithm's
    def xcount(coords):
        return count_crossings(G, {nodes[i]: (coords[i, 0], coords[i, 1])
                                   for i in range(n)})

    def draw(coords, ease=0.16):
        ax.clear()
        tc = (coords.min(0) + coords.max(0)) / 2
        tr = (coords.max(0) - coords.min(0)).max() / 2 * 1.18 + 1e-6
        if not cam["init"]:
            cam["c"], cam["r"], cam["init"] = tc, tr, True
        cam["c"] = cam["c"] + ease * (tc - cam["c"])
        cam["r"] = cam["r"] + ease * (tr - cam["r"])
        ax.set_xlim(cam["c"][0] - cam["r"], cam["c"][0] + cam["r"])
        ax.set_ylim(cam["c"][1] - cam["r"], cam["c"][1] + cam["r"])
        ax.set_aspect("equal")
        ax.set_axis_off()
        ax.add_collection(LineCollection([(coords[a], coords[b]) for a, b in E],
                                         colors=EDGE_GRAY, linewidths=edge_lw, zorder=1))
        ax.scatter(coords[:, 0], coords[:, 1], s=node_s, color=NODE_COLOR,
                   linewidths=0, zorder=2)

    writer = PillowWriter(fps=FPS)
    state = {"suffix": ""}

    def frame(coords):
        """One rendered frame: draw + LIVE crossing count of what's on screen."""
        draw(coords)
        cross_text.set_text(f"crossings  {xcount(coords)}{state['suffix']}")
        writer.grab_frame()

    with writer.saving(fig, out_path, dpi=100):

        def hold(coords, n):
            for _ in range(n):
                frame(coords)

        def morph(a, b, n):
            for f in range(1, n + 1):
                t = smoothstep(f / n)
                frame((1 - t) * a + t * b)

        def play(seq, per):
            """Morph through a list of captured coord states, `per` frames each."""
            for a, b in zip(seq, seq[1:]):
                morph(a, b, per)

        # 0 — random start
        phase_text.set_text("random start")
        sub_text.set_text("nodes placed at random")
        hold(rand, 40)

        # 1 — baseline: spectral layout
        phase_text.set_text("baseline · spectral")
        sub_text.set_text("eigenvectors of the graph Laplacian")
        morph(rand, spec, 55)
        hold(spec, 30)

        # 1 — baseline: spring forces (damped Fruchterman-Reingold)
        phase_text.set_text("baseline · spring")
        sub_text.set_text("spring forces spread the nodes")
        play([spec] + spring, 1)             # spring is fine-grained already
        morph(spring[-1], B, 34)             # settle onto the true baseline
        hold(B, 40)

        # 2 — current-flow edge relaxation
        phase_text.set_text("current-flow relaxation")
        sub_text.set_text("relax the highest current-flow edges, one at a time")
        play([B] + cf_traj, 16)
        morph(cf_traj[-1], C, 16)
        hold(C, 50)

        # 3 — crossing repair — ONE smooth flow: the graph draws itself
        # together to cut crossings (crossing repair genuinely packs nodes,
        # ~3.7x smaller here — that packing is what separation then undoes)
        phase_text.set_text("crossing repair")
        sub_text.set_text("nodes flow together to cut crossings — this packs them")
        morph(C, Rep, 90)
        hold(Rep, 50)

        # 4 — separation
        phase_text.set_text("separation")
        sub_text.set_text("push the stack apart, then reclaim the crossings")
        for a, (b, _) in zip([f[0] for f in sep_frames], sep_frames[1:]):
            morph(a, b, 28)
        state["suffix"] = "    ·    0 stacked"
        hold(F, 95)

    plt.close(fig)

    gallery = count_crossings(G, algo.layout(G))
    print(f"Saved: {out_path}")
    print(f"{G.number_of_nodes()} nodes, {len(E)} edges")
    print(f"crossings:  cf={xin}  repair={xrep}  final={xfin}  "
          f"(gallery={gallery}, match={xfin == gallery})")
    print(f"spring steps {len(spring)}, cf iters {len(cf_traj)-1}, "
          f"sep keyframes {len(sep_frames)}")
    print(f"{os.path.getsize(out_path) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
