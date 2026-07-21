"""
presentation/make_layout_video.py
==================================
Animated visualization of the baseline layout algorithm (spectral + spring),
in the same visual style as the gallery figures. No text anywhere.

Timeline:
  1. random layout, held briefly
  2. continuous morph (smoothstep) from random to the spectral layout
  3. spring phase: a damped mass-spring simulation, integrated in real time
  4. final layout, held

About the spring phase: this is a smooth APPROXIMATION of Fruchterman-Reingold,
not FR itself. FR normalizes every node's step to exactly the current
temperature, so each node teleports ~10% of the layout span on iteration 1 and
barely moves by iteration 50 -- correct as an optimizer, but it reads as jerky.
Here the same FR force law (repulsion k^2/d, attraction d^2/k) drives a real
integrator with velocity and viscous damping, so nodes accelerate, coast and
settle the way actual springs do. Same forces, same fixed points, smooth motion.

Default example: connected_caveman_graph(5, 4) -- five 4-cliques in a ring.
Spectral places the five communities around a pentagon but squashes each clique;
the spring phase then inflates them. Global structure from spectral, local
spacing from springs -- both stages visibly do something.

HOW TO RUN:
    ./.venv/bin/python presentation/make_layout_video.py            # all examples
    ./.venv/bin/python presentation/make_layout_video.py caveman54  # just one

OUTPUT:
    presentation/videos/spectral_spring_<example>.gif
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

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "videos")

# Gallery figure style (make_gallery.py)
NODE_COLOR = "#2f5f9e"
EDGE_COLOR = "#9aa5b1"

SEED = 42
FPS = 30
HOLD_RANDOM, MORPH, HOLD_SPECTRAL, HOLD_FINAL = 20, 75, 25, 45

# Damped spring integrator. dt small enough that no node crosses a meaningful
# distance in one frame; damping high enough that the mesh settles instead of
# ringing. FORCE_CAP tames the huge repulsion spikes where spectral has left
# clique nodes almost coincident.
SPRING_STEPS = 260
DT = 0.055
DAMPING = 0.90
FORCE_CAP = 3.0


def _largest_component(G):
    """Keep the biggest connected piece; spectral layout needs one component."""
    if nx.is_connected(G):
        return nx.convert_node_labels_to_integers(G)
    big = max(nx.connected_components(G), key=len)
    return nx.convert_node_labels_to_integers(G.subgraph(big).copy())


def ring_of_cycles(n_rings=4, ring_size=6):
    G = nx.Graph()
    for r in range(n_rings):
        nx.add_cycle(G, [r * ring_size + i for i in range(ring_size)])
    for r in range(n_rings):
        G.add_edge(r * ring_size,
                   ((r + 1) % n_rings) * ring_size + ring_size // 2)
    return G


EXAMPLES = {
    # planar / clustered
    "caveman54":   lambda: nx.connected_caveman_graph(5, 4),
    "caveman36":   lambda: nx.connected_caveman_graph(3, 6),
    "tree":        lambda: nx.balanced_tree(3, 3),
    "ringcycles":  lambda: ring_of_cycles(4, 6),
    # non-planar
    "petersen":    lambda: nx.petersen_graph(),
    "smallworld":  lambda: nx.watts_strogatz_graph(26, 4, 0.18, seed=SEED),
    "communities": lambda: _largest_component(
        nx.random_partition_graph([7, 7, 7, 7], 0.55, 0.035, seed=SEED)),
    "karate":      lambda: nx.karate_club_graph(),
}


def normalized(P):
    """Center on the origin and scale the largest half-extent to 1."""
    P = P - P.mean(axis=0)
    return P / np.abs(P).max()


def spring_sim(A, pos, steps, dt=DT, damping=DAMPING, cap=FORCE_CAP):
    """Fruchterman-Reingold forces, integrated with momentum and damping."""
    n = A.shape[0]
    k = np.sqrt(1.0 / n)
    pos = pos.astype(float).copy()
    vel = np.zeros_like(pos)
    for _ in range(steps):
        delta = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dist = np.linalg.norm(delta, axis=-1)
        np.clip(dist, 0.01, None, out=dist)
        # repulsion k^2/d between every pair, attraction d^2/k along edges
        force = np.einsum("ijk,ij->ik", delta,
                          (k * k / dist**2 - A * dist / k))
        mag = np.linalg.norm(force, axis=1, keepdims=True)
        np.clip(mag, 1e-12, None, out=mag)
        force = np.where(mag > cap, force / mag * cap, force)
        vel = (vel + force * dt) * damping
        pos = pos + vel * dt
        yield pos.copy()


def smoothstep(u):
    return u * u * (3.0 - 2.0 * u)


def build_frames(G):
    nodes = list(G.nodes())
    A = nx.to_numpy_array(G, nodelist=nodes)
    edges = np.array([(nodes.index(u), nodes.index(v)) for u, v in G.edges()])

    rand_pos = nx.random_layout(G, seed=SEED)
    spec_pos = nx.spectral_layout(G)
    P_rand = normalized(np.array([rand_pos[v] for v in nodes]))
    P_spec = normalized(np.array([spec_pos[v] for v in nodes]))

    spring = list(spring_sim(A, P_spec, SPRING_STEPS))

    frames = [P_rand] * HOLD_RANDOM
    for f in range(1, MORPH + 1):
        s = smoothstep(f / MORPH)
        frames.append((1 - s) * P_rand + s * P_spec)
    frames += [P_spec] * HOLD_SPECTRAL
    frames += spring
    frames += [spring[-1]] * HOLD_FINAL
    return frames, edges, spring


def render(name, G):
    out_path = os.path.join(OUT_DIR, f"spectral_spring_{name}.gif")
    frames, edges, spring = build_frames(G)

    all_pts = np.vstack(frames)
    lo, hi = all_pts.min(axis=0), all_pts.max(axis=0)
    pad = 0.07 * (hi - lo).max()
    node_size = max(35, 120 - 2 * G.number_of_nodes())

    fig, ax = plt.subplots(figsize=(5.6, 5.6))
    ax.set_xlim(lo[0] - pad, hi[0] + pad)
    ax.set_ylim(lo[1] - pad, hi[1] + pad)
    ax.set_aspect("equal")
    ax.set_axis_off()
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    lines = LineCollection(frames[0][edges], colors=EDGE_COLOR,
                           linewidths=1.4, zorder=1)
    ax.add_collection(lines)
    dots = ax.scatter(frames[0][:, 0], frames[0][:, 1], s=node_size,
                      color=NODE_COLOR, linewidths=0, zorder=2)

    writer = PillowWriter(fps=FPS)
    with writer.saving(fig, out_path, dpi=100):
        for P in frames:
            lines.set_segments(P[edges])
            dots.set_offsets(P)
            writer.grab_frame()
    plt.close(fig)

    step = np.array([np.linalg.norm(b - a, axis=1).max()
                     for a, b in zip(spring, spring[1:])])
    planar = nx.check_planarity(G)[0]
    print(f"{name:12s} {G.number_of_nodes():3d}n {G.number_of_edges():3d}e  "
          f"{'planar' if planar else 'NON-planar':10s} "
          f"max jump/frame {step.max():.4f}  settles to {step[-1]:.5f}  "
          f"{os.path.getsize(out_path)/1e6:.1f} MB")


def main():
    wanted = sys.argv[1:] or list(EXAMPLES)
    unknown = [w for w in wanted if w not in EXAMPLES]
    if unknown:
        raise SystemExit(f"unknown example(s): {unknown}\n"
                         f"available: {list(EXAMPLES)}")
    os.makedirs(OUT_DIR, exist_ok=True)
    for name in wanted:
        render(name, EXAMPLES[name]())
    print(f"\n{len(wanted)} gif(s) in {OUT_DIR}")


if __name__ == "__main__":
    main()
