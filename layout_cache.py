"""
layout_cache.py
===============
Disk cache for expensive layout stages, keyed by graph structure.

The currentflow relaxation (~30-60 s/graph) is the shared first stage of
several pipelines; recomputing it for every pipeline variant dominates
iteration time. This cache stores the settled positions once per
(stage, graph) pair, so later runs and downstream pipelines reuse them.

The key is a hash of the graph's node count and sorted edge list, so it is
independent of file names and stable across runs. Positions are stored as an
(n, 2) array in list(G.nodes()) order — safe because every runner loads
graphs with convert_node_labels_to_integers, giving a stable node order.

NOTE: layouts are deterministic only within one numerical environment
(numpy/scipy versions); a cache produced in this venv matches what this venv
would recompute. Delete data/layout_cache/ after changing the environment.

Usage:
    from layout_cache import cached_layout
    pos = cached_layout("currentflow_s42", G,
                        lambda: EdgeRelaxationCurrentFlow(seed=42).layout(G))
"""

import hashlib
import os

import numpy as np

CACHE_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "data", "layout_cache")


def graph_key(G) -> str:
    """Structure hash: node count + sorted edge list."""
    h = hashlib.sha1()
    h.update(f"n={G.number_of_nodes()}|".encode())
    for u, v in sorted(tuple(sorted((str(u), str(v)))) for u, v in G.edges()):
        h.update(f"{u},{v};".encode())
    return h.hexdigest()[:16]


def load(stage: str, G):
    """Return cached {node: (x, y)} for this stage+graph, or None."""
    path = os.path.join(CACHE_ROOT, stage, graph_key(G) + ".npz")
    if not os.path.exists(path):
        return None
    try:
        arr = np.load(path)["pos"]
    except Exception:
        return None
    nodes = list(G.nodes())
    if arr.shape != (len(nodes), 2):
        return None
    return {v: (float(arr[i, 0]), float(arr[i, 1]))
            for i, v in enumerate(nodes)}


def save(stage: str, G, pos: dict) -> None:
    d = os.path.join(CACHE_ROOT, stage)
    os.makedirs(d, exist_ok=True)
    nodes = list(G.nodes())
    arr = np.array([pos[v] for v in nodes], dtype=float)
    path = os.path.join(d, graph_key(G) + ".npz")
    # savez appends .npz to names lacking it, so the temp name must keep it
    tmp = path[:-4] + f".{os.getpid()}.tmp.npz"
    np.savez_compressed(tmp, pos=arr)
    os.replace(tmp, path)              # atomic under parallel workers


def cached_layout(stage: str, G, compute) -> dict:
    """Load stage layout from cache, or compute() and store it."""
    pos = load(stage, G)
    if pos is None:
        pos = compute()
        save(stage, G, pos)
    return pos
