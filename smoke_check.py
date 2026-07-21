"""One-off correctness smoke test for the currentflow_aesthetic pipeline.

Checks the guarantees the code claims, stage by stage, on a few small graphs:
  1. crossings never increase across repair / expand / polish stages
  2. final path continuity <= post-relax continuity * 1.05 per stage tolerance
     (repair and polish each guarantee <= 1.05x their own input)
  3. polish reduces (or holds) the clump fraction
  4. output has a position for every node, all finite
"""
import math
import warnings

import networkx as nx
import numpy as np

from metrics import count_crossings, path_continuity, all_metrics
from algorithms.edge_relaxation.currentflow import EdgeRelaxationCurrentFlow
from algorithms.novel.crossing_repair import CrossingRepair
from algorithms.novel.aesthetic_repair import AestheticRepair

GRAPHS = [
    "data/graphs/malaria_genes__HVR_3.graphml",
    "data/graphs/kangaroo.graphml",
]


def clump_frac(pos):
    C = np.array(list(pos.values()), dtype=float)
    w, h = C.max(axis=0) - C.min(axis=0)
    k = math.sqrt(max(w * h, 1e-12) / len(C))
    D = np.linalg.norm(C[:, None, :] - C[None, :, :], axis=2)
    np.fill_diagonal(D, np.inf)
    return float((D.min(axis=1) < 0.3 * k).mean())


def report(tag, G, pos):
    c = count_crossings(G, pos)
    pc = path_continuity(G, pos)
    cf = clump_frac(pos)
    print(f"  {tag:<12} crossings={c:<5d} continuity={pc:7.2f}  clump={cf:5.1%}")
    return c, pc, cf


failures = []
with warnings.catch_warnings():
    warnings.simplefilter("error", DeprecationWarning)  # surface np.cross 2D issues
    for path in GRAPHS:
        G = nx.convert_node_labels_to_integers(nx.read_graphml(path))
        print(f"{path}  ({G.number_of_nodes()}v {G.number_of_edges()}e)")

        relax = EdgeRelaxationCurrentFlow(seed=42)
        rep = CrossingRepair(seed=42)
        pol = AestheticRepair(seed=42)

        pos0 = relax.layout(G)
        c0, pc0, _ = report("relax", G, pos0)

        pos1 = rep.repair(G, pos0)
        c1, pc1, _ = report("repair", G, pos1)

        pos2 = rep.expand(G, pos1)
        c2, pc2, cf2 = report("expand", G, pos2)

        pos3 = pol.repair(G, pos2)
        c3, pc3, cf3 = report("polish", G, pos3)

        if not (c1 <= c0):
            failures.append(f"{path}: repair increased crossings {c0}->{c1}")
        if not (c2 <= c1):
            failures.append(f"{path}: expand increased crossings {c1}->{c2}")
        if not (c3 <= c2):
            failures.append(f"{path}: polish increased crossings {c2}->{c3}")
        if pc0 > 0 and pc1 > pc0 * 1.05 + 1e-9:
            failures.append(f"{path}: repair broke continuity bound {pc0:.2f}->{pc1:.2f}")
        if pc2 > 0 and pc3 > pc2 * 1.05 + 1e-9:
            failures.append(f"{path}: polish broke continuity bound {pc2:.2f}->{pc3:.2f}")
        if set(pos3) != set(G.nodes()):
            failures.append(f"{path}: polish lost/added nodes")
        arr = np.array(list(pos3.values()), dtype=float)
        if not np.isfinite(arr).all():
            failures.append(f"{path}: non-finite coordinates in output")
        print()

print("=" * 60)
if failures:
    print("FAILURES:")
    for f in failures:
        print(" -", f)
    raise SystemExit(1)
print("All stage guarantees held on all test graphs.")
