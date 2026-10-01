"""
algorithms/baseline.py
======================
Baseline algorithm: spectral layout followed by ForceAtlas2, then a minimum
node separation pass.
Same pipeline as the paper's starting point, but with ForceAtlas2 (networkx)
in place of the Fruchterman-Reingold spring layout.
Running this alone lets us measure how much edge relaxation actually improves things.
"""

import warnings

import networkx as nx
import numpy as np
from scipy.spatial import cKDTree
from algorithms.base import GraphDrawingAlgorithm

JITTER = 1e-3
MAX_SEP_ROUNDS = 1000


def _dense_spectral_layout(G: nx.Graph) -> dict:
    """
    nx.spectral_layout(G, weight=None), but always with the dense solver.

    For n >= 500 networkx switches to scipy's ARPACK (eigsh), which starts
    from an unseeded random vector, so the layout -- and everything FA2
    builds on it -- changes from run to run. This mirrors networkx's own
    dense path (used for n < 500), so smaller graphs get identical results.
    """
    if len(G) <= 2:
        return nx.spectral_layout(G, weight=None)
    A = nx.to_numpy_array(G, weight=None)
    L = np.identity(len(A)) * A.sum(axis=1) - A
    eigenvalues, eigenvectors = np.linalg.eig(L)
    index = np.argsort(eigenvalues)[1:3]    # skip the zero eigenvalue
    P = nx.rescale_layout(np.real(eigenvectors[:, index]))
    return dict(zip(G, P))


class Baseline(GraphDrawingAlgorithm):
    """
    Spectral layout + ForceAtlas2 + minimum node separation.

    Step 1 — Spectral layout:
        Places nodes using the eigenvectors of the graph Laplacian.
        This captures the global structure of the graph well.
        Always solved densely, so the result is reproducible for any n.

    Step 2 — ForceAtlas2 (nx.forceatlas2_layout):
        Refines the positions with degree-weighted repulsion between all
        nodes, attraction along edges, and gravity toward the centre.
        Uses the spectral positions as starting points so it doesn't
        get stuck in a bad local minimum.

    Step 3 — Minimum separation:
        Guarantees no two nodes are closer than sep_frac * k, where
        k = sqrt(bounding-box area / n) is the average spacing per node.
        Pairs below the floor are pushed apart along the line joining them,
        each by half the shortfall, until no pair is too close. Moves are a
        fraction of k, so the overall shape is unchanged.

    Step 4 — Rescale:
        ForceAtlas2 returns coordinates in arbitrary units, so the result is
        centred and scaled to [-1, 1] — the same normalisation spring_layout
        applies — keeping length metrics comparable with the other algorithms.
        Rescaling multiplies distances and k alike, so the floor still holds.
    """

    def __init__(self, seed: int = 42, iterations: int = 300,
                 sep_frac: float = 0.30):
        """
        Parameters
        ----------
        seed       : random seed for reproducibility
        iterations : number of ForceAtlas2 iterations (max_iter)
        sep_frac   : minimum node distance as a fraction of k (0 disables)
        """
        self.seed = seed
        self.iterations = iterations
        self.sep_frac = sep_frac

    @property
    def name(self) -> str:
        return "baseline"

    def layout(self, G: nx.Graph) -> dict:
        """Compute spectral + ForceAtlas2 + separated layout. Returns {node: (x, y)}."""
        try:
            pos = _dense_spectral_layout(G)
        except Exception:
            pos = nx.random_layout(G, seed=self.seed)
        # Spectral layout puts structurally equivalent nodes (e.g. leaves on
        # the same hub) at identical positions, and networkx's ForceAtlas2
        # divides by their zero distance -> NaN. A small seeded jitter
        # (~0.05% of the [-1, 1] spectral extent) separates them.
        rng = np.random.default_rng(self.seed)
        pos = {v: np.asarray(xy) + rng.uniform(-JITTER, JITTER, 2)
               for v, xy in pos.items()}
        pos = nx.forceatlas2_layout(G, pos=pos, max_iter=self.iterations,
                                    weight=None, seed=self.seed)
        pos = self._separate(pos, rng)
        return nx.rescale_layout_dict(pos, scale=1)

    def _separate(self, pos: dict, rng) -> dict:
        """Push node pairs apart until all are >= sep_frac * k from each other."""
        nodes = list(pos)
        P = np.array([pos[v] for v in nodes], dtype=float)
        if len(nodes) < 2 or self.sep_frac <= 0:
            return pos

        for _ in range(MAX_SEP_ROUNDS):
            # pushes grow the bounding box, so k is re-measured every round
            w, h = np.ptp(P, axis=0)
            k = np.sqrt(max(w * h, 1e-12) / len(nodes))
            d_min = self.sep_frac * k
            target = d_min * 1.001   # small overshoot so a pushed pair isn't re-flagged by rounding
            pairs = cKDTree(P).query_pairs(d_min, output_type="ndarray")
            if len(pairs) == 0:
                break
            # Gauss-Seidel: each push sees the moves made before it this round
            for i, j in pairs:
                delta = P[j] - P[i]
                d = np.hypot(*delta)
                if d >= d_min:
                    continue
                if d == 0:   # coincident: no direction, pick a random one
                    delta = rng.normal(size=2)
                    d = np.hypot(*delta)
                shift = (target - np.hypot(*(P[j] - P[i]))) / 2 * delta / d
                P[i] -= shift
                P[j] += shift
        else:
            warnings.warn(f"min separation not reached after {MAX_SEP_ROUNDS} rounds")

        return {v: P[idx] for idx, v in enumerate(nodes)}
