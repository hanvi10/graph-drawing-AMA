"""
algorithms2/baseline.py
=======================
Baseline for `algorithms2`: spectral layout -> ForceAtlas2 -> minimum node
separation.

This replaces the `algorithms/baseline.py` pipeline (spectral ->
Fruchterman-Reingold via `nx.spring_layout`). Everything downstream is
unchanged: the same three-stage idea the paper uses — a spectral embedding for
global structure, a force-directed refinement for even spreading — with
ForceAtlas2 as the force model and an explicit separation guarantee at the end.

    1. SPECTRAL      the two lowest non-trivial Laplacian eigenvectors as
                     coordinates. Gets relative positions right; leaves nodes
                     piled up at the boundary. Computed here rather than via
                     `nx.spectral_layout`, which is not reproducible at
                     len(G) >= 500 — see `Baseline._eigen_layout`.

    2. TIE-BREAK     a tiny separation pass, `SPECTRAL_TIE_FRAC` (0.02k). Not
                     cosmetic — ForceAtlas2's repulsion is O(1/d) and singular
                     at d = 0, and spectral layouts routinely contain duplicate
                     coordinates because structurally equivalent nodes get
                     identical eigenvector entries. Without this pass,
                     malaria_genes__HVR_3 returns all-NaN coordinates and
                     urban_streets__new-york has two nodes fired out to 5.6e7
                     while the rest of the drawing stays put. See
                     algorithms2/separation.py.

    3. FORCEATLAS2   attraction linear in distance, repulsion proportional to
                     m_i*m_j/d with mass = degree+1, plus gravity toward the
                     centroid. Degree-weighted mass is the substantive
                     difference from Fruchterman-Reingold: hubs push harder and
                     end up with room around them, which is what stops dense
                     neighbourhoods collapsing into the ink blots FR produces.

    4. SEPARATION    no two nodes closer than `min_dist_frac * k`, with
                     k = sqrt(area/n). The paper says it ran FR "with the only
                     modification of forcing a minimum distance between nodes"
                     and never says how; this is that modification, stated and
                     measurable. Default 0.30 — see `min_dist_frac` below.

ForceAtlas2 is vendored in algorithms2/forceatlas2.py because `nx.forceatlas2_
layout` needs NetworkX >= 3.4, which needs Python >= 3.10, and this venv is
Python 3.9.
"""

import math

import networkx as nx
import numpy as np

from algorithms2.base import GraphDrawingAlgorithm
from algorithms2.forceatlas2 import forceatlas2_layout
from algorithms2.separation import (SPECTRAL_TIE_FRAC, enforce_min_distance,
                                    ideal_distance, separation_report)


class Baseline(GraphDrawingAlgorithm):
    """Spectral + ForceAtlas2 + minimum node separation."""

    def __init__(
        self,
        seed: int = 42,
        max_iter: int = 500,
        min_dist_frac: float = 0.30,
        fa2_kwargs: dict = None,
    ):
        """
        Parameters
        ----------
        seed : int
            Reproducibility. Used for the random fallback layout, for
            ForceAtlas2, and for the direction chosen when separating a pair of
            exactly coincident nodes.

        max_iter : int
            ForceAtlas2 iterations. 500 rather than the upstream default of
            100: started from a spectral layout, FA2 spends its early
            iterations expanding a drawing that arrives compressed into
            [-1, 1], and at 100 iterations a third of the dataset is still
            visibly mid-expansion. FA2 stops early once the total displacement
            of a step falls below 1e-10, so on graphs that settle sooner the
            extra budget costs nothing.

        min_dist_frac : float
            THE MINIMUM DISTANCE. No two nodes end up closer than

                min_dist_frac * k,    k = sqrt(bounding box area / n)

            k is the Fruchterman-Reingold ideal distance — the typical spacing
            between nodes — so this is scale-free and graph-size independent:
            0.3 means "no two nodes closer than 30% of the typical spacing"
            whatever the graph. Set 0 to turn the step off.

            0.30 is the default because it is the largest floor that reliably
            un-stacks nodes without deforming the drawing: the area n disks of
            diameter f*k need is (pi*f^2/4) of the drawing, so 0.3 asks for 7%
            and is comfortable, while 0.5 asks for 20% and starts pushing
            clusters apart hard enough to cost crossings. It also matches the
            floor the sibling pipeline in algorithms/novel/ settled on after a
            sweep, which keeps clumping numbers comparable across the project.

            Implementation and the alternatives considered: algorithms2/
            separation.py.

        fa2_kwargs : dict, optional
            Extra keyword arguments passed through to ForceAtlas2, e.g.
            `{"scaling_ratio": 5.0}` to spread the drawing out, or
            `{"linlog": True}` for logarithmic attraction, which separates
            clusters more sharply. `weight` stays None regardless: the paper
            ignores the edge weights that ship with these graphs.
        """
        self.seed = seed
        self.max_iter = max_iter
        self.min_dist_frac = min_dist_frac
        self.fa2_kwargs = dict(fa2_kwargs or {})

    @property
    def name(self) -> str:
        return "baseline2"

    def _eigen_layout(self, G: nx.Graph) -> dict:
        """Stage 1: the Fiedler and third Laplacian eigenvectors as coordinates.

        This is what `nx.spectral_layout` computes, done here instead of called,
        because `nx.spectral_layout` is not reproducible on the larger half of
        this dataset. At `len(G) >= 500` it switches from a dense solve to
        `scipy.sparse.linalg.eigsh`, and it passes no `v0`, so ARPACK starts
        from a fresh random vector on every call and exposes no seed parameter
        to pin it. Where the graph is symmetric enough that its second and
        third eigenvalues are degenerate, any basis of that eigenspace is a
        valid answer and ARPACK returns a different one each run:
        `fullerene_structures__C500` moves by 0.78 of its own extent between two
        consecutive calls. Four graphs in data/graphs are at or above that
        threshold (C500, C540, urban_streets__bologna, urban_streets__savannah)
        and all four produced a different drawing on every run.

        `np.linalg.eigh` is the right solver regardless: the Laplacian is
        symmetric, so its eigenvalues are real and its eigenvectors orthogonal,
        and eigh returns them in ascending order without the complex arithmetic
        `np.linalg.eig` (which networkx's dense path uses) needs. It is O(n^3),
        which is irrelevant at this dataset's scale — the largest graph here is
        584 nodes — but would matter in the thousands.

        eigh is free to return either sign of an eigenvector, so the sign is
        pinned here as well: mirroring a drawing changes no metric, but it does
        change the coordinates, and a stable sign makes layouts comparable
        across runs and machines.
        """
        nodes = list(G)
        A = nx.to_numpy_array(G, nodelist=nodes, weight=None)
        L = np.diag(A.sum(axis=1)) - A
        _, vectors = np.linalg.eigh(L)

        # column 0 is the trivial constant eigenvector (eigenvalue 0); the
        # Fiedler vector and the next one are the 2-D embedding
        coords = np.array(vectors[:, [1, 2]], dtype=float)

        for axis in range(coords.shape[1]):
            column = coords[:, axis]
            extreme = column[np.argmax(np.abs(column))]
            if extreme < 0:
                coords[:, axis] = -column

        coords = nx.rescale_layout(coords, scale=1.0)
        return {v: (float(coords[i, 0]), float(coords[i, 1]))
                for i, v in enumerate(nodes)}

    def spectral(self, G: nx.Graph) -> dict:
        """Stages 1-2: spectral layout, with coincident coordinates broken apart.

        The eigendecomposition needs at least 3 nodes to have a third
        eigenvector to read, and can fail on a degenerate Laplacian; a random
        layout is the fallback, and ForceAtlas2 recovers from it perfectly well.
        """
        try:
            if G.number_of_nodes() < 3:
                raise ValueError("too few nodes for a 2-D spectral embedding")
            pos = self._eigen_layout(G)
        except Exception:
            pos = {v: (float(p[0]), float(p[1]))
                   for v, p in nx.random_layout(G, seed=self.seed).items()}
        return enforce_min_distance(pos, SPECTRAL_TIE_FRAC, seed=self.seed)

    def force(self, G: nx.Graph, pos: dict) -> dict:
        """Stage 3: ForceAtlas2, started from `pos`."""
        kwargs = dict(self.fa2_kwargs)
        kwargs.setdefault("scaling_ratio", 2.0)
        kwargs.setdefault("gravity", 1.0)
        out = forceatlas2_layout(G, pos=pos, max_iter=self.max_iter,
                                 seed=self.seed, weight=None, **kwargs)
        return {v: (float(p[0]), float(p[1])) for v, p in out.items()}

    def separate(self, G: nx.Graph, pos: dict) -> dict:
        """Stage 4: enforce the minimum distance."""
        return enforce_min_distance(pos, self.min_dist_frac, seed=self.seed)

    def layout(self, G: nx.Graph) -> dict:
        """Run all four stages. Returns {node: (x, y)}."""
        if G.number_of_nodes() == 0:
            return {}
        if G.number_of_nodes() == 1:
            return {next(iter(G)): (0.0, 0.0)}

        pos = self.spectral(G)
        pos = self.force(G, pos)

        # ForceAtlas2 can still diverge on a pathological graph (its speed
        # controller accumulates swing across iterations and never resets, so a
        # violent start is never fully forgiven). Falling back to the spectral
        # layout is far better than returning NaNs to the metrics.
        coords = np.array(list(pos.values()), dtype=float)
        if not np.isfinite(coords).all():
            pos = self.spectral(G)

        return self.separate(G, pos)

    def report(self, G: nx.Graph) -> dict:
        """Diagnostics for one graph: what each stage did to the separation.

        Useful when tuning `min_dist_frac` — it shows how clumped the drawing
        was before the floor was applied and whether the floor actually held.
        """
        spec = self.spectral(G)
        forced = self.force(G, spec)
        final = self.separate(G, forced)
        return {
            "n": G.number_of_nodes(),
            "m": G.number_of_edges(),
            "k_force": ideal_distance(np.array(list(forced.values()))),
            "before": separation_report(forced, self.min_dist_frac),
            "after": separation_report(final, self.min_dist_frac),
            "displacement_over_k": float(
                np.linalg.norm(
                    np.array([final[v] for v in G]) - np.array([forced[v] for v in G]),
                    axis=1,
                ).max() / max(ideal_distance(np.array(list(forced.values()))), 1e-12)
            ),
        }
