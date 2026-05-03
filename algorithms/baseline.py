"""
algorithms/baseline.py
======================
Baseline algorithm: spectral layout followed by spring/force-directed layout.
This is exactly what the paper uses as the STARTING POINT before edge relaxation.
Running this alone lets us measure how much edge relaxation actually improves things.
"""

import networkx as nx
from algorithms.base import GraphDrawingAlgorithm


class Baseline(GraphDrawingAlgorithm):
    """
    Spectral layout + Fruchterman-Reingold spring layout.

    Step 1 — Spectral layout:
        Places nodes using the eigenvectors of the graph Laplacian.
        This captures the global structure of the graph well.

    Step 2 — Spring layout (Fruchterman-Reingold):
        Refines the positions by simulating attractive forces on edges
        and repulsive forces between all nodes.
        Uses the spectral positions as starting points so it doesn't
        get stuck in a bad local minimum.
    """

    def __init__(self, seed: int = 42, iterations: int = 50):
        """
        Parameters
        ----------
        seed       : random seed for reproducibility
        iterations : number of spring layout iterations
        """
        self.seed = seed
        self.iterations = iterations

    @property
    def name(self) -> str:
        return "baseline"

    def layout(self, G: nx.Graph) -> dict:
        """Compute spectral + spring layout. Returns {node: (x, y)}."""
        return self._initial_layout(G, self.seed, self.iterations)
