"""
algorithms/base.py
==================
Abstract base class that every graph drawing algorithm must inherit from.

To add a NEW algorithm:
    1. Create a new file in algorithms/  (e.g. algorithms/my_algorithm.py)
    2. Create a class that inherits from GraphDrawingAlgorithm
    3. Implement the two required methods: name and layout()
    4. Import it in algorithms/__init__.py

Example skeleton:
    from algorithms.base import GraphDrawingAlgorithm
    import networkx as nx

    class MyAlgorithm(GraphDrawingAlgorithm):
        @property
        def name(self) -> str:
            return "my_algorithm"

        def layout(self, G: nx.Graph) -> dict:
            # ... compute positions ...
            return pos   # {node: (x, y)}
"""

from abc import ABC, abstractmethod
import networkx as nx


class GraphDrawingAlgorithm(ABC):
    """
    Base class for all graph drawing algorithms.
    Subclasses must implement `name` and `layout()`.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """A short identifier for this algorithm (used in result CSVs)."""
        pass

    @abstractmethod
    def layout(self, G: nx.Graph) -> dict:
        """
        Compute 2D positions for all nodes in G.

        Returns
        -------
        pos : dict {node: (x, y)}
        """
        pass

    def _initial_layout(self, G: nx.Graph, seed: int, iterations: int) -> dict:
        """
        The Baseline's spectral → ForceAtlas2 layout, used as the starting
        point by the edge relaxation algorithms (EBC, currentflow). It is in
        ForceAtlas2's own units with no min separation; callers finish with
        Baseline.finalize(), so separation is imposed once, at the end.
        `iterations` is the number of ForceAtlas2 iterations (Baseline's
        default is 300).
        """
        from algorithms.baseline import Baseline   # deferred: baseline imports base
        return Baseline(seed=seed, iterations=iterations).fa2_layout(G)
