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
        """Spectral → spring layout used as the starting point by most algorithms."""
        try:
            pos = nx.spectral_layout(G, weight=None)
        except Exception:
            pos = nx.random_layout(G, seed=seed)
        return nx.spring_layout(G, pos=pos, weight=None, seed=seed,
                                iterations=iterations)
