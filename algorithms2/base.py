"""
algorithms2/base.py
===================
Abstract base class that every algorithm in `algorithms2` inherits from.

Same contract as `algorithms/base.py` — `name` and `layout()` — so the runners,
metrics and render scripts at the repo root accept these algorithms unchanged.
What differs is the shared starting layout: `algorithms/` refines a spectral
layout with Fruchterman-Reingold (`nx.spring_layout`), `algorithms2/` refines it
with ForceAtlas2 and then enforces a minimum node separation. See
`algorithms2/baseline.py`.

To add a new algorithm here:
    1. Create a file in algorithms2/
    2. Subclass GraphDrawingAlgorithm
    3. Implement `name` and `layout()`
    4. Export it from algorithms2/__init__.py
"""

from abc import ABC, abstractmethod

import networkx as nx


class GraphDrawingAlgorithm(ABC):
    """Base class for all algorithms in `algorithms2`."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier for this algorithm (used in result CSVs)."""

    @abstractmethod
    def layout(self, G: nx.Graph) -> dict:
        """Compute 2D positions for all nodes in G.

        Returns
        -------
        pos : dict {node: (x, y)}
        """

    def _initial_layout(self, G: nx.Graph, **baseline_kwargs) -> dict:
        """The shared starting layout: spectral + ForceAtlas2 + separation.

        Subclasses that relax edges or repair crossings start from this,
        exactly as the ones in `algorithms/` start from spectral + spring.
        It carries `Baseline`'s separation floor, so the layout handed to a
        later stage is already legible: no two nodes closer than 0.30k.

        A later stage that moves nodes can of course breach the floor again —
        the guarantee belongs to the drawing it was measured on, not to the
        graph. Re-establish it on your final positions before returning:

            pos = self._initial_layout(G)            # floor holds here
            pos = self.my_relaxation(G, pos)         # ... may breach it
            return enforce_min_distance(pos, 0.30, seed=self.seed)

        Pass `min_dist_frac=0.0` to skip the floor on the starting layout when
        a stage is going to rearrange it wholesale anyway.
        """
        from algorithms2.baseline import Baseline

        baseline_kwargs.setdefault("seed", getattr(self, "seed", 42))
        return Baseline(**baseline_kwargs).layout(G)
