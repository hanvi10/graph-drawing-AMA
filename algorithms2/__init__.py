"""
algorithms2
===========
Second generation of the graph drawing pipeline.

`algorithms/` builds every layout on spectral + Fruchterman-Reingold
(`nx.spring_layout`), following the paper. This package rebuilds the starting
point on spectral + ForceAtlas2, and makes the paper's undocumented "minimum
distance between nodes" an explicit, measurable parameter.

    from algorithms2 import Baseline
    pos = Baseline(min_dist_frac=0.30).layout(G)

Modules
-------
baseline.py      the four-stage pipeline (spectral, tie-break, FA2, separation)
separation.py    the minimum-distance step, and why it is built this way
forceatlas2.py   nx.forceatlas2_layout vendored, because it needs NetworkX 3.4
                 and this venv is Python 3.9
base.py          the GraphDrawingAlgorithm contract, shared with algorithms/
"""

from .base import GraphDrawingAlgorithm
from .baseline import Baseline
from .separation import (SPECTRAL_TIE_FRAC, enforce_min_distance,
                         ideal_distance, separation_report)

__all__ = [
    "Baseline",
    "GraphDrawingAlgorithm",
    "enforce_min_distance",
    "separation_report",
    "ideal_distance",
    "SPECTRAL_TIE_FRAC",
]
