"""
algorithms2/separation.py
=========================
Minimum node separation: the post-processing step that guarantees no two nodes
in a drawing sit closer than a stated fraction of the typical node spacing.

WHY A DRAWING NEEDS ONE
-----------------------
ForceAtlas2, like Fruchterman-Reingold, treats nodes as dimensionless points.
Its repulsion term keeps most pairs apart, but nothing in the model forbids two
points from landing on top of each other, and several situations push them
there: structurally equivalent nodes (identical neighbourhoods) feel identical
forces, leaves hanging off one hub are symmetric with each other, and FA2's
adaptive speed control damps exactly the high-swing nodes that a tight cluster
produces — so a cluster can freeze before it finishes spreading.

Coincident nodes are not a cosmetic problem. A reader cannot count them, cannot
trace an edge into them, and cannot tell a 2-node stack from a 20-node stack.
The paper this project follows says only that it ran Fruchterman-Reingold
"with the only modification of forcing a minimum distance between nodes" and
never says how. This module is that modification, made explicit.

There is a second, harder reason. FA2's repulsion between two nodes has
magnitude `scaling_ratio * m_i * m_j / d`, which is singular at d = 0. Handing
it an initial layout with duplicate coordinates does not merely look bad, it
destroys the run: see SPECTRAL_TIE_FRAC below.

WHAT UNIT THE FLOOR IS MEASURED IN
----------------------------------
The floor is a fraction of

    k = sqrt(W * H / n)                    (`ideal_distance`)

the Fruchterman-Reingold ideal distance: the side of the square each node would
own if n nodes were spread evenly over the drawing's W x H bounding box. So
`min_dist_frac=0.3` reads as "no two nodes closer than 30% of the typical node
spacing", and means the same thing on a 50-node graph as on a 1000-node one.

Three alternatives were rejected:

  * An absolute distance. FA2's output scale is arbitrary and varies by orders
    of magnitude between graphs, so a fixed number is meaningless.
  * A fraction of the bounding-box diagonal. Ignores n: 1% of the diagonal is
    enormous on a 1000-node graph and invisible on a 50-node one.
  * A fraction of the mean edge length. Tempting, since it is what the eye
    compares against, but it is exactly the wrong choice for this project. Edge
    relaxation deliberately produces a heavy-tailed edge length distribution —
    lengthening a few bridges to shorten everything else is the paper's whole
    thesis — so a handful of relaxed bridges would inflate the unit and
    over-separate the entire drawing. k depends only on area and count, both
    stable.

k is also the unit the rest of this repo already reports clumping in, so the
numbers are directly comparable with `algorithms/novel/cf_cross_sep.py`.

HOW MUCH SEPARATION IS FEASIBLE
-------------------------------
n disks of diameter f*k need area n * pi * (f*k/2)^2 = (pi*f^2/4) * W * H, i.e.
a fraction pi*f^2/4 of the drawing. f=0.3 asks for 7% of the area and is easy;
f=0.5 asks for 20%; f=1.0 asks for 79%, above what a non-uniform layout can
give. So the usable range is roughly f <= 0.5, and the cost of raising f inside
that range is that more of the drawing has to deform to make room.

HOW THE FLOOR IS ENFORCED
-------------------------
Local constraint projection. Repeatedly find the pairs that violate the floor
and push each pair apart along the line joining them, each node taking half the
deficit; repeat until no pair violates it. This is the minimal-displacement
repair: a node that already satisfies the floor never moves, and a node that
violates it moves only as far as the violation requires. The drawing FA2 found
survives; only the illegible parts of it change.

Alternatives, and why not:

  * Scale the whole drawing up until the closest pair clears the floor. A
    similarity transform preserves the drawing perfectly, but it cannot work
    here: both distances and k scale together, so the ratio min_dist/k this
    module targets is scale-invariant and uniform scaling changes nothing. (A
    coincident pair would also demand an infinite factor.)
  * Let FA2 do it, via its `node_size` argument. In-model and appealing, but it
    is only a soft bias with no guarantee, and the upstream implementation
    computes `distance += -size[i] - size[j]` without clamping, so overlapping
    halos produce a negative distance whose square is positive — the repulsion
    silently reverses sign and pulls the pair together.
  * PRISM (Gansner-Hu) and VPSC (Dwyer), the two standard overlap-removal
    algorithms. Both were implemented and benchmarked for the sibling pipeline
    in `algorithms/novel/` and both lost to local projection: PRISM's stress
    objective is not crossing-aware and shreds near-planar layouts, and VPSC's
    axis-aligned constraints fan a packed hub into a grid that crosses its own
    spokes. Their advantage — a globally optimal rearrangement — is worth
    paying for only when the whole layout overlaps, not when a few pairs do.

Convergence is not proved, it is measured: `separation_report` states what the
returned drawing actually achieves, and the benchmark records it per graph.
"""

import math

import numpy as np
from scipy.spatial import cKDTree

__all__ = [
    "ideal_distance",
    "enforce_min_distance",
    "separation_report",
    "SPECTRAL_TIE_FRAC",
]


# Floor applied to the spectral layout BEFORE ForceAtlas2 runs. This one is not
# about aesthetics, it is about the run finishing at all.
#
# A spectral layout routinely contains duplicate coordinates: structurally
# equivalent nodes have identical entries in every eigenvector, and coordinates
# outside the leading eigenvectors' support collapse to the same value to within
# floating-point noise. Two observed cases from data/graphs:
#
#   malaria_genes__HVR_3      one exactly coincident pair (d = 0)
#       -> repulsion factor m_i*m_j/d^2 = inf, forces become NaN, and every
#          coordinate in the drawing is NaN by the end of iteration 1.
#   urban_streets__new-york   closest pair 1.9e-15 apart (1.6e-14 * k)
#       -> repulsion ~1e16, those two nodes are launched to |x| ~ 5.6e7 on the
#          first step. Nothing else moves much, so after the unit-box rescaling
#          every metric is computed on what is visually a single dot.
#
# 0.02k is chosen to be small enough to leave the spectral layout's real
# structure intact — its congestion at the boundary is genuine and is what FA2
# is supposed to start from — while bounding repulsion at 50 * m_i*m_j/k, which
# is the same order as the rest of the forces in the first iteration.
SPECTRAL_TIE_FRAC = 0.02

_TINY = 1e-12

# Pairs are pushed to `min_dist * (1 + _MARGIN)` rather than to exactly
# `min_dist`. A pair parked exactly on the floor is still returned by
# `query_pairs(r=min_dist)`, which tests `<= r`, so without the margin every
# separated pair reappears in the next sweep carrying a zero-length correction:
# the projection would never report convergence and would burn all its passes
# moving nothing. The margin puts settled pairs strictly outside the query
# radius, so sweeps end when the work is done and the floor holds strictly.
_MARGIN = 1e-6


def _as_array(pos, order=None):
    """{node: (x, y)} -> (keys, (n, 2) float array)."""
    keys = list(pos) if order is None else list(order)
    return keys, np.array([pos[v] for v in keys], dtype=float)


def ideal_distance(coords) -> float:
    """Fruchterman-Reingold ideal distance k = sqrt(area / n) of a point set.

    The typical spacing between nodes, and the unit every floor in this module
    is expressed in. Two fallbacks keep it meaningful on degenerate layouts:
    a perfectly flat (collinear) layout has zero area, so its spacing is taken
    along its one populated axis; a layout whose points are all identical has
    no scale at all, so k = 1 and the floor becomes an absolute distance.
    """
    coords = np.asarray(coords, dtype=float)
    n = len(coords)
    if n == 0:
        return 1.0
    w, h = coords.max(axis=0) - coords.min(axis=0)
    if w > 0 and h > 0:
        return math.sqrt(w * h / n)
    span = max(w, h)
    if span > 0:
        return span / n                 # collinear: each node owns span/n
    return 1.0                          # all points identical: no scale to read


def _random_units(rng, count):
    theta = rng.uniform(0.0, 2.0 * math.pi, size=count)
    return np.column_stack((np.cos(theta), np.sin(theta)))


def _project(coords, min_dist, rng, max_passes, step_cap):
    """Push every pair closer than `min_dist` apart, in place.

    Returns (passes_used, violations_remaining). Each pass is one Jacobi
    sweep: all violating pairs are found against the same positions, their
    corrections are summed per node, and the sum is applied at once. A node
    picking up corrections from several pairs can overshoot, so its total
    displacement is capped at `step_cap * min_dist` and the next pass cleans
    up whatever the cap left behind.
    """
    if min_dist <= 0 or len(coords) < 2:
        return 0, 0

    target = min_dist * (1.0 + _MARGIN)
    for used in range(1, max_passes + 1):
        pairs = cKDTree(coords).query_pairs(r=min_dist, output_type="ndarray")
        if len(pairs) == 0:
            return used - 1, 0

        i, j = pairs[:, 0], pairs[:, 1]
        delta = coords[i] - coords[j]
        d = np.linalg.norm(delta, axis=1)

        # Direction to separate along. Undefined for a coincident pair, so pick
        # one from the seeded generator — the caller's seed keeps it reproducible.
        unit = np.empty_like(delta)
        apart = d > _TINY
        unit[apart] = delta[apart] / d[apart, None]
        n_stacked = int((~apart).sum())
        if n_stacked:
            unit[~apart] = _random_units(rng, n_stacked)

        half = (0.5 * (target - d))[:, None] * unit
        disp = np.zeros_like(coords)
        np.add.at(disp, i, half)        # unbuffered: a node can appear in many pairs
        np.add.at(disp, j, -half)

        length = np.linalg.norm(disp, axis=1)
        scale = np.minimum(1.0, step_cap * min_dist / np.maximum(length, _TINY))
        coords += disp * scale[:, None]

    remaining = len(cKDTree(coords).query_pairs(r=min_dist, output_type="ndarray"))
    return max_passes, int(remaining)


def enforce_min_distance(pos, min_dist_frac=0.30, *, seed=42,
                         max_passes=200, k_updates=16, step_cap=1.0):
    """Separate nodes until none are closer than `min_dist_frac * k`.

    Parameters
    ----------
    pos : dict {node: (x, y)}
        Layout to repair. Not modified.
    min_dist_frac : float
        Floor as a fraction of the ideal distance k = sqrt(area / n).
        0 disables the step and returns `pos` unchanged. See the module
        docstring for why fractions above ~0.5 are not achievable.
    seed : int
        Seeds the directions chosen for exactly coincident pairs, so the
        result is reproducible.
    max_passes : int
        Cap on projection sweeps per k estimate. The heaviest graph in
        data/graphs (london_transport, 369 nodes with 98% of them clumped out
        of a spectral start) needs ~100 sweeps in its first round.
    k_updates : int
        How many times k is re-measured. Separating nodes grows the bounding
        box, which grows k, which raises the floor — so projecting once at the
        input layout's k lands under the target when the target is re-measured
        on the output. Each round re-measures k and re-projects, and the loop
        exits as soon as a round finds nothing left to do, which is the
        verified condition "no pair violates the floor at this layout's own k".

        Most graphs need 2-3 rounds, because un-clumping interior nodes barely
        moves the convex hull. The badly clumped ones need more: the
        20-graph trace had london_transport at 8, interactome_pdz and
        celegans_interactomes__BPmaps at 6. 16 leaves headroom above the worst
        case observed and costs nothing when the loop exits early.
    step_cap : float
        Per-node displacement limit in one sweep, in units of `min_dist`.

    Returns
    -------
    dict {node: (x, y)}
    """
    if min_dist_frac <= 0 or len(pos) < 2:
        return dict(pos)

    keys, coords = _as_array(pos)
    rng = np.random.default_rng(seed)

    for _ in range(max(1, k_updates)):
        min_dist = min_dist_frac * ideal_distance(coords)
        passes, _ = _project(coords, min_dist, rng, max_passes, step_cap)
        if passes == 0:
            # Nothing violated the floor measured on these exact coordinates,
            # and nothing moved, so this k is the returned drawing's own k and
            # the floor holds against it. That is the guarantee; stop here.
            break

    return {v: (float(coords[idx, 0]), float(coords[idx, 1]))
            for idx, v in enumerate(keys)}


def separation_report(pos, min_dist_frac=0.30) -> dict:
    """Measure what a drawing actually achieves, rather than what was asked for.

    Returns
    -------
    dict with
        k                 : ideal distance of this layout
        min_dist          : smallest distance between any two nodes
        min_dist_over_k   : that distance in units of k — compare to
                            `min_dist_frac` to check the floor held
        violating_pairs   : pairs still closer than the floor
        clumped_frac      : fraction of nodes whose nearest neighbour is inside
                            the floor (how much of the drawing is affected)
        satisfied         : True when no pair violates the floor
    """
    _, coords = _as_array(pos)
    n = len(coords)
    k = ideal_distance(coords)
    if n < 2:
        return {"k": k, "min_dist": math.inf, "min_dist_over_k": math.inf,
                "violating_pairs": 0, "clumped_frac": 0.0, "satisfied": True}

    tree = cKDTree(coords)
    nearest = tree.query(coords, k=2)[0][:, 1]      # [0] is the node itself
    floor = min_dist_frac * k
    # A pair parked exactly on the floor is a success, not a violation, so the
    # test is relaxed by a rounding-sized amount before being called a breach.
    strict = floor * (1.0 - 1e-9)
    violating = len(tree.query_pairs(r=strict, output_type="ndarray")) if floor > 0 else 0
    return {
        "k": k,
        "min_dist": float(nearest.min()),
        "min_dist_over_k": float(nearest.min() / k) if k > 0 else math.inf,
        "violating_pairs": int(violating),
        "clumped_frac": float((nearest < strict).mean()),
        "satisfied": violating == 0,
    }
