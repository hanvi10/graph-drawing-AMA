"""
algorithms/novel/vpsc.py
========================
VPSC + FNOR: minimal-displacement node overlap removal.

Implements the separation-constraint solver and overlap-removal pass of
Dwyer, Marriott & Stuckey, "Fast Node Overlap Removal" (GD 2005):

  * solve_vpsc(desired, weights, constraints) — the 1-D Variable Placement with
    Separation Constraints problem: place points to minimise
    sum_i w_i (x_i - desired_i)^2 subject to x_right - x_left >= gap, via the
    block-merge active-set method. Blocks of variables joined by tight (active)
    constraints move together, each block sitting at its members' weighted
    desired position, which is what makes the solution minimal-displacement.

  * remove_overlaps(coords, sep) — model each node as a square of side `sep`
    and remove all square overlaps by an X pass then a Y pass of VPSC. Two
    non-overlapping side-`sep` squares are >= sep apart in at least one axis, so
    Euclidean centre distance >= sep: the node-node floor holds by construction.

Minimal displacement is the whole reason to prefer this over a stress-based
overlap remover (PRISM): it disturbs the input layout as little as the floor
allows, so a crossing-minimised layout keeps its crossings.
"""

import numpy as np


class _Var:
    __slots__ = ("d", "w", "offset", "block")

    def __init__(self, d, w=1.0):
        self.d = d
        self.w = w
        self.offset = 0.0
        self.block = None

    def pos(self):
        return self.block.posn + self.offset


class _Block:
    __slots__ = ("vars", "posn", "wsum")

    def __init__(self, v):
        self.vars = [v]
        v.block = self
        v.offset = 0.0
        self.wsum = v.w
        self.posn = v.d

    def update(self):
        self.wsum = sum(v.w for v in self.vars)
        self.posn = sum(v.w * (v.d - v.offset) for v in self.vars) / self.wsum


class _Con:
    __slots__ = ("l", "r", "gap", "active")

    def __init__(self, l, r, gap):
        self.l = l
        self.r = r
        self.gap = gap
        self.active = False

    def viol(self):
        return self.l.pos() + self.gap - self.r.pos()


def _merge(L, R, c):
    """Merge block R into L making constraint c tight."""
    shift = (c.l.offset + c.gap) - c.r.offset   # offset R needs in L's frame
    for v in R.vars:
        v.offset += shift
        v.block = L
    L.vars.extend(R.vars)
    L.update()
    c.active = True


def solve_vpsc(desired, weights, constraints, tol=1e-9):
    """1-D VPSC. `constraints` = list of (left_idx, right_idx, gap). Returns
    positions minimising weighted squared displacement, feasible under the
    constraints (merge-to-satisfy; blocks sit at weighted desired)."""
    n = len(desired)
    vs = [_Var(float(desired[i]), float(weights[i])) for i in range(n)]
    C = [_Con(vs[l], vs[r], gap) for (l, r, gap) in constraints]
    for v in vs:
        _Block(v)
    if not C:
        return np.array([v.pos() for v in vs])

    remaining = C[:]
    # each merge reduces the block count by 1, so at most n-1 merges; the
    # same-block-implied case just retires a constraint, bounded by len(C)
    for _ in range(n - 1 + len(C)):
        best, bv = None, tol
        for c in remaining:
            if c.active:
                continue
            v = c.viol()
            if v > bv:
                bv, best = v, c
        if best is None:
            break
        L, R = best.l.block, best.r.block
        if L is R:
            best.active = True          # already implied by the block geometry
            continue
        _merge(L, R, best)
    return np.array([v.pos() for v in vs])


def remove_overlaps(coords, sep, passes=3):
    """Remove side-`sep` square overlaps by FNOR (X pass then Y pass), repeated
    a few times (a Y move can re-open an X overlap on tangled inputs)."""
    coords = coords.astype(float).copy()
    n = len(coords)
    w = np.ones(n)

    def constraints(axis):
        """Separate along `axis` every pair whose squares overlap; orient
        left->right by current `axis` coordinate."""
        other = 1 - axis
        a, o = coords[:, axis], coords[:, other]
        cons = []
        order = np.argsort(a)
        for ii in range(n):
            i = order[ii]
            for jj in range(ii + 1, n):
                j = order[jj]
                if a[j] - a[i] >= sep:
                    break                      # sorted: no further j overlaps i
                if abs(o[i] - o[j]) < sep:      # squares overlap in the other axis
                    cons.append((i, j, sep))    # i is left (a[i] <= a[j])
        return cons

    for _ in range(passes):
        moved = False
        for axis in (0, 1):
            cons = constraints(axis)
            if not cons:
                continue
            new = solve_vpsc(coords[:, axis], w, cons)
            if not np.allclose(new, coords[:, axis]):
                moved = True
            coords[:, axis] = new
        if not moved:
            break
    return coords
