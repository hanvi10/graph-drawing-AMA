"""
algorithms2/forceatlas2.py
==========================
`nx.forceatlas2_layout`, vendored.

WHY THIS FILE EXISTS
--------------------
`forceatlas2_layout` was added in NetworkX 3.4, which requires Python >= 3.10.
This project's venv is Python 3.9.6 with NetworkX 3.2.1, and pip refuses every
networkx >= 3.3 on that interpreter:

    ERROR: Ignored the following versions that require a different python
    version: 3.4 Requires-Python >=3.10 ...

Rather than rebuild the environment (which would invalidate `data/layout_cache/`
— those layouts are only reproducible inside one numerical environment) the
upstream implementation is copied here verbatim. It is pure numpy + networkx
calls that all exist in 3.2.1, so it runs unchanged.

Source: networkx/drawing/layout.py at tag `networkx-3.7`
        https://github.com/networkx/networkx/blob/networkx-3.7/networkx/drawing/layout.py
NetworkX is 3-clause BSD licensed; see the NetworkX LICENSE.txt for terms.

The 3.7 revision is used rather than the 3.4 one that first shipped the
function, because 3.4 has three defects that hit this project directly:

  * gravity pulled toward the coordinate origin instead of the layout's
    centroid. A spectral layout is not centred on the origin, so on 3.4
    gravity drags the whole drawing sideways.
  * a node sitting exactly at the origin produced 0/0 -> NaN gravity, which
    then propagates to every node through the next repulsion step. 3.7 guards
    it with `nan_to_num`.
  * the convergence test was `abs(sum(update))`, which fires when opposing
    displacements cancel — a layout still in motion could stop on iteration 2.
    3.7 uses `sum(abs(update))`.

3.7 also drops the `dissuade_hubs` argument, which 3.4 accepted and then never
read (it appears nowhere in the 3.4 function body).

The only edits to the upstream body are marked `# VENDORED:` and are limited to
import plumbing — no change to the numerics.
"""

import networkx as nx
import numpy as np
from networkx.utils import np_random_state

__all__ = ["forceatlas2_layout"]


@np_random_state("seed")
def forceatlas2_layout(
    G,
    pos=None,
    *,
    max_iter=100,
    jitter_tolerance=1.0,
    scaling_ratio=2.0,
    gravity=1.0,
    distributed_action=False,
    strong_gravity=False,
    node_mass=None,
    node_size=None,
    weight=None,
    linlog=False,
    seed=None,
    dim=2,
    store_pos_as=None,
):
    """Position nodes using the ForceAtlas2 force-directed layout algorithm.

    This function applies the ForceAtlas2 layout algorithm [1]_ to a NetworkX graph,
    positioning the nodes in a way that visually represents the structure of the graph.
    The algorithm uses physical simulation to minimize the energy of the system,
    resulting in a more readable layout.

    Parameters
    ----------
    G : nx.Graph
        A NetworkX graph to be laid out.
    pos : dict or None, optional
        Initial positions of the nodes. If None, random initial positions are used.
    max_iter : int (default: 100)
        Number of iterations for the layout optimization.
    jitter_tolerance : float (default: 1.0)
        Controls the tolerance for adjusting the speed of layout generation.
    scaling_ratio : float (default: 2.0)
        Determines the scaling of attraction and repulsion forces.
    gravity : float (default: 1.0)
        Determines the amount of attraction on nodes to the center. Prevents islands
        (i.e. weakly connected or disconnected parts of the graph) from drifting out.
    distributed_action : bool (default: False)
        Distributes the attraction force evenly among nodes.
    strong_gravity : bool (default: False)
        Applies a strong gravitational pull towards the center.
    node_mass : dict or None, optional
        Maps nodes to their masses, influencing the attraction to other nodes.
    node_size : dict or None, optional
        Maps nodes to their sizes, preventing crowding by creating a halo effect.
    weight : string or None, optional (default: None)
        The edge attribute that holds the numerical value used for the edge weight.
        If None, then all edge weights are 1.
    linlog : bool (default: False)
        Uses logarithmic attraction instead of linear.
    seed : int, RandomState instance or None  optional (default=None)
        Used only for the initial positions in the algorithm.
        Set the random state for deterministic node layouts.
    dim : int (default: 2)
        Sets the dimensions for the layout. Ignored if `pos` is provided.
    store_pos_as : str, optional
        If non-None, the position of each node will be stored on the graph as
        an attribute with this string as its name.

    Returns
    -------
    pos : dict
        A dictionary of positions keyed by node.

    References
    ----------
    .. [1] Jacomy, M., Venturini, T., Heymann, S., & Bastian, M. (2014).
           ForceAtlas2, a continuous graph layout algorithm for handy network
           visualization designed for the Gephi software. PloS one, 9(6), e98679.
           https://doi.org/10.1371/journal.pone.0098679
    """
    # VENDORED: upstream does `import numpy as np` here; it is a module-level
    # import in this file instead.

    if len(G) == 0:
        return {}
    # parse optional pos positions
    if pos is None:
        pos = nx.random_layout(G, dim=dim, seed=seed)
        pos_arr = np.array(list(pos.values()))
    elif len(pos) == len(G):
        pos_arr = np.array([pos[node] for node in G])
    else:
        # set random node pos within the initial pos values
        pos_init = np.array(list(pos.values()))
        max_pos = pos_init.max(axis=0)
        min_pos = pos_init.min(axis=0)
        dim = max_pos.size
        pos_arr = min_pos + seed.rand(len(G), dim) * (max_pos - min_pos)
        for idx, node in enumerate(G):
            if node in pos:
                pos_arr[idx] = pos[node].copy()

    mass = np.zeros(len(G))
    size = np.zeros(len(G))

    # Only adjust for size when the users specifies size other than default (1)
    adjust_sizes = False
    if node_size is None:
        node_size = {}
    else:
        adjust_sizes = True

    if node_mass is None:
        node_mass = {}

    for idx, node in enumerate(G):
        mass[idx] = node_mass.get(node, G.degree(node) + 1)
        size[idx] = node_size.get(node, 1)

    n = len(G)
    gravities = np.zeros((n, dim))
    attraction = np.zeros((n, dim))
    repulsion = np.zeros((n, dim))
    A = nx.to_numpy_array(G, weight=weight)

    def estimate_factor(n, swing, traction, speed, speed_efficiency, jitter_tolerance):
        """Computes the scaling factor for the force in the ForceAtlas2 layout
        algorithm.

        This helper function adjusts the speed and efficiency of the layout
        generation based on the current state of the system, such as the number
        of nodes, current swing, and traction forces.
        """
        # estimate jitter
        opt_jitter = 0.05 * np.sqrt(n)
        min_jitter = np.sqrt(opt_jitter)
        max_jitter = 10
        min_speed_efficiency = 0.05

        other = min(max_jitter, opt_jitter * traction / n**2)
        jitter = jitter_tolerance * max(min_jitter, other)

        if swing / traction > 2.0:
            if speed_efficiency > min_speed_efficiency:
                speed_efficiency *= 0.5
            jitter = max(jitter, jitter_tolerance)
        if swing == 0:
            target_speed = np.inf
        else:
            target_speed = jitter * speed_efficiency * traction / swing

        if swing > jitter * traction:
            if speed_efficiency > min_speed_efficiency:
                speed_efficiency *= 0.7
        elif speed < 1000:
            speed_efficiency *= 1.3

        max_rise = 0.5
        speed = speed + min(target_speed - speed, max_rise * speed)
        return speed, speed_efficiency

    speed = 1
    speed_efficiency = 1
    swing = 1
    traction = 1
    for _ in range(max_iter):
        # compute pairwise difference
        diff = pos_arr[:, None] - pos_arr[None]
        # compute pairwise distance
        distance = np.linalg.norm(diff, axis=-1)

        # linear attraction
        if linlog:
            attraction = -np.log(1 + distance) / distance
            np.fill_diagonal(attraction, 0)
            attraction = np.einsum("ij, ij -> ij", attraction, A)
            attraction = np.einsum("ijk, ij -> ik", diff, attraction)

        else:
            attraction = -np.einsum("ijk, ij -> ik", diff, A)

        if distributed_action:
            attraction /= mass[:, None]

        # repulsion
        tmp = mass[:, None] @ mass[None]
        if adjust_sizes:
            distance += -size[:, None] - size[None]

        d2 = distance**2
        # remove self-interaction
        np.fill_diagonal(tmp, 0)
        np.fill_diagonal(d2, 1)
        factor = (tmp / d2) * scaling_ratio
        repulsion = np.einsum("ijk, ij -> ik", diff, factor)

        # gravity
        pos_centered = pos_arr - np.mean(pos_arr, axis=0)
        if strong_gravity:
            gravities = -gravity * mass[:, None] * pos_centered
        else:
            # hide warnings for divide by zero. Then change nan to 0
            with np.errstate(divide="ignore", invalid="ignore"):
                unit_vec = pos_centered / np.linalg.norm(pos_centered, axis=-1)[:, None]
            unit_vec = np.nan_to_num(unit_vec, nan=0)
            gravities = -gravity * mass[:, None] * unit_vec

        # total forces
        update = attraction + repulsion + gravities

        # compute total swing and traction
        swing += (mass * np.linalg.norm(pos_arr - update, axis=-1)).sum()
        traction += (0.5 * mass * np.linalg.norm(pos_arr + update, axis=-1)).sum()

        speed, speed_efficiency = estimate_factor(
            n,
            swing,
            traction,
            speed,
            speed_efficiency,
            jitter_tolerance,
        )

        # update pos
        if adjust_sizes:
            df = np.linalg.norm(update, axis=-1)
            swinging = mass * df
            factor = 0.1 * speed / (1 + np.sqrt(speed * swinging))
            factor = np.minimum(factor * df, 10.0 * np.ones(df.shape)) / df
        else:
            swinging = mass * np.linalg.norm(update, axis=-1)
            factor = speed / (1 + np.sqrt(speed * swinging))

        factored_update = update * factor[:, None]
        pos_arr += factored_update
        if abs(factored_update).sum() < 1e-10:
            break

    pos = dict(zip(G, pos_arr))
    if store_pos_as is not None:
        nx.set_node_attributes(G, pos, store_pos_as)

    return pos
