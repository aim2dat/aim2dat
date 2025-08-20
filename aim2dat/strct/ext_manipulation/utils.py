"""Module to check interatomic distances."""

# Standard library imports
import itertools
from typing import Union, List, Tuple

# Internal library imports
from aim2dat.strct import Structure
from aim2dat.elements import get_atomic_radius


class DistanceThresholdError(ValueError):
    """Error in case distances between atom sites are too short or too long."""

    pass


def _build_distance_dict(
    dist_threshold: Union[list, dict, tuple, str, float, int], structure, guest_structure=None
) -> Tuple[Union[dict, None], float]:
    """Construct dictionary from dist_threshold parameter."""
    elements = set(structure._element_dict.keys())

    if guest_structure is not None:
        elements = elements.union(guest_structure._element_dict.keys())
    el_pairs = itertools.combinations_with_replacement(elements, 2)
    if dist_threshold is None:
        return None, 0.0
    # If dist_threshold is dict, make sure that it contains index or element pairs and check that
    # value is list:
    elif isinstance(dist_threshold, dict):
        new_dict = {}
        for key, val in dist_threshold.items():
            if len(key) != 2:
                raise ValueError(
                    "`dist_threshold` needs to have keys with length 2 containing site "
                    + "indices or element symbols."
                )
            if not (all(isinstance(k, int) for k in key) or all(isinstance(k, str) for k in key)):
                raise ValueError(
                    "`dist_threshold` needs to have keys of type List[str/int] containing "
                    + "site indices or element symbols."
                )
            if isinstance(val, (float, int)):
                val = [val, None]
            new_dict[tuple(sorted(key))] = val
        dist_threshold = new_dict
    # Transfer string to element-pair dict:
    elif isinstance(dist_threshold, str):
        tol = 1.0
        if "+" in dist_threshold:
            dist_threshold, tol = dist_threshold.split("+")
            tol = 1.0 + float(tol) / 100.0
        elif "-" in dist_threshold:
            dist_threshold, tol = dist_threshold.split("-")
            tol = 1.0 - float(tol) / 100.0
        atomic_radii = {el: get_atomic_radius(el, radius_type=dist_threshold) for el in elements}
        dist_threshold = {
            tuple(sorted(pair)): [(atomic_radii[pair[0]] + atomic_radii[pair[1]]) * tol, None]
            for pair in el_pairs
        }
    # Transfer list of min/max values to element-pair dict:
    elif isinstance(dist_threshold, (list, tuple)):
        dist_threshold = {tuple(sorted(pair)): dist_threshold for pair in el_pairs}
    # Transfer float/int to element-pair dict:
    elif isinstance(dist_threshold, (float, int)):
        dist_threshold = {tuple(sorted(pair)): [dist_threshold, None] for pair in el_pairs}
    else:
        raise TypeError("`dist_threshold` needs to be of type int/float/list/tuple/dict or None.")
    return dist_threshold, min(val[0] for val in dist_threshold.values())


def _check_distances(
    structure: Structure,
    indices: Union[List[int], slice],
    dist_threshold: Union[dict, list, float, int, None],
    distance_dict: Union[dict, None],
    silent: bool,
    return_score: bool = False,
) -> bool:
    """
    Check if the distance between the specified atoms and all other atoms in the structure
    is in the given threshold.

    Parameters
    ----------
    structure : aim2dat.strct.Structure
        The structure containing the atom positions.
    indices : list of int
        A list of indices identifying the atoms in the structure whose distances are to be
        checked.
    dist_threshold : dict, list, float, int, str or None (optional)
        Check the distances between all site pairs to ensure that none of the changed atoms
        collide or are too far apart. For example, ``0.8`` to ensure a minimum distance of
        ``0.8`` for all site pairs. A list ``[0.8, 1.5]`` adds a check for the maximum distance
        as well. Giving a dictionary ``{("C", "H"): 0.8, (0, 4): 0.8}`` allows distance checks
        for individual pairs of elements or site indices. Specifying an atomic radius type as
        str, e.g. ``covalent+10`` sets the minimum threshold to the sum of covalent radii plus
        10%.
    distance_dict : dict
        Dictionary containing element or index tuples as keys and distance thresholds as values.
    silent : bool (optional)
        If True, no error is raised when atoms are too close. If False, a ValueError is raised
        when atoms are too close to each other.
    return_score : bool (optional)
        If true a score will be returned instead of a bool.

    Returns
    ----------
    bool
        True if all distances are in between the threshold, or no check was performed.
        False if any distance outside the threshold and `silent` is False.

    Raises
    ----------
    ValueError
        If any distance between atoms is outside the threshold and `silent` is False.
    """
    score = 0.0
    max_dist = {}

    if dist_threshold is not None:
        distance_dict, _ = _build_distance_dict(dist_threshold, structure)

    if distance_dict is None:
        return True

    # Calculate pair-wise distances:
    if isinstance(indices, slice):
        indices = list(range(len(structure)))[indices]
    other_indices = [i for i in range(len(structure)) if i not in indices]
    dists = structure.calc_distance(other_indices, indices, backfold_positions=True)

    if isinstance(dist_threshold, (int, float)):
        min_key = min(dists, key=dists.get)
        min_value = dists[min_key]
        if min_value < dist_threshold:
            if silent:
                return False
            raise DistanceThresholdError(
                f"Atoms {min_key[0]} and {min_key[1]} are too close to each other."
            )
        if return_score:
            return abs(min(dists.values()) - dist_threshold)
        return True

    for idx_pair, dist in dists.items():
        key = tuple(sorted(idx_pair))
        threshold = distance_dict.get(key, None)

        if threshold is None:
            key = tuple(sorted([structure.elements[idx_pair[0]], structure.elements[idx_pair[1]]]))
            threshold = distance_dict.get(key, None)
        if threshold is None:
            continue

        if dist < threshold[0]:
            if silent:
                return False
            raise DistanceThresholdError(
                f"Atoms {idx_pair[0]} and {idx_pair[1]} are too close to each other."
            )

        if threshold[1] is None:
            score += (
                abs(dist - threshold[0]) if abs(dist - threshold[0]) < 1.5 * threshold[0] else 0
            )
            continue

        if dist <= threshold[1]:
            score += abs(dist - threshold[0])
            max_dist[key] = True
            min_max_dist = 0
        elif key not in max_dist:
            max_dist[key] = False
            max_pair = idx_pair
            min_max_dist = dist
        elif dist < min_max_dist:
            max_pair = idx_pair
    if max_dist and not all(max_dist.values()):
        if silent:
            return False
        raise DistanceThresholdError(
            f"Atoms {max_pair[0]} and {max_pair[1]} are too far from each other."
        )
    if return_score:
        return score
    return True
