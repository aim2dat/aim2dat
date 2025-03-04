"""Module to check interatomic distances."""

# Standard library imports
import itertools
from typing import Union, Dict, List

# Internal library imports
from aim2dat.strct import Structure
from aim2dat.utils.element_properties import get_atomic_radius


class DistanceThresholdError(ValueError):
    """Error in case distances between atom sites are too short or too long."""

    pass


def _check_distances(
    structure: Structure,
    indices: List[int],
    dist_threshold: Union[Dict, List, float, None],
    silent: bool,
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
    dist_threshold : dict, list of float, or float (optional)
        Check the distances between all site pairs of the host and guest structure to ensure that
        none of the added atoms collide or are too far apart.
        Example: dist_threshold = 0.8
        Example: dist_threshold = [0.8, 1.5]
        Example: dist_threshold = {"C": {"H": 0.8, "C": [0.8, 1.5]}}
    silent : bool (optional)
        If True, no error is raised when atoms are too close. If False, a ValueError is raised
        when atoms are too close to each other.

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
    if dist_threshold is None:
        return True

    # Calculate pair-wise distances:
    other_indices = [i for i in range(len(structure)) if i not in indices]
    if len(other_indices) == 0:
        indices1, indices2 = zip(*itertools.combinations(indices, 2))
    else:
        indices1, indices2 = zip(*itertools.product(other_indices, indices))
    dists = structure.calculate_distance(list(indices1), list(indices2), backfold_positions=True)

    el_pairs = itertools.combinations_with_replacement(structure._element_dict.keys(), 2)
    # If dist_threshold is dict, make sure that it contains index or element pairs and check that
    # value is list:
    if isinstance(dist_threshold, dict):
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
        print(tol)

        atomic_radii = {
            el: get_atomic_radius(el, radius_type=dist_threshold)
            for el in structure._element_dict.keys()
        }
        dist_threshold = {
            tuple(sorted(pair)): [(atomic_radii[pair[0]] + atomic_radii[pair[1]]) * tol, None]
            for pair in el_pairs
        }
        print(dist_threshold, tol)
    # Transfer list of min/max values to element-pair dict:
    elif isinstance(dist_threshold, (list, tuple)):
        dist_threshold = {tuple(sorted(pair)): dist_threshold for pair in el_pairs}
    # Transfer float/int to element-pair dict:
    elif isinstance(dist_threshold, (list, tuple)):
        dist_threshold = {tuple(sorted(pair)): [dist_threshold, None] for pair in el_pairs}
    else:
        raise TypeError("`dist_threshold` needs to be of type int/float/list/tuple/dict.")

    for idx_pair, dist in dists.items():
        threshold = dist_threshold.get(tuple(sorted(idx_pair)), None)
        if threshold is None:
            el_pair = tuple(
                sorted([structure.elements[idx_pair[0]], structure.elements[idx_pair[1]]])
            )
            threshold = dist_threshold.get(el_pair, None)
        if threshold is None:
            continue

        if dist < dist_threshold[el_pair][0]:
            if silent:
                return False
            raise DistanceThresholdError(
                f"Atoms {idx_pair[0]} and {idx_pair[1]} are too close to each other."
            )
        if dist_threshold[el_pair][1] is not None and dist > dist_threshold[el_pair][1]:
            if silent:
                return False
            raise DistanceThresholdError(
                f"Atoms {idx_pair[0]} and {idx_pair[1]} are too far from each other."
            )
    return True
