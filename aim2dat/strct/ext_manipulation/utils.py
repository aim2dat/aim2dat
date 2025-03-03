"""Module to check interatomic distances."""

import itertools
from typing import Union, Dict, List

from aim2dat.strct import Structure


# TODO Add doc strings.
def _check_distances(
    structure: Structure,
    indices: List[int],
    dist_threshold: Union[
        Dict[str, Dict[str, List[float]]], Dict[str, Dict[str, float]], List[float], float, None
    ],
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

    other_indices = [i for i in range(len(structure)) if i not in indices]
    if len(other_indices) == 0:
        indices1, indices2 = zip(*itertools.combinations(indices, 2))
    else:
        indices1, indices2 = zip(*itertools.product(other_indices, indices))
    dists = structure.calculate_distance(list(indices1), list(indices2), backfold_positions=True)
    if isinstance(dist_threshold, (int, float)):
        if any(d0 < dist_threshold for d0 in dists.values()):
            if not silent:
                raise ValueError("Atoms are too close to each other.")
            return False
    elif isinstance(dist_threshold, list):
        if any(d0 < dist_threshold[0] for d0 in dists.values()):
            if not silent:
                raise ValueError("Atoms are too close to each other.")
            return False
        if all(d0 > dist_threshold[1] for d0 in dists.values()):
            if not silent:
                raise ValueError("Atoms are too far from each other.")
            return False
    elif isinstance(dist_threshold, dict):
        for el1, el2_dist_thresh in dist_threshold.items():
            # Find indicies for first element type in first index list
            el1_idx1 = [idx for idx in other_indices if structure.elements[idx] == el1]
            # Find indicies for first element type in second index list
            el1_idx2 = [idx for idx in indices if structure.elements[idx] == el1]
            for el2, dist_thresh in el2_dist_thresh.items():
                # Find indicies for second element in first index lists
                el2_idx1 = [idx for idx in other_indices if structure.elements[idx] == el2]
                # Find indicies for second element in second index lists
                el2_idx2 = [idx for idx in indices if structure.elements[idx] == el2]

                # Get element pairs
                el1_el2_pair = []
                if el1_idx1 and el2_idx2:
                    el1_el2_pair += [(idx1, idx2) for idx1 in el1_idx1 for idx2 in el2_idx2]
                elif el1_idx2 and el2_idx1:
                    el1_el2_pair += [(idx1, idx2) for idx1 in el1_idx2 for idx2 in el2_idx1]

                if isinstance(dist_thresh, (int, float)):
                    if any(dists.get(d0) < dist_thresh for d0 in el1_el2_pair):
                        if not silent:
                            raise ValueError("Atoms are too close to each other.")
                        return False
                elif isinstance(dist_thresh, list):
                    if any(dists.get(d0) < dist_thresh[0] for d0 in el1_el2_pair):
                        if not silent:
                            raise ValueError("Atoms are too close to each other.")
                        return False
                    if all(dists.get(d0) > dist_thresh[1] for d0 in el1_el2_pair):
                        if not silent:
                            raise ValueError("Atoms are too far from each other.")
                        return False
    return True
