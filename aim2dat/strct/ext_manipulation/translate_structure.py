"""Module that implements routines to translate a structure."""

# Standard library imports
from typing import List, Union

# Third party library imports
import numpy as np

# Internal library imports
from aim2dat.strct.structure import Structure
from aim2dat.strct.ext_manipulation.decorator import (
    external_manipulation_method,
)
from aim2dat.strct.ext_manipulation.utils import _check_distances


@external_manipulation_method
def translate_structure(
    structure: Structure,
    vector: List[float],
    site_indices: Union[slice, List[int]] = slice(None),
    wrap: bool = False,
    dist_threshold: Union[dict, list, float, int, str, None] = None,
    change_label: bool = False,
) -> Structure:
    """
    Translate structure.

    Parameters
    ----------
    structure : aim2dat.strct.Structure
        Structure to translate.
    vector : list of float (optional)
        Translation vector.
    site_indices : list of int (optional)
        Indices of the sites to translate. If not given, all sites of the structure are translated.
    wrap : bool (optional)
        Wrap atomic positions back into the unit cell.
    dist_threshold : dict, list, float, int, str or None (optional)
        Check the distances between all site pairs to ensure that none of the changed atoms
        collide or are too far apart. For example, ``0.8`` to ensure a minimum distance of
        ``0.8`` for all site pairs. A list ``[0.8, 1.5]`` adds a check for the maximum distance
        as well. Giving a dictionary ``{("C", "H"): 0.8, (0, 4): 0.8}`` allows distance checks
        for individual pairs of elements or site indices. Specifying an atomic radius type as
        str, e.g. ``covalent+10`` sets the minimum threshold to the sum of covalent radii plus
        10%.
    change_label : bool (optional)
        Add suffix to the label of the new structure highlighting the performed manipulation.

    Returns
    -------
    aim2dat.strct.Structure
        Translated structure.

    Raises
    ------
    ValueError
        `dist_threshold` needs to have keys with length 2 containing site indices or element
        symbols.
    ValueError
        `dist_threshold` needs to have keys of type List[str/int] containing site indices or
        element symbols.
    TypeError
        `dist_threshold` needs to be of type int/float/list/tuple/dict or None.
    ValueError
        If any distance between atoms is outside the threshold.
    """
    site_indices = list(site_indices) if isinstance(site_indices, tuple) else site_indices
    positions = np.array(structure.positions)
    positions[site_indices] += np.array(vector)

    new_structure = structure.to_dict()
    new_structure["positions"] = positions
    new_structure = Structure(**new_structure, wrap=wrap)
    _check_distances(new_structure, site_indices, dist_threshold, None, False)
    return new_structure, f"_translated-{[round(v, 2) for v in vector]}"
