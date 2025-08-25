"""
Module that implements routines to add a functional group or adsorbed molecule to a structure.
"""

# Standard library imports
from typing import Union, List

# Third party library imports
import numpy as np
from scipy.spatial.transform import Rotation

# Internal library imports
from aim2dat.strct.structure import Structure
from aim2dat.strct.ext_manipulation.decorator import (
    external_manipulation_method,
)
from aim2dat.strct.ext_manipulation.utils import _check_distances


@external_manipulation_method
def rotate_structure(
    structure: Structure,
    angles: Union[float, List[float]],
    vector: Union[None, List[float]] = None,
    origin: Union[None, List[float]] = None,
    site_indices: Union[slice, List[int]] = slice(None),
    wrap: bool = False,
    dist_threshold: Union[dict, list, float, int, str, None] = None,
    change_label: bool = False,
):
    """
    Rotate structure. The rotation is either defined by a list of 3 angles or a rotation
    vector and one angle.

    Parameters
    ----------
    structure : aim2dat.strct.Structure
        Structure to rotate.
    angles : float or list of float
        Angles for the rotation in degree. Type ``list`` for 3 individual rotations around
        the x, y, and z directions, respectively. Type ``float`` for a rotation around a
        rotation vector given by ``vector``..
    vector : list of float (optional)
        Rotation vector in cartesian coordinates, needs to be given if ``angles`` is single
        number.
    origin : list of float (optional)
        Rotation center for the rotation in cartesian coordinates. If not given, the mean position
        of all sites that are rotated is used.
    site_indices : list of int (optional)
        Indices of the sites to rotate. If not given, all sites of the structure are rotated.
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
        Rotated structure.

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
    if isinstance(angles, (list, tuple, np.ndarray)):
        rotation = Rotation.from_euler("xyz", angles, degrees=True)
    elif isinstance(angles, (int, float)):
        vector /= np.linalg.norm(vector)
        rotation = Rotation.from_rotvec(angles * vector, degrees=True)
    else:
        raise TypeError("angles must be type list or type float.")

    site_indices = list(site_indices) if isinstance(site_indices, tuple) else site_indices
    positions = np.array(structure.positions)
    if origin is None:
        origin = np.mean(positions[site_indices], axis=0)
    origin = np.array(origin)
    positions[site_indices] -= origin
    positions[site_indices] = rotation.apply(positions[site_indices])
    positions[site_indices] += origin
    new_structure = structure.to_dict()
    new_structure["positions"] = positions
    new_structure = Structure(**new_structure, wrap=wrap)
    _check_distances(new_structure, site_indices, dist_threshold, None, False)
    return new_structure, "_rotated-" + f"{angles}"
