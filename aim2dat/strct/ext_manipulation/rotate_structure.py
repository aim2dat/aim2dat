"""
Module that implements routines to add a functional group or adsorbed molecule to a structure.
"""

# Standard library imports
from typing import Union, List
import copy

# Third party library imports
import numpy as np
from scipy.spatial.transform import Rotation

# Internal library imports
from aim2dat.strct.ext_manipulation.decorator import (
    external_manipulation_method,
)
from aim2dat.strct.strct import Structure


@external_manipulation_method
def rotate_structure(
    structure: Structure,
    angles: Union[float, List[float]],
    vector: Union[None, List[float]] = None,
    origin: Union[None, List[float]] = None,
    site_indices: Union[None, List[int]] = None,
    wrap: bool = False,
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
        roation vector given by ``vector``..
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
    change_label : bool (optional)
        Add suffix to the label of the new structure highlighting the performed manipulation.

    Returns
    -------
    aim2dat.strct.Structure
        Rotated structure.
    """
    if isinstance(angles, (list, tuple, np.ndarray)):
        rotation = Rotation.from_euler("xyz", angles, degrees=True)
    elif isinstance(angles, (int, float)):
        vector /= np.linalg.norm(vector)
        rotation = Rotation.from_rotvec(angles * vector, degrees=True)
    else:
        raise TypeError("angles must be type list or type float.")

    if site_indices is None:
        site_indices = list(range(len(structure)))

    positions = np.array([structure["positions"][idx] for idx in site_indices])
    if origin is None:
        origin = np.mean(positions, axis=0)
    origin = np.array(origin)
    positions -= origin
    rotated_points = rotation.apply(positions)
    rotated_points += origin

    new_structure = copy.deepcopy(structure)
    all_positions = list(new_structure["positions"])
    for idx, pos in zip(site_indices, rotated_points):
        all_positions[idx] = pos
    new_structure.set_positions(all_positions)

    return Structure(**new_structure, wrap=wrap), "_rotated-" + f"{angles}"
