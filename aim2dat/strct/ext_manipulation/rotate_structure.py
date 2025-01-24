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
def rotate_structure_around_point(
    structure: Structure,
    site_indices: List[int],
    angles: List[float],
    rotation_center: Union[None, List[float]] = None,
    change_label: bool = False,
):
    """
    Add structure at random position and orientation.

    Parameters
    ----------
    structure : aim2dat.strct.Structure
        Structure to which the guest structure is added.
    angles : list of float
        Angles for the rotation in degree or around direction of `rotation_vector` if given.
    rotation_center : list of float (optional)
        Rotation center for the rotation in cartesian coordinates. If not given, the center of molecule is used.
    dist_threshold : float or None (optional)
        Check the distances between all site pairs of the host and guest structure to ensure that
        none of the added atoms collide.

    Returns
    -------
    aim2dat.strct.Structure
        Structure with rotated sub structure.
    """

    r = Rotation.from_euler('xyz', angles, degrees=True)

    positions = np.array([structure["positions"][idx] for idx in site_indices])
    if rotation_center is None:
        rotation_center = np.mean(positions, axis=0)
    rotation_center = np.array(rotation_center)
    positions -= rotation_center
    rotated_point = r.apply(positions)
    rotated_point += rotation_center

    new_structure = copy.deepcopy(structure)
    all_positions = list(new_structure["positions"])
    for idx, pos in zip(site_indices, rotated_point):
        all_positions[idx] = pos
    new_structure.set_positions(all_positions)

    return new_structure, "_added-" + f"{angles}"
