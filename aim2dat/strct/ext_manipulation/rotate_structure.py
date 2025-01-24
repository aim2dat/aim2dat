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
    wrap: bool = False,
    change_label: bool = False,
):
    """
    Rotate sub structure around a point.

    Parameters
    ----------
    structure : aim2dat.strct.Structure
        Structure to which the guest structure is added.
    site_indices : list of int
        Indices of the atoms to rotate.
    angles : list of float
        Angles for the rotation in degree or around direction of `rotation_vector` if given.
    rotation_center : list of float (optional)
        Rotation center for the rotation in cartesian coordinates. If not given, the center of molecule is used.
    wrap : bool (optional)
        Wrap atomic positions back into the unit cell.
    change_label : bool (optional)
        Add suffix to the label of the new structure highlighting the performed manipulation.

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
    rotated_points = r.apply(positions)
    rotated_points += rotation_center

    new_structure = copy.deepcopy(structure)
    all_positions = list(new_structure["positions"])
    for idx, pos in zip(site_indices, rotated_points):
        all_positions[idx] = pos
    new_structure.set_positions(all_positions)
    if wrap:
        new_structure = Structure(**new_structure, wrap=wrap)

    return new_structure, "_rotated-" + f"{angles}"


@external_manipulation_method
def rotate_structure_around_vector(
    structure: Structure,
    site_indices: List[int],
    angle: float,
    rotation_vector: List[List[float]],
    origin: Union[None, List[float]] = None,
    wrap: bool = False,
    change_label: bool = False,
):
    """
    Rotate sub structure around a vector.

    Parameters
    ----------
    structure : aim2dat.strct.Structure
        Structure to which the guest structure is added.
    site_indices : list of int
        Indices of the atoms to rotate.
    angle : float or list of float
        Angles for the rotation in degree or around direction of `rotation_vector` if given.
    rotation_vector : list of float (optional)
        Rotation vector for the rotation in cartesian coordinates.
    origin : list of float (optional)
        Origin for the rotation in cartesian coordinates. If not given, the center of molecule is used.
    wrap : bool (optional)
        Wrap atomic positions back into the unit cell.
    change_label : bool (optional)
        Add suffix to the label of the new structure highlighting the performed manipulation.

    Returns
    -------
    aim2dat.strct.Structure
        Structure with rotated sub structure around vector.
    """

    positions = np.array([structure["positions"][idx] for idx in site_indices])
    if origin is None:
        origin = np.mean(positions, axis=0)
    positions -= origin
    rotation_vector /= np.linalg.norm(rotation_vector)
    rotation = Rotation.from_rotvec(angle * rotation_vector, degrees=True)
    rotated_points = rotation.apply(positions)
    rotated_points += origin

    new_structure = copy.deepcopy(structure)
    all_positions = list(new_structure["positions"])
    for idx, pos in zip(site_indices, rotated_points):
        all_positions[idx] = pos
    new_structure.set_positions(all_positions)
    if wrap:
        new_structure = Structure(**new_structure, wrap=wrap)

    return new_structure, "_rotated-" + f"{angle}"
