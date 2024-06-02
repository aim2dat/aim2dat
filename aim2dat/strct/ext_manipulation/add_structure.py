"""
Module that implements routines to add a functional group or adsorbed molecule to a structure.
"""

# Standard library imports
import os
from typing import Union, List
import itertools

# Third party library imports
import numpy as np
from scipy.spatial.transform import Rotation

# Internal library imports
from aim2dat.strct.ext_manipulation.decorator import (
    external_manipulation_method,
)
from aim2dat.strct.strct import Structure
from aim2dat.strct.strct_misc import _calc_atomic_distance
from aim2dat.utils.element_properties import get_element_symbol
from aim2dat.utils.maths import calc_angle
from aim2dat.io.yaml import load_yaml_file


cwd = os.path.dirname(__file__)


@external_manipulation_method
def add_structure(
    structure: Structure,
    wrap: bool = False,
    host_indices: Union[int, List[int]] = 0,
    guest_index: int = 0,
    guest_structure: Union[Structure, str] = "CH3",
    bond_length: float = 1.25,
    r_max: float = 15.0,
    cn_method: str = "minimum_distance",
    min_dist_delta: float = 0.1,
    n_nearest_neighbours: int = 5,
    econ_tolerance: float = 0.5,
    econ_conv_threshold: float = 0.001,
    voronoi_weight_type: float = "rel_solid_angle",
    voronoi_weight_threshold: float = 0.5,
    okeeffe_weight_threshold: float = 0.5,
    dist_constraints=[],
    dist_threshold: Union[float, None] = 0.8,
    change_label: bool = False,
) -> Structure:
    """
    Add a functional group or an atom to a host site.

    Parameters
    ----------
    structure : aim2dat.strct.Structure
        Structure to which the guest structure is added.
    wrap : bool (optional)
        Wrap atomic positions back into the unit cell.
    host_indices : int or list (optional)
        Index or indices of the host site(s). In case several indices are given the center of the
        sites is defined as host site as reference point for the bond length.
    guest_index : int (optional)
        Index of the guest site.
    guest_structure : str or aim2dat.strct.Structure (optional)
        A representation of the guest structure given as a string of a functional group or molecule
        (viable options are ``'CH3'``, ``'COOH'``, ``'H2O'``, ``'NH2'``, ``'NO2'`` or ``'OH'``), a
        ``Structure`` object (bond direction is assumed to be the ``[-1.0, 0.0, 0.0]`` direction)
        or the element symbol to add one single atom.
    bond_length : float
        Bond length between the host atom and the base atom of the functional group.
    r_max : float (optional)
        Cut-off value for the maximum distance between two atoms in angstrom.
    cn_method : str (optional)
        Method used to calculate the coordination environment.
    min_dist_delta : float (optional)
        Tolerance parameter that defines the relative distance from the nearest neighbour atom
        for the ``'minimum_distance'`` method.
    n_nearest_neighbours : int (optional)
        Number of neighbours that are considered coordinated for the ``'n_neighbours'``
        method.
    econ_tolerance : float (optional)
        Tolerance parameter for the econ method.
    econ_conv_threshold : float (optional)
        Convergence threshold for the econ method.
    voronoi_weight_type : str (optional)
        Weight type of the Voronoi facets. Supported options are ``'covalent_atomic_radius'``,
        ``'area'`` and ``'solid_angle'``. The prefix ``'rel_'`` specifies that the relative
        weights with respect to the maximum value of the polyhedron are calculated.
    voronoi_weight_threshold : float (optional)
        Weight threshold to consider a neighbouring atom coordinated.
    dist_constraints : list (optional)
        List of three-fold tuples containing the index of the site of the host structure, the index
        of the site of the guest structure and the target distance. The position of the guest
        structure is varied based on a grid search until the sum of the absolute errors in
        minimized.
    dist_threshold : float or None (optional)
        Check the distances between all site pairs of the host and guest structure to ensure that
        none of the added atoms collide.
    change_label : bool (optional)
        Add suffix to the label of the new structure highlighting the performed manipulation.

    Returns
    -------
    aim2dat.strct.Structure
        Structure with attached functional group.
    """
    guest_strct, guest_strct_label = _check_guest_structure(guest_structure)
    if isinstance(host_indices, int):
        host_indices = [host_indices]

    if all(idx < len(structure) for idx in host_indices) and guest_index < len(
        guest_strct["elements"]
    ):
        # Shift guest site to [0.0, 0.0, 0.0]
        guest_strct["positions"] = [
            np.array(pos0) - np.array(guest_strct["positions"][guest_index])
            for pos0 in guest_strct["positions"]
        ]

        # Calculate coordination:
        coord = structure.calculate_coordination(
            method=cn_method,
            min_dist_delta=min_dist_delta,
            n_nearest_neighbours=n_nearest_neighbours,
            econ_tolerance=econ_tolerance,
            econ_conv_threshold=econ_conv_threshold,
            okeeffe_weight_threshold=okeeffe_weight_threshold,
        )

        # Derive bond directions and rotation to align guest towards bond direction:
        bond_dir = _derive_bond_dir(host_indices[0], coord)
        host_positions = [structure.get_positions()[host_indices[0]]]
        for idx in host_indices[1:]:
            bond_dir += _derive_bond_dir(idx, coord)
            _, pos = _calc_atomic_distance(
                structure, host_indices[0], idx, backfold_positions=True
            )
            host_positions.append(pos)
        bond_dir /= np.linalg.norm(bond_dir)
        host_pos_np = np.mean(np.array(host_positions), axis=0)
        rot_dir = np.cross(bond_dir, np.array([1.0, 0.0, 0.0]))
        rot_dir /= np.linalg.norm(rot_dir)
        rot_angle = -calc_angle(np.array([1.0, 0.0, 0.0]), bond_dir)
        rotation = Rotation.from_rotvec(rot_angle * rot_dir)
        rot_matrix = rotation.as_matrix()
        for idx, pos in enumerate(guest_strct["positions"]):
            guest_strct["positions"][idx] = rot_matrix.dot(np.array(pos).T)

        # Define reference directions for rotations:
        ref_dir_alpha = bond_dir
        aux_dir = bond_dir.copy()
        aux_dir[0] += 1.0
        ref_dir_beta = np.cross(bond_dir, aux_dir)
        ref_dir_beta /= np.linalg.norm(ref_dir_beta)
        ref_dir_gamma = bond_dir
        ref_dirs = [ref_dir_alpha, ref_dir_beta, ref_dir_gamma]

        # Create new structure
        new_structure, score = _add_mol(
            structure,
            guest_strct,
            host_pos_np,
            bond_length,
            [0.0, 0.0, 0.0],
            ref_dirs,
            dist_constraints,
        )

        # Optimize positions to reduce score
        if len(dist_constraints) > 0:
            for alpha in np.linspace(-0.5 * np.pi, 0.5 * np.pi, num=5):
                for beta in np.linspace(-0.5 * np.pi, 0.5 * np.pi, num=5):
                    for gamma in np.linspace(0.0, 2.0 * np.pi, num=10):
                        new_strct0, score0 = _add_mol(
                            structure,
                            guest_strct,
                            host_pos_np,
                            bond_length,
                            [alpha, beta, gamma],
                            ref_dirs,
                            dist_constraints,
                        )
                        if score0 < score and _check_distances(
                            new_strct0, len(guest_strct["elements"]), dist_threshold, True
                        ):
                            score = score0
                            new_structure = new_strct0
        else:
            new_structure, _ = _add_mol(
                structure,
                guest_strct,
                host_pos_np,
                bond_length,
                [0.0, 0.0, 0.0],
                ref_dirs,
                dist_constraints,
            )
            _check_distances(new_structure, len(guest_strct["elements"]), dist_threshold, False)
        return new_structure, "_added-" + guest_strct_label


def _check_guest_structure(guest_strct: Union[Structure, str]) -> dict:
    if isinstance(guest_strct, Structure):
        label = "" if guest_strct.label is None else guest_strct.label
        return {"elements": guest_strct.elements, "positions": guest_strct.positions}, label
    elif isinstance(guest_strct, str):
        guest_strct_dict = {}
        try:
            guest_strct_dict["elements"] = [get_element_symbol(guest_strct)]
            guest_strct_dict["positions"] = [[0.0, 0.0, 0.0]]
        except ValueError:
            try:
                guest_strct_dict = load_yaml_file(
                    os.path.join(cwd, "pred_structures", guest_strct + ".yaml")
                )
            except FileNotFoundError:
                raise ValueError(f"`guest_structure` '{guest_strct}' is not supported.")
        return guest_strct_dict, guest_strct
    else:
        raise TypeError("`guest_structure` needs to be of type Structure or str.")


def _derive_bond_dir(host_index, coord):
    # Derive bond direction and rotation to align guest towards bond direction:
    cn_details = coord["sites"][host_index]
    host_pos_np = np.array(cn_details["position"])
    bond_dir = np.zeros(3)
    for neigh in cn_details["neighbours"]:
        dir_v = np.array(neigh["position"]) - host_pos_np
        bond_dir += dir_v / np.linalg.norm(dir_v)
    if np.linalg.norm(bond_dir) < 1e-5:
        bond_dir = np.cross(
            np.array(cn_details["neighbours"][0]["position"]),
            np.array(cn_details["neighbours"][1]["position"]),
        )
    bond_dir *= -1.0 / np.linalg.norm(bond_dir)
    return bond_dir


def _add_mol(
    structure, guest_strct, host_pos, bond_length, angle_pars, ref_dirs, dist_constraints
):
    # Reorient and shift guest structure:
    guest_pos = [pos.copy() for pos in guest_strct["positions"]]
    shifts = [bond_length * ref_dirs[2], np.zeros(3), np.zeros(3)]
    for p0, ref_dir, shift in zip(angle_pars, ref_dirs, shifts):
        rotation = Rotation.from_rotvec(p0 * ref_dir)
        for idx, pos in enumerate(guest_pos):
            guest_pos[idx] = rotation.as_matrix().dot(pos.T) + shift

    # Add guest structure to host:
    new_structure = structure.to_dict()
    new_structure["elements"] = list(new_structure["elements"])
    if new_structure["kinds"] is not None:
        new_structure["kinds"] = list(new_structure["kinds"])
    new_structure["positions"] = list(new_structure["positions"])
    for el, pos in zip(guest_strct["elements"], guest_pos):
        new_structure["elements"].append(el)
        new_structure["positions"].append(pos + host_pos)
        if new_structure["kinds"] is not None:
            new_structure["kinds"].append(None)
    new_structure = Structure(**new_structure)

    # Evaluate complex based on the distance constraints:
    score = 0.0
    if len(dist_constraints) > 0:
        host_indices = [idx for idx, _, _ in dist_constraints]
        guest_indices = [idx + len(structure) for _, idx, _ in dist_constraints]
        ref_dists = [dist for _, _, dist in dist_constraints]
        dists = new_structure.calculate_distance(
            list(host_indices), list(guest_indices), backfold_positions=True
        )
        if isinstance(dists, float):
            dists = [dists]
        score = sum(abs(dist - ref_dist) for dist, ref_dist in zip(dists, ref_dists))
    return new_structure, score


def _check_distances(
    new_structure: Structure, n_atoms: int, dist_threshold: Union[float, None], silent: bool
):
    if dist_threshold is None:
        return True

    indices_old = list(range(len(new_structure) - n_atoms))
    indices_new = list(range(len(new_structure) - n_atoms, len(new_structure)))
    indices1, indices2 = zip(*itertools.product(indices_old, indices_new))
    dists = new_structure.calculate_distance(
        list(indices1), list(indices2), backfold_positions=True
    )
    if any(d0 < dist_threshold for d0 in dists):
        if not silent:
            raise ValueError("Atoms are too close to each other.")
        return False
    return True
