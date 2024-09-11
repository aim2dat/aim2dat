"""
Module that implements routines to add a functional group or adsorbed molecule to a structure.
"""

# Standard library imports
import os
from typing import Union, List
import itertools
import random
import copy

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
def add_structure_random(
    structure: Structure,
    wrap: bool = False,
    guest_structure: Union[Structure, str] = "CH3",
    dist_threshold: Union[float, None] = 0.8,
    random_state: Union[float, None] = None,
    change_label: bool = False,
):
    """
    Add structure at random position and orientation.

    Parameters
    ----------
    structure : aim2dat.strct.Structure
        Structure to which the guest structure is added.
    wrap : bool (optional)
        Wrap atomic positions back into the unit cell.
    guest_structure : str or aim2dat.strct.Structure (optional)
        A representation of the guest structure given as a string of a functional group or molecule
        (viable options are ``'CH3'``, ``'COOH'``, ``'H2O'``, ``'NH2'``, ``'NO2'`` or ``'OH'``), a
        ``Structure`` object (bond direction is assumed to be the ``[-1.0, 0.0, 0.0]`` direction)
        or the element symbol to add one single atom.
    dist_threshold : float or None (optional)
        Check the distances between all site pairs of the host and guest structure to ensure that
        none of the added atoms collide.
    random_state : float or None (optional)
        Specify the initial random state to ensure reproducible results.
    change_label : bool (optional)
        Add suffix to the label of the new structure highlighting the performed manipulation.

    Raises
    ------
    ValueError
        Could not add guest structure, host structure seems to be too aggregated.
    """
    guest_strct, guest_strct_label = _check_guest_structure(guest_structure)

    # In case no cell is given we would like to have the structure reasonably close:
    if structure.cell is None:
        threshold = 1.5 if dist_threshold is None else dist_threshold + 1.5
        positions = np.array(structure.positions)
        min_pos = np.amin(positions, axis=0) - threshold
        max_pos = np.amax(positions, axis=0)
        cell = np.zeros((3, 3))
        for d in range(3):
            cell[d][d] = max_pos[d] + threshold
    else:
        min_pos = np.zeros(3)
        cell = np.array(structure.cell)

    random.seed(a=random_state)
    max_tries = 1000
    for _ in range(max_tries):
        guest_positions = np.array(guest_strct["positions"])
        rot_v = np.array([random.random(), random.random(), random.random()])
        rotation = Rotation.from_rotvec(2.0 * random.random() * np.pi * rot_v)
        rot_matrix = rotation.as_matrix()
        guest_positions = (rot_matrix.dot(guest_positions.T)).T
        shift = np.array([random.random(), random.random(), random.random()])
        shift = (cell.T).dot(shift)
        guest_positions += shift - min_pos
        guest_strct0 = copy.deepcopy(guest_strct)
        guest_strct0.set_positions(guest_positions)

        new_structure = _merge_structures(structure, guest_strct0, wrap)
        is_added = _check_distances(
            new_structure, len(guest_strct["elements"]), dist_threshold, True
        )
        if is_added:
            return new_structure, "_added-" + guest_strct_label
    raise ValueError("Could not add guest structure, host structure seems to be too aggregated.")


@external_manipulation_method
def add_structure_coord(
    structure: Structure,
    wrap: bool = False,
    host_indices: Union[int, List[int]] = 0,
    guest_index: int = 0,
    guest_structure: Union[Structure, str] = "CH3",
    guest_dir: Union[None, List[float]] = None,
    bond_length: float = 1.25,
    r_max: float = 15.0,
    cn_method: str = "minimum_distance",
    min_dist_delta: float = 0.1,
    n_nearest_neighbours: int = 5,
    radius_type: str = "chen_manz",
    atomic_radius_delta: float = 0.0,
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
    guest_dir: list of floats (optional)
        Defines the orientation of the guest molecule. If not defined, a vector of nearest
        neighbors is constructed based on the guest index.
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

    if max(host_indices) >= len(structure) or guest_index >= len(guest_strct):
        idx = max(host_indices)
        return structure

    # Shift guest site to [0.0, 0.0, 0.0]
    guest_strct.set_positions(
        [
            np.array(pos0) - np.array(guest_strct["positions"][guest_index])
            for pos0 in guest_strct["positions"]
        ]
    )

    if guest_dir is None:
        guest_dir = np.array([1.0, 0.0, 0.0])
        if len(guest_strct) > 1:
            # Get vector of guest atoms for rotation
            guest_strct_coord = guest_strct.calculate_coordination(
                r_max=r_max,
                method=cn_method,
                min_dist_delta=min_dist_delta,
                n_nearest_neighbours=n_nearest_neighbours,
                radius_type=radius_type,
                atomic_radius_delta=atomic_radius_delta,
                econ_tolerance=econ_tolerance,
                econ_conv_threshold=econ_conv_threshold,
                voronoi_weight_type=voronoi_weight_type,
                voronoi_weight_threshold=voronoi_weight_threshold,
                okeeffe_weight_threshold=okeeffe_weight_threshold,
            )
            guest_dir = -1.0 * _derive_bond_dir(guest_index, guest_strct_coord)

    # Calculate coordination:
    coord = structure.calculate_coordination(
        r_max=r_max,
        method=cn_method,
        min_dist_delta=min_dist_delta,
        n_nearest_neighbours=n_nearest_neighbours,
        radius_type=radius_type,
        atomic_radius_delta=atomic_radius_delta,
        econ_tolerance=econ_tolerance,
        econ_conv_threshold=econ_conv_threshold,
        voronoi_weight_type=voronoi_weight_type,
        voronoi_weight_threshold=voronoi_weight_threshold,
        okeeffe_weight_threshold=okeeffe_weight_threshold,
    )

    # Derive bond directions and rotation to align guest towards bond direction:
    bond_dir = _derive_bond_dir(host_indices[0], coord)
    host_positions = [structure.get_positions()[host_indices[0]]]
    for idx in host_indices[1:]:
        bond_dir += _derive_bond_dir(idx, coord)
        _, pos = _calc_atomic_distance(structure, host_indices[0], idx, backfold_positions=True)
        host_positions.append(pos)
    bond_dir /= np.linalg.norm(bond_dir)
    host_pos_np = np.mean(np.array(host_positions), axis=0)
    if len(guest_strct) > 1:
        rot_dir = np.cross(bond_dir, guest_dir)
        rot_dir /= np.linalg.norm(rot_dir)
        rot_angle = -calc_angle(guest_dir, bond_dir)
        rotation = Rotation.from_rotvec(rot_angle * rot_dir)
        rot_matrix = rotation.as_matrix()
        guest_strct.set_positions(
            [rot_matrix.dot(np.array(pos).T) for pos in guest_strct["positions"]]
        )

    # Check bond length and adjusts if necessary
    if all(host_pos_np == host_positions[0]):
        pass
    elif all(
        [np.linalg.norm(host_pos_np - host_pos) >= bond_length for host_pos in host_positions]
    ):
        bond_length = 0.0
    else:
        bond_length = _rescale_bond_length(host_pos_np, host_positions, bond_dir, bond_length)

    # # Define reference directions for rotations:
    ref_dir_alpha = bond_dir
    aux_dir = bond_dir.copy()
    aux_dir[0] += 1.0
    ref_dir_beta = np.cross(bond_dir, aux_dir)
    ref_dir_beta /= np.linalg.norm(ref_dir_beta)
    ref_dir_gamma = bond_dir
    ref_dirs = [ref_dir_alpha, ref_dir_beta, ref_dir_gamma]

    # # Create new structure
    new_structure, score = _add_mol(
        structure,
        guest_strct,
        wrap,
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
                        wrap,
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
            wrap,
            host_pos_np,
            bond_length,
            [0.0, 0.0, 0.0],
            ref_dirs,
            dist_constraints,
        )
        _check_distances(new_structure, len(guest_strct["elements"]), dist_threshold, False)
    return new_structure, "_added-" + guest_strct_label


def _check_guest_structure(guest_strct: Union[Structure, str]) -> Structure:
    if isinstance(guest_strct, Structure):
        label = "" if guest_strct.label is None else guest_strct.label
        return guest_strct, label
    elif isinstance(guest_strct, str):
        guest_strct_dict = {}
        try:
            strct = Structure(
                label=guest_strct,
                elements=[get_element_symbol(guest_strct)],
                positions=[[0.0, 0.0, 0.0]],
                pbc=False,
            )
        except ValueError:
            try:
                guest_strct_dict = load_yaml_file(
                    os.path.join(cwd, "pred_structures", guest_strct + ".yaml")
                )
                strct = Structure(
                    label=guest_strct,
                    elements=guest_strct_dict["elements"],
                    positions=guest_strct_dict["positions"],
                    pbc=False,
                )
            except FileNotFoundError:
                raise ValueError(f"`guest_structure` '{guest_strct}' is not supported.")
        return strct, guest_strct
    else:
        raise TypeError("`guest_structure` needs to be of type Structure or str.")


def _derive_bond_dir(index, coord):
    # Derive bond direction and rotation to align guest towards bond direction:
    cn_details = coord["sites"][index]
    pos = np.array(cn_details["position"])
    bond_dir = np.zeros(3)
    for neigh in cn_details["neighbours"]:
        dir_v = np.array(neigh["position"]) - pos
        bond_dir += dir_v / np.linalg.norm(dir_v)
    if np.linalg.norm(bond_dir) < 1e-1:
        bond_dir = np.cross(
            np.array(cn_details["neighbours"][0]["position"]) - pos,
            np.array(cn_details["neighbours"][1]["position"]) - pos,
        )
    bond_dir *= -1.0 / np.linalg.norm(bond_dir)
    return bond_dir


def _rescale_bond_length(host_pos_np, host_positions, bond_dir, bond_length):
    # Rescale bond length to match the distance between guest molecule and host atoms
    scaled_bond_lengths = []
    for host_pos in host_positions:
        # Calculate the coefficients for the quadratic equation
        dis_host_center = host_pos_np - host_pos
        bond_dir_square = np.dot(bond_dir, bond_dir)
        bond_dir_dot_dis_host_center = np.dot(bond_dir, dis_host_center)
        dis_host_center_square = np.dot(dis_host_center, dis_host_center)
        # Coefficients of the quadratic equation at^2 + bt + c = 0
        a = bond_dir_square
        b = 2 * bond_dir_dot_dis_host_center
        c = dis_host_center_square - bond_length**2
        # Solve the quadratic equation
        t1, t2 = np.roots([a, b, c])
        scaled_bond_lengths.append(max([t1, t2]))

    for sbl in scaled_bond_lengths:
        guest = host_pos_np + sbl * bond_dir
        if all(
            [np.linalg.norm(guest - host_pos) >= bond_length * 0.95 for host_pos in host_positions]
        ):
            bond_length = sbl
            break
    return bond_length


def _add_mol(
    structure, guest_strct, wrap, host_pos, bond_length, angle_pars, ref_dirs, dist_constraints
):
    # Reorient and shift guest structure:
    guest_strct = copy.deepcopy(guest_strct)
    guest_pos = [list(pos) for pos in guest_strct["positions"]]
    shifts = [bond_length * ref_dirs[2], np.zeros(3), np.zeros(3)]
    for p0, ref_dir, shift in zip(angle_pars, ref_dirs, shifts):
        rotation = Rotation.from_rotvec(p0 * ref_dir)
        for idx, pos in enumerate(guest_pos):
            guest_pos[idx] = rotation.as_matrix().dot(np.array(pos).T) + shift
    for idx in range(len(guest_pos)):
        guest_pos[idx] += host_pos
    guest_strct.set_positions(guest_pos)

    # Add guest structure to host:
    new_structure = _merge_structures(structure, guest_strct, wrap)

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


def _merge_structures(host_strct, guest_strct, wrap):
    new_structure = host_strct.to_dict()
    new_structure["elements"] = list(new_structure["elements"])
    new_structure["kinds"] = list(new_structure["kinds"])
    new_structure["positions"] = list(new_structure["positions"])
    new_structure["site_attributes"] = {
        key: list(val) for key, val in new_structure["site_attributes"].items()
    }
    if len(guest_strct["site_attributes"]) > 0:
        for site_attr in guest_strct["site_attributes"].keys():
            if site_attr not in new_structure["site_attributes"]:
                new_structure["site_attributes"][site_attr] = [None] * len(
                    new_structure["elements"]
                )
    for el, kind, pos in zip(
        guest_strct["elements"], guest_strct["kinds"], guest_strct["positions"]
    ):
        new_structure["elements"].append(el)
        new_structure["kinds"].append(kind)
        new_structure["positions"].append(pos)
        for site_attr, val in new_structure["site_attributes"].items():
            val.append(guest_strct["site_attributes"].get(site_attr, None))
    return Structure(**new_structure, wrap=wrap)


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
