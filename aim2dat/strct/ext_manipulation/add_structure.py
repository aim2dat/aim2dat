"""
Module that implements routines to add a functional group or adsorbed molecule to a structure.
"""

# Standard library imports
import os
from typing import Union, List
import copy

# Third party library imports
import numpy as np

# Internal library imports
from aim2dat.strct.strct import Structure
from aim2dat.strct.strct_validation import SamePositionsError
from aim2dat.strct.ext_manipulation.decorator import external_manipulation_method
from aim2dat.strct.ext_manipulation.utils import (
    _build_distance_dict,
    _check_distances,
    DistanceThresholdError,
)
from aim2dat.strct.ext_manipulation.rotate_structure import rotate_structure
from aim2dat.strct.strct_misc import _calc_atomic_distance
from aim2dat.utils.element_properties import get_element_symbol
from aim2dat.utils.maths import calc_angle, create_lin_ind_vector
from aim2dat.io import read_yaml_file


cwd = os.path.dirname(__file__)


@external_manipulation_method
def add_structure_random(
    structure: Structure,
    guest_structure: Union[Structure, str] = "CH3",
    max_tries: int = 1000,
    random_seed: Union[float, None] = None,
    random_nrs: Union[list, None] = None,
    dist_threshold: Union[dict, list, float, int, str, None] = 0.8,
    wrap: bool = False,
    change_label: bool = False,
) -> Structure:
    """
    Add structure at random position and orientation.

    Parameters
    ----------
    structure : aim2dat.strct.Structure
        Structure to which the guest structure is added.
    guest_structure : str or aim2dat.strct.Structure (optional)
        A representation of the guest structure given as a string of a functional group or molecule
        (viable options are ``'CH3'``, ``'COOH'``, ``'H2O'``, ``'NH2'``, ``'NO2'`` or ``'OH'``), a
        ``Structure`` object or the element symbol to add one single atom.
    max_tries : int
        Number of tries to add the guest structure. A try is rejected via the criteria given by
        the ``dist_treshold`` parameter.
    random_seed : int or None (optional)
        Specify the random seed to ensure reproducible results.
    random_nrs : list or None (optional)
        List of random numbers used to derive the position and rotation of the guest molecule. It
        should contain ``max_tries * 7`` entries to cover the maximum amout of tries.
    dist_threshold : dict, list, float, int, str or None (optional)
        Check the distances between all site pairs of the host and guest structure to ensure that
        none of the added atoms collide or are too far apart. For example, ``0.8`` to ensure a
        minimum distance of ``0.8`` for all site pairs. A list ``[0.8, 1.5]`` adds a check for
        the maximum distance as well. Giving a dictionary ``{("C", "H"): 0.8, (0, 4): 0.8}``
        allows distance checks for individual pairs of elements or site indices. Specifying an
        atomic radius type as str, e.g. ``covalent+10`` sets the minimum threshold to the sum
        of covalent radii plus 10%.
    wrap : bool (optional)
        Wrap atomic positions back into the unit cell.
    change_label : bool (optional)
        Add suffix to the label of the new structure highlighting the performed manipulation.

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
        Could not add guest structure, host structure seems to be too aggregated.
    """
    guest_strct, guest_strct_label = _check_guest_structure(guest_structure)
    distance_dict, min_dist = _build_distance_dict(dist_threshold, structure, guest_strct)

    # In case no cell is given we would like to have the structure reasonably close:
    if structure.cell is None:
        positions = np.array(structure.positions)
        min_pos = np.amin(positions, axis=0) - min_dist - 1.5
        max_pos = np.amax(positions, axis=0) + min_dist + 1.5
        cell = np.zeros((3, 3))
        for d in range(3):
            cell[d][d] = max_pos[d]
    else:
        min_pos = np.zeros(3)
        cell = np.array(structure.cell)

    if random_nrs is None:
        rng = np.random.default_rng(seed=random_seed)
    for _ in range(max_tries):
        r_nrs = (
            [rng.random() for _ in range(7)]
            if random_nrs is None
            else [random_nrs.pop(0) for _ in range(7)]
        )
        rot_v = np.array([v - 0.5 for v in r_nrs[:3]])
        guest_strct0 = rotate_structure(
            guest_strct, angles=360 * r_nrs[3], origin=np.zeros(3), vector=rot_v
        )

        guest_positions = np.array(guest_strct0["positions"])
        shift = np.array(r_nrs[4:7])
        shift = (cell.T).dot(shift)
        guest_positions += shift - min_pos
        guest_strct1 = guest_strct.copy()
        guest_strct1.set_positions(guest_positions)

        try:
            new_structure = _merge_structures(structure, guest_strct1, wrap)
        except SamePositionsError:
            continue
        new_indices = list(
            range(len(new_structure) - len(guest_strct["elements"]), len(new_structure))
        )

        is_added = _check_distances(new_structure, new_indices, None, distance_dict, True)
        if is_added:
            return new_structure, "_added-" + guest_strct_label
    raise DistanceThresholdError(
        "Could not add guest structure, host structure seems to be too aggregated."
    )


@external_manipulation_method
def add_structure_coord(
    structure: Structure,
    wrap: bool = False,
    host_indices: Union[int, List[int]] = 0,
    guest_index: int = 0,
    guest_structure: Union[Structure, str] = "CH3",
    guest_dir: Union[None, List[float]] = None,
    bond_length: float = 1.25,
    dist_constraints=None,
    dist_threshold: Union[dict, list, float, int, str, None] = 0.8,
    change_label: bool = False,
    **cn_kwargs,
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
        ``Structure`` object or the element symbol to add one single atom.
    guest_dir: list of floats (optional)
        Defines the orientation of the guest molecule. If not defined, a vector of nearest
        neighbors is constructed based on the guest index.
    bond_length : float
        Bond length between the host atom and the base atom of the functional group.
    dist_constraints : list (optional)
        List of three-fold tuples containing the index of the site of the host structure, the index
        of the site of the guest structure and the target distance. The position of the guest
        structure is varied based on a grid search until the sum of the absolute errors in
        minimized.
    dist_threshold : dict, list, float, int, str or None (optional)
        Check the distances between all site pairs of the host and guest structure to ensure that
        none of the added atoms collide or are too far apart. For example, ``0.8`` to ensure a
        minimum distance of ``0.8`` for all site pairs. A list ``[0.8, 1.5]`` adds a check for
        the maximum distance as well. Giving a dictionary ``{("C", "H"): 0.8, (0, 4): 0.8}``
        allows distance checks for individual pairs of elements or site indices. Specifying an
        atomic radius type as str, e.g. ``covalent+10`` sets the minimum threshold to the sum
        of covalent radii plus 10%.
    change_label : bool (optional)
        Add suffix to the label of the new structure highlighting the performed manipulation.
    cn_kwargs :
        Optional keyword arguments passed on to the ``calc_coordination`` function.

    Returns
    -------
    aim2dat.strct.Structure
        Structure with attached functional group.

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
    guest_strct, guest_strct_label = _check_guest_structure(guest_structure)
    if isinstance(host_indices, int):
        host_indices = [host_indices]

    if max(host_indices) >= len(structure) or guest_index >= len(guest_strct):
        idx = max(host_indices)
        return structure

    if dist_constraints is None:
        dist_constraints = []

    if guest_dir is None:
        guest_dir = [1.0, 0.0, 0.0]
        if len(guest_strct) > 1:
            # Get vector of guest atoms for rotation
            guest_strct_coord = guest_strct.calc_coordination(**cn_kwargs)
            guest_dir = -1.0 * _derive_bond_dir(guest_index, guest_strct_coord)
    guest_dir /= np.linalg.norm(np.array(guest_dir))

    # Calculate coordination:
    coord = structure.calc_coordination(**cn_kwargs)

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
        rot_angle = np.rad2deg(-calc_angle(guest_dir, bond_dir))
        if np.isclose(abs(rot_angle), 0.0) or np.isclose(abs(rot_angle), 180.0):
            guest_dir = create_lin_ind_vector(guest_dir)
        guest_strct = rotate_structure(
            structure=guest_strct,
            vector=np.cross(bond_dir, guest_dir),
            angles=rot_angle,
            origin=guest_strct["positions"][guest_index],
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
    ref_dir_beta = np.cross(bond_dir, create_lin_ind_vector(bond_dir))
    ref_dir_beta /= np.linalg.norm(ref_dir_beta)
    ref_dir_gamma = np.cross(bond_dir, ref_dir_beta)
    ref_dir_gamma /= np.linalg.norm(ref_dir_gamma)
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
    new_indices = list(
        range(len(new_structure) - len(guest_strct["elements"]), len(new_structure))
    )

    # Optimize positions to reduce score
    dist_dict, _ = _build_distance_dict(dist_threshold, structure, guest_strct)
    if len(dist_constraints) > 0:
        for alpha in np.linspace(0.0, 2.0 * np.pi, num=10):
            for beta in np.linspace(-1.0 * np.pi, 1.0 * np.pi, num=10):
                for gamma in np.linspace(-1.0 * np.pi, 1.0 * np.pi, num=10):
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
                        new_strct0, new_indices, None, dist_dict, True
                    ):
                        score = score0
                        new_structure = new_strct0
    else:
        _check_distances(new_structure, new_indices, None, dist_dict, False)
    return new_structure, "_added-" + guest_strct_label


@external_manipulation_method
def add_structure_position(
    structure: Structure,
    position: List[float],
    guest_structure: Union[Structure, str] = "CH3",
    wrap: bool = False,
    dist_threshold: Union[dict, list, float, int, str, None] = None,
    change_label: bool = False,
) -> Structure:
    """
    Add structure at a defined position.

    Parameters
    ----------
    structure : aim2dat.strct.Structure
        Structure to which the guest structure is added.
    position : list of floats
        Position of the guest structure.
    guest_structure : str or aim2dat.strct.Structure (optional)
        A representation of the guest structure given as a string of a functional group or molecule
        (viable options are ``'CH3'``, ``'COOH'``, ``'H2O'``, ``'NH2'``, ``'NO2'`` or ``'OH'``), a
        ``Structure`` object (bond direction is assumed to be the ``[-1.0, 0.0, 0.0]`` direction)
        or the element symbol to add one single atom.
    wrap : bool (optional)
        Wrap atomic positions back into the unit cell.
    dist_threshold : dict, list, float, int, str or None (optional)
        Check the distances between all site pairs of the host and guest structure to ensure that
        none of the added atoms collide or are too far apart. For example, ``0.8`` to ensure a
        minimum distance of ``0.8`` for all site pairs. A list ``[0.8, 1.5]`` adds a check for
        the maximum distance as well. Giving a dictionary ``{("C", "H"): 0.8, (0, 4): 0.8}``
        allows distance checks for individual pairs of elements or site indices. Specifying an
        atomic radius type as str, e.g. ``covalent+10`` sets the minimum threshold to the sum
        of covalent radii plus 10%.
    change_label : bool (optional)
        Add suffix to the label of the new structure highlighting the performed manipulation.

    Returns
    -------
    aim2dat.strct.Structure
        Structure with added sub structure at defined postion.

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
    guest_strct, guest_strct_label = _check_guest_structure(guest_structure)

    guest_positions = np.array(guest_strct["positions"])
    guest_center = np.mean(guest_positions, axis=0)
    guest_positions -= guest_center
    guest_positions += np.array(position)
    guest_strct0 = copy.deepcopy(guest_strct)
    guest_strct0.set_positions(guest_positions)

    new_structure = _merge_structures(structure, guest_strct0, wrap)
    new_indices = list(
        range(len(new_structure) - len(guest_strct["elements"]), len(new_structure))
    )
    _check_distances(new_structure, new_indices, dist_threshold, None, False)

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
                guest_strct_dict = read_yaml_file(
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
    for p0, ref_dir in zip(angle_pars, ref_dirs):
        guest_strct = rotate_structure(
            guest_strct, angles=p0 * 180.0 / np.pi, vector=ref_dir, origin=np.zeros(3)
        )
    guest_strct.set_positions(
        [np.array(pos) + bond_length * ref_dirs[0] + host_pos for pos in guest_strct.positions]
    )

    # Add guest structure to host:
    new_structure = _merge_structures(structure, guest_strct, wrap)

    # Evaluate complex based on the distance constraints:
    score = 0.0
    if len(dist_constraints) > 0:
        host_indices = [idx for idx, _, _ in dist_constraints]
        guest_indices = [idx + len(structure) for _, idx, _ in dist_constraints]
        ref_dists = [dist for _, _, dist in dist_constraints]
        dists = new_structure.calc_distance(host_indices, guest_indices, backfold_positions=True)
        if isinstance(dists, float):
            dists = [dists]
        elif isinstance(dists, dict):
            dists = [dists[tuple(idx)] for idx in zip(list(host_indices), list(guest_indices))]
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
