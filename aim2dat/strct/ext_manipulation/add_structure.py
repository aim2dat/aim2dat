"""
Module that implements routines to add a functional group or adsorbed molecule to a structure.
"""

# Standard library imports
import os
import copy
from typing import Union, List
from itertools import product, combinations

# Third party library imports
import numpy as np

# Internal library imports
from aim2dat.strct.structure import Structure
from aim2dat.strct.validation import SamePositionsError
from aim2dat.strct.ext_manipulation.decorator import external_manipulation_method
from aim2dat.strct.ext_manipulation.utils import (
    _build_distance_dict,
    _check_distances,
    DistanceThresholdError,
)
from aim2dat.strct.ext_manipulation.rotate_structure import rotate_structure
from aim2dat.strct.analysis.geometry import _calc_atomic_distance
from aim2dat.elements import get_element_symbol
from aim2dat.utils.maths import (
    calc_angle,
    create_lin_ind_vector,
)


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
        ``dist_threshold`` needs to have keys with length 2 containing site indices or element
        symbols.
    ValueError
        ``dist_threshold`` needs to have keys of type List[str/int] containing site indices or
        element symbols.
    TypeError
        ``dist_threshold`` needs to be of type int/float/list/tuple/dict or None.
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

        if _check_distances(new_structure, new_indices, None, distance_dict, True):
            return new_structure, "_added-" + guest_strct_label
    raise DistanceThresholdError(
        "Could not add guest structure, host structure seems to be too aggregated."
    )


@external_manipulation_method
def add_structure_coord(
    structure: Structure,
    wrap: bool = False,
    host_indices: Union[int, List[int]] = 0,
    guest_indices: Union[int, List[int]] = 0,
    guest_structure: Union[Structure, str] = "CH3",
    rotate_guest: bool = False,
    bond_length: float = 1.25,
    constrain_steps: int = 90,
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
    guest_indices : int or list (optional)
        Index or indices of the guest site. In case several indices are given the center of the
        sites will face the host structure. The rest of the guest will point away.
    guest_structure : str or aim2dat.strct.Structure (optional)
        A representation of the guest structure given as a string of a functional group or molecule
        (viable options are ``'CH3'``, ``'COOH'``, ``'H2O'``, ``'NH2'``, ``'NO2'`` or ``'OH'``), a
        ``Structure`` object or the element symbol to add one single atom.
    bond_length : float
        Bond length between the host atom and the base atom of the functional group.
    rotate_guest : bool (optional)
        The rotation of the guest structure is varied based on a grid search until the sum of the
        absolute errors is minimized.
    constrain_steps : int (optional)
        Rotation steps in degree to consider for the grid search in ``rot_guest``.
        Additionally, if the first position guess of the molecule does not match the
        ```dist_threshold```, a grid search around the host atom is utilized until the sum of the
        absolute errors is minimized.
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
        ``dist_threshold`` needs to have keys with length 2 containing site indices or element
        symbols.
    ValueError
        ``dist_threshold`` needs to have keys of type List[str/int] containing site indices or
        element symbols.
    TypeError
        ``dist_threshold`` needs to be of type int/float/list/tuple/dict or None.
    ValueError
        If any distance between atoms is outside the threshold.
    """
    # Checks
    guest_strct, guest_strct_label = _check_guest_structure(guest_structure)
    dist_dict, _ = _build_distance_dict(dist_threshold, structure, guest_strct)
    if isinstance(host_indices, int):
        host_indices = [host_indices]
    if isinstance(guest_indices, int):
        guest_indices = [guest_indices]

    if max(host_indices) >= len(structure) or max(guest_indices) >= len(guest_strct):
        return structure, ""

    if len(guest_strct) == 1:
        guest_dir = [1.0, 0.0, 0.0]
    else:
        # Get vector of guest atoms for rotation
        guest_dir, guest_center, guest_positions = _derive_bond(
            guest_strct, guest_indices, cn_kwargs
        )
        guest_dir *= -1.0
        guest_strct.set_positions(np.array(guest_strct.positions) - guest_center)
    guest_dir /= np.linalg.norm(np.array(guest_dir))

    # Calculate coordination:

    # Derive bond directions
    bond_dir, host_center, host_positions = _derive_bond(
        structure, host_indices, cn_kwargs, bond_length
    )

    # Check bond length and adjusts if necessary
    if all([np.linalg.norm(host_center - bond_pos) >= bond_length for bond_pos in host_positions]):
        bond_length = 0.0
    else:
        if len(host_indices) > 1:
            bond_length = _rescale_bond_length(host_center, host_positions, bond_dir, bond_length)
        if len(guest_indices) > 1:
            bond_length = _rescale_bond_length(
                guest_center, guest_positions, guest_dir, bond_length
            )

    # Rotate guest to align the x-axis
    if len(guest_strct) > 1:
        rot_angle = np.rad2deg(-calc_angle(guest_dir, [1, 0, 0]))
        if np.isclose(abs(rot_angle), 180.0):
            guest_dir = create_lin_ind_vector(guest_dir)
        if not np.isclose(abs(rot_angle), 0.0):
            guest_pos = np.array(guest_strct.positions)
            origin = np.mean(guest_pos[guest_indices], axis=0)
            guest_strct = rotate_structure(
                structure=guest_strct,
                vector=np.cross([1, 0, 0], guest_dir),
                angles=rot_angle,
                origin=origin,
            )

    # Create new structure
    new_structure, new_guest = _add_mol(
        structure,
        guest_strct,
        wrap,
        host_center,
        bond_length,
        [0.0, 0.0, 0.0],
        bond_dir,
    )
    new_indices = list(range(len(new_structure) - len(guest_strct), len(new_structure)))

    # Optimize positions to reduce score
    # Numbers for the grid search
    num1 = round(360 / constrain_steps)
    num2 = round(num1 / 2)
    score = _check_distances(new_structure, new_indices, None, dist_dict, True, True)
    # If the distance threshold does not hold, a grid search will be performed.
    # The grid is based on `constrain_steps` and rotates the molecule around `host_indices` within
    # a sphere with the ´bond_distance´ as rotation vector.
    if isinstance(score, bool) and not score:
        score = score if score else float("inf")
        for alpha in np.linspace(-180, 180, num=num1, endpoint=False):
            for beta in np.linspace(0, 180, num=num2, endpoint=False):
                try:
                    new_strct0, new_guest0 = _add_mol(
                        structure,
                        guest_strct,
                        wrap,
                        host_center,
                        bond_length,
                        [0.0, alpha, beta],
                        bond_dir,
                    )
                except Exception:
                    continue
                score0 = _check_distances(new_strct0, new_indices, None, dist_dict, True, True)
                if isinstance(score0, float) and score0 < score:
                    score = score0
                    new_structure = new_strct0
                    new_guest = new_guest0

    # A grid search will be performed when `rotate_guest` is set to `True`.
    # The grid is based on `constrain_steps` and rotates the molecule around its own axis.
    if rotate_guest:
        score = score if score else float("inf")
        # Prepare guest to rotate. Need to align guest with x-axis.
        origin = np.mean(new_guest.positions, axis=0)
        guest_vec = origin - host_center
        rot_angle = np.rad2deg(-calc_angle(guest_vec, [1, 0, 0]))
        if np.isclose(abs(rot_angle), 180.0):
            guest_vec = create_lin_ind_vector(guest_vec)
        if not np.isclose(abs(rot_angle), 0.0):
            new_guest = rotate_structure(
                new_guest,
                rot_angle,
                np.cross([1, 0, 0], guest_vec),
                origin,
            )

        for alpha, beta in product(np.linspace(-180, 180, num=num1, endpoint=False), repeat=2):
            for gamma in np.linspace(0, 180, num=num2, endpoint=False):
                new_guest0 = rotate_structure(
                    new_guest,
                    [alpha, beta, gamma],
                    origin=origin,
                )
                try:
                    # Need to align guest with original position.
                    new_guest0 = rotate_structure(
                        new_guest0,
                        -rot_angle,
                        np.cross([1, 0, 0], guest_vec),
                        origin,
                    )
                    new_strct0 = _merge_structures(structure, new_guest0, wrap)
                except Exception:
                    continue
                score0 = _check_distances(new_strct0, new_indices, None, dist_dict, True, True)
                if isinstance(score0, float) and score0 < score:
                    score = score0
                    new_structure = new_strct0

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
        Structure with added sub structure at defined position.

    Raises
    ------
    ValueError
        ``dist_threshold`` needs to have keys with length 2 containing site indices or element
        symbols.
    ValueError
        ``dist_threshold`` needs to have keys of type List[str/int] containing site indices or
        element symbols.
    TypeError
        ``dist_threshold`` needs to be of type int/float/list/tuple/dict or None.
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
        try:
            strct = Structure(
                label=guest_strct,
                elements=[get_element_symbol(guest_strct)],
                positions=[[0.0, 0.0, 0.0]],
                pbc=False,
            )
        except ValueError:
            try:
                strct = Structure.from_str(guest_strct)
            except ValueError:
                raise ValueError(f"``guest_structure`` '{guest_strct}' is not supported.")
        return strct, guest_strct
    else:
        raise TypeError("``guest_structure`` needs to be of type Structure or str.")


def _derive_bond(structure, index, cn_kwargs, bond_length=None):
    bond_direction = np.zeros(3)
    bond_positions = []
    if isinstance(index, int):
        index = [index]
    coord = structure.calc_coordination(indices=index, get_statistics=False, **cn_kwargs)
    for idx, cn_details in zip(index, coord):
        bond_dir = np.zeros(3)
        pos = np.array(cn_details["position"])
        all_pos = np.array(structure.positions)
        if not bond_positions:
            bond_positions.append(pos)
        else:
            _, bond_pos = _calc_atomic_distance(structure, index[0], idx, backfold_positions=True)
            bond_positions.append(bond_pos)
        for neigh in cn_details["neighbours"]:
            dir_v = np.array(neigh["position"]) - pos
            bond_dir += dir_v / np.linalg.norm(dir_v)
        # We need to ensure that the bond direction is not zero.
        if len(cn_details["neighbours"]) == 2 and np.linalg.norm(bond_dir) < 1e-1:
            bond_dir = np.cross(
                create_lin_ind_vector(np.array(cn_details["neighbours"][0]["position"]) - pos),
                np.array(cn_details["neighbours"][0]["position"]) - pos,
            )
        # We need to ensure that the bond direction is not zero.
        elif np.linalg.norm(bond_dir) < 1e-1:
            max_dist = 0.0
            for neigh1, neigh2 in combinations(cn_details["neighbours"], 2):
                cross_dir = np.cross(
                    np.array(neigh1["position"]) - pos,
                    np.array(neigh2["position"]) - pos,
                )
                if np.linalg.norm(cross_dir) < 1e-1:
                    continue
                pos1 = pos + cross_dir / np.linalg.norm(cross_dir) * bond_length
                diffs = all_pos - pos1
                dists = np.linalg.norm(diffs, axis=1)
                if min(dists) > max_dist:
                    bond_dir = cross_dir
        bond_dir *= -1.0 / np.linalg.norm(bond_dir)
        bond_direction += bond_dir
    bond_position_center = np.mean(np.array(bond_positions), axis=0)
    bond_direction /= np.linalg.norm(bond_direction)
    return bond_direction, bond_position_center, bond_positions


def _rescale_bond_length(atom_center, atom_positions, bond_dir, bond_length):
    # Rescale bond length to match the distance between guest molecule and host atoms
    scaled_bond_lengths = []
    for atom_pos in atom_positions:
        # Calculate the coefficients for the quadratic equation
        dis_atom_pos = atom_center - atom_pos
        bond_dir_square = np.dot(bond_dir, bond_dir)
        bond_dir_dot_dis_atom_pos = np.dot(bond_dir, dis_atom_pos)
        dis_atom_pos_square = np.dot(dis_atom_pos, dis_atom_pos)
        # Coefficients of the quadratic equation at^2 + bt + c = 0
        a = bond_dir_square
        b = 2 * bond_dir_dot_dis_atom_pos
        c = dis_atom_pos_square - bond_length**2
        # Solve the quadratic equation
        t1, t2 = np.roots([a, b, c])
        scaled_bond_lengths.append(max([t1, t2]))

    for sbl in scaled_bond_lengths:
        ref_position = atom_center + sbl * bond_dir
        if all(
            [
                np.linalg.norm(ref_position - atom_pos) >= bond_length * 0.95
                for atom_pos in atom_positions
            ]
        ):
            bond_length = sbl
            break
    return bond_length


def _add_mol(
    structure,
    guest_strct,
    wrap,
    host_pos,
    bond_length,
    angles,
    bond_dir,
):
    guest_strct0 = guest_strct.copy()
    guest_pos = np.array(guest_strct.positions) + bond_length * np.array([1, 0, 0])
    # Shift guest structure:
    guest_strct0.set_positions(guest_pos)
    # Rotate if ``dist_threshold`` is not satisfied
    guest_strct0 = rotate_structure(
        guest_strct0,
        angles,
        origin=np.zeros(3),
    )

    # Rotate guest to align bond direction
    rot_angle = np.rad2deg(-calc_angle([1, 0, 0], bond_dir))
    if np.isclose(abs(rot_angle) % 180, 0.0):
        rot_dir = np.cross(create_lin_ind_vector(bond_dir), [1, 0, 0])
    else:
        rot_dir = np.cross(bond_dir, [1, 0, 0])
    guest_strct0 = rotate_structure(
        guest_strct0,
        rot_angle,
        rot_dir,
        np.zeros(3),
    )
    guest_strct0.set_positions(guest_strct0.positions + host_pos)

    # Add guest structure to host:
    new_structure = _merge_structures(structure, guest_strct0, wrap)
    return new_structure, guest_strct0


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
