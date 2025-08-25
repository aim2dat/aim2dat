"""Functions to generate surfaces."""

# Standard library imports
import copy
from typing import Union, List, Tuple
import decimal

# Third party library imports
import numpy as np
from scipy.spatial.transform import Rotation

# Internal library imports
from aim2dat.strct.structure import Structure
from aim2dat.strct.validation import _structure_validate_cell
from aim2dat.ext_interfaces.ase_surface import _create_ase_surface_from_structure
from aim2dat.ext_interfaces.ase_atoms import _extract_structure_from_atoms
from aim2dat.ext_interfaces.spglib import _space_group_analysis
from aim2dat.utils.maths import calc_angle, calc_reflection_matrix
from aim2dat.chem_f import transform_list_to_dict, reduce_formula


SPACE_GROUP_HN_TO_LAYER_GROUP = {
    1: 1,
    2: 2,
    4: 3,
    19: 4,
    24: 5,
    25: 5,
    26: 5,
    58: 6,
    75: 7,
    76: 7,
    77: 7,
    5: 8,
    3: 8,
    8: 9,
    6: 9,
    16: 10,
    9: 10,
    20: 11,
    18: 11,
    27: 12,
    23: 12,
    37: 13,
    30: 13,
    59: 14,
    57: 14,
    62: 15,
    60: 15,
    78: 16,
    74: 16,
    81: 17,
    87: 17,
    83: 17,
    70: 18,
    63: 18,
    108: 19,
    110: 20,
    111: 20,
    112: 21,
    119: 22,
    125: 23,
    137: 24,
    138: 24,
    161: 25,
    173: 26,
    127: 27,
    126: 27,
    128: 28,
    133: 28,
    130: 28,
    132: 29,
    131: 29,
    136: 30,
    135: 30,
    142: 31,
    139: 31,
    155: 32,
    156: 32,
    157: 32,
    158: 32,
    159: 32,
    160: 32,
    157: 32,
    148: 33,
    145: 33,
    153: 34,
    152: 34,
    189: 35,
    188: 35,
    195: 36,
    194: 36,
    227: 37,
    231: 38,
    232: 38,
    234: 39,
    244: 40,
    241: 40,
    239: 41,
    240: 41,
    256: 42,
    253: 42,
    259: 43,
    262: 43,
    263: 44,
    273: 45,
    272: 45,
    278: 46,
    279: 46,
    310: 47,
    316: 48,
    349: 49,
    355: 50,
    357: 51,
    359: 52,
    360: 52,
    366: 53,
    367: 54,
    376: 55,
    377: 56,
    388: 57,
    390: 58,
    392: 59,
    394: 60,
    400: 61,
    402: 62,
    403: 62,
    406: 63,
    408: 64,
    409: 64,
    430: 65,
    435: 66,
    438: 67,
    439: 68,
    446: 69,
    447: 70,
    454: 71,
    456: 72,
    462: 73,
    468: 74,
    469: 75,
    471: 76,
    477: 77,
    481: 78,
    483: 79,
    485: 80,
}


def _surface_create(
    sg_details: dict, miller_indices: Union[List[int], Tuple[int]], ter: int, tol: float
) -> dict:
    # Create surface slabs:
    structure = sg_details["standardized_structure"]
    surf_1l = _create_ase_surface_from_structure(
        Structure(**structure), miller_indices, 1, 0.0, True
    )
    surf_2l = _create_ase_surface_from_structure(
        Structure(**structure), miller_indices, 2, 0.0, True
    )
    cell_1l = surf_1l.cell[:]
    cell_1l[2] = surf_2l.cell[:][2] - cell_1l[2]
    surf_1l.set_cell(cell_1l)

    # Determine horizontal shift vector of the repeating unit:
    dec_tol = abs(decimal.Decimal(str(tol)).as_tuple().exponent)
    positions_1l = surf_1l.get_scaled_positions(wrap=True)
    positions_2l = surf_2l.get_scaled_positions(wrap=True)
    shifts = [
        np.round(positions_2l[len(surf_1l) + idx] - positions_1l[idx], dec_tol + 1)
        for idx in range(len(surf_1l))
    ]
    for shift_idx, shift in enumerate(shifts):
        shift[2] = 0.0
        for dir0 in range(2):
            while shift[dir0] < 0.0:
                shift[dir0] += 1.0
            while shift[dir0] > 1.0:
                shift[dir0] -= 1.0
        shifts[shift_idx] = shift
    if any(np.linalg.norm(shift - shifts[0]) > tol for shift in shifts):
        raise TypeError("Could not determine shift vector between two slab-layers.")

    # Create rpeating structure
    rep_structure = _extract_structure_from_atoms(surf_1l)
    del rep_structure["attributes"]
    del rep_structure["site_attributes"]
    rep_cell_np, rep_inv_cell_np = _structure_validate_cell(rep_structure["cell"])
    backfolded_positions = []
    for pos in rep_structure["positions"]:
        if rep_structure["is_cartesian"]:
            pos_scaled = np.transpose(rep_inv_cell_np).dot(np.array(pos))
        else:
            pos_scaled = np.array(pos)
        for direction in range(3):
            while pos_scaled[direction] < tol:
                pos_scaled[direction] += 1.0
            while pos_scaled[direction] >= 1.0 - tol:
                pos_scaled[direction] -= 1.0
        backfolded_positions.append(np.transpose(rep_cell_np).dot(np.array(pos_scaled)).tolist())
    rep_structure["positions"] = backfolded_positions
    rep_structure["is_cartesian"] = True
    rep_structure["translational_vector"] = np.transpose(cell_1l).dot(np.array(shifts[0])).tolist()
    del rep_structure["pbc"]
    del rep_structure["kinds"]
    ter_pos = _determine_max_terminations(
        surf_1l, sg_details["space_group"], rep_structure["translational_vector"], tol
    )
    if ter > len(ter_pos):
        return None

    # Create bottom termination
    bot_structure = {
        "elements": [],
        "positions": [],
        "cell": surf_1l.cell[:].tolist(),
        "is_cartesian": True,
    }
    top_structure_nsym = {
        "elements": [],
        "positions": [],
        "cell": surf_1l.cell[:].tolist(),
        "is_cartesian": True,
    }
    bot_structure["cell"][2][2] -= ter_pos[ter - 1]
    for el, position in zip(rep_structure["elements"], rep_structure["positions"]):
        if position[2] - ter_pos[ter - 1] > -tol:
            pos = np.array(position) - np.array([0.0, 0.0, ter_pos[ter - 1]])
            bot_structure["elements"].append(el)
            bot_structure["positions"].append(pos.tolist())
        else:
            top_structure_nsym["elements"].append(el)
            top_structure_nsym["positions"].append(position)
    if len(top_structure_nsym["positions"]) == 0:
        top_structure_nsym = None
    else:
        top_structure_nsym["cell"][2][2] = max(
            [pos[2] for pos in top_structure_nsym["positions"]] + [0.01]
        )

    top_structure = _determine_top_structure(
        copy.deepcopy(bot_structure), copy.deepcopy(rep_structure), tol
    )
    if top_structure is None:
        print("No symmetric termination could be found.")
        return None
    else:
        return {
            "repeating_structure": rep_structure,
            "bottom_structure": bot_structure,
            "top_structure": top_structure,
            "top_structure_nsym": top_structure_nsym,
        }


def _surface_create_slab(
    surface: dict,
    nr_layers: int,
    periodic: bool,
    vacuum: float,
    vacuum_factor: float,
    symmetrize: bool,
) -> dict:
    rep_cell = np.array(surface["repeating_structure"]["cell"])
    bot_cell = np.array(surface["bottom_structure"]["cell"])
    if symmetrize:
        top_strct = surface["top_structure"]
        top_cell = np.array(top_strct["cell"])
    elif surface["top_structure_nsym"] is None:
        top_strct = {"positions": [], "elements": []}
        top_cell = np.zeros((3, 3))
    else:
        top_strct = surface["top_structure_nsym"]
        top_cell = np.array(top_strct["cell"])
    nr_layers -= 1
    trans_vector = np.array(surface["repeating_structure"]["translational_vector"])
    c_vector = bot_cell[2] + rep_cell[2] * nr_layers + top_cell[2]
    if vacuum_factor > 0.0:
        vacuum_v = c_vector * vacuum_factor
    else:
        vacuum_v = np.array([0.0, 0.0, vacuum])
    c_vector += vacuum_v * 2

    surface_slab = {
        "elements": list(surface["bottom_structure"]["elements"]),
        "positions": [
            np.array(pos0) + vacuum_v for pos0 in surface["bottom_structure"]["positions"]
        ],
        "pbc": [True, True, periodic],
        "cell": list(surface["repeating_structure"]["cell"]),
        "is_cartesian": True,
        "wrap": True,
    }
    surface_slab["cell"][2] = c_vector.tolist()

    for layer in range(nr_layers):
        for element, position in zip(
            surface["repeating_structure"]["elements"],
            surface["repeating_structure"]["positions"],
        ):
            final_pos = (
                np.array(position)
                + bot_cell[2]
                + rep_cell[2] * layer
                + trans_vector * (layer + 1)
                + vacuum_v
            )
            surface_slab["positions"].append(final_pos.tolist())
            surface_slab["elements"].append(element)
    for element, position in zip(top_strct["elements"], top_strct["positions"]):
        final_pos = (
            np.array(position)
            + bot_cell[2]
            + rep_cell[2] * nr_layers
            + trans_vector * (nr_layers + 1)
            + vacuum_v
        )
        surface_slab["positions"].append(final_pos.tolist())
        surface_slab["elements"].append(element)
    return surface_slab


def _create_reflection_matrix(n_vector: np.ndarray) -> np.ndarray[:, :]:
    n_vector = np.array(n_vector)
    n_vector /= np.linalg.norm(n_vector)
    sigma = np.zeros((3, 3))
    for dir0 in range(3):
        sigma[dir0, dir0] = 1.0 - 2.0 * n_vector[dir0] ** 2.0
        sigma[dir0, dir0 - 2] = -2.0 * n_vector[dir0] * n_vector[dir0 - 2]
        sigma[dir0 - 2, dir0] = -2.0 * n_vector[dir0] * n_vector[dir0 - 2]
    return sigma


def _transform_slab_to_primitive(slab, symprec, angle_tolerance, hall_number, aperiodic_dir=2):
    if isinstance(slab, dict):
        slab = Structure(**slab)

    # Based on Litvin and Wike
    spglib_output = _space_group_analysis(
        slab,
        symprec,
        angle_tolerance,
        hall_number,
        return_primitive_structure=True,
        return_sym_operations=True,
        no_idealize=True,
    )

    # Define vectors:
    periodic_dirs = [dir0 for dir0 in range(3) if dir0 != aperiodic_dir]
    plane_v = np.zeros(3)
    for pdir0 in periodic_dirs:
        plane_v += np.array(slab["cell"][pdir0])
    plane_v /= np.linalg.norm(plane_v)
    aperiodic_v = np.array(slab["cell"][aperiodic_dir])
    aperiodic_v /= np.linalg.norm(aperiodic_v)

    # Transform unit cell:
    transf_matrix = spglib_output["transformation_matrix"]
    prim_cell = np.array(spglib_output["primitive_structure"]["cell"])
    reor_matrix = np.identity(3)
    if round(abs(transf_matrix[aperiodic_dir][aperiodic_dir]), 6) != 1.0:
        for row_idx in range(3):
            if abs(round(transf_matrix[row_idx][aperiodic_dir], 6)) == 1.0:
                break
            elif transf_matrix[row_idx][aperiodic_dir] > 1e-6:
                raise ValueError("Cannot transform to primitive slab.")
        aperiodic_cell_v = prim_cell[aperiodic_dir].copy()
        prim_cell[aperiodic_dir] = prim_cell[row_idx].copy()
        prim_cell[row_idx] = aperiodic_cell_v

    # Aperiodic direction should point into the original direction:
    if calc_angle(prim_cell[aperiodic_dir], slab["cell"][aperiodic_dir]) > 1e-6:
        n_vector = np.cross(prim_cell[periodic_dirs[0]], prim_cell[periodic_dirs[1]])
        reflection_plane = _create_reflection_matrix(n_vector)
        prim_cell = np.dot(reflection_plane, prim_cell.T).T
        reor_matrix = np.dot(reflection_plane, reor_matrix)
    # rot_angle = calc_angle(prim_cell[aperiodic_dir], slab["cell"][aperiodic_dir])
    # rot = Rotation.from_rotvec(rot_angle * plane_v)
    # prim_cell = np.dot(rot.as_matrix(), prim_cell.T).T
    # reor_matrix = np.dot(rot.as_matrix(), reor_matrix)

    # First periodic vector should point into the [1 0 0]:
    target_v = np.array([1.0 if dir0 == periodic_dirs[0] else 0.0 for dir0 in range(3)])
    rot_v = np.cross(prim_cell[periodic_dirs[0]], target_v)
    if np.linalg.norm(rot_v) > 1e-6:
        rot_v /= np.linalg.norm(rot_v)
        rot_angle = calc_angle(prim_cell[periodic_dirs[0]], target_v)
        rot = Rotation.from_rotvec(rot_angle * rot_v)
        prim_cell = np.dot(rot.as_matrix(), prim_cell.T).T
        reor_matrix = np.dot(rot.as_matrix(), reor_matrix)

    # Second periodic vector should have mostly positive values:
    if all(val < 1.0e-6 for val in prim_cell[periodic_dirs[1]]):
        ref_matrix = calc_reflection_matrix(
            np.cross(prim_cell[periodic_dirs[0]], prim_cell[aperiodic_dir])
        )
        prim_cell = np.dot(ref_matrix, prim_cell.T).T
        reor_matrix = np.dot(ref_matrix, reor_matrix)

    # Delete numerical noise:
    prim_cell[np.abs(prim_cell) < 1e-6] = 0.0

    # Transform atom sites:
    if "positions" in slab:
        inp_pos_cart = np.array(slab["positions"])
        if "is_cartesian" in slab and not slab["is_cartesian"]:
            inp_pos_cart = np.array(
                [np.transpose(slab["cell"]).dot(np.array(pos)) for pos in slab["positions"]]
            )
    else:
        raise ValueError("Cartesian positions could not be retrieved.")

    used_sites = []
    prim_positions = []
    prim_elements = []
    inv_prim_cell = np.linalg.inv(prim_cell)
    for el, pos, mapping in zip(
        slab["elements"], inp_pos_cart, spglib_output["mapping_to_primitive"]
    ):
        if mapping in used_sites:
            continue
        prim_pos = np.transpose(inv_prim_cell).dot((reor_matrix.dot(pos.T)).T)
        for direction in range(3):
            while prim_pos[direction] < 0.0:
                prim_pos[direction] += 1.0
            while prim_pos[direction] >= 1.0:
                prim_pos[direction] -= 1.0
        prim_pos[np.abs(prim_pos) < 1e-6] = 0.0
        prim_elements.append(el)
        prim_positions.append(prim_pos)
        used_sites.append(mapping)
    prim_slab = {
        "cell": prim_cell,
        "positions": prim_positions,
        "elements": prim_elements,
        "is_cartesian": False,
        "pbc": True,
    }
    layer_group = SPACE_GROUP_HN_TO_LAYER_GROUP[spglib_output["space_group"]["hall_number"]]
    # TODO periodic boundary conditions...
    return prim_slab, layer_group


def _determine_max_terminations(surf_1l, centrosymmetric, trans_v, tol):
    if np.linalg.norm(trans_v) > tol:
        centrosymmetric = False
    z_pos = {}
    for atom in surf_1l:
        added_atom = False
        for z_pos0 in z_pos.keys():
            if abs(atom.position[2] - z_pos0) < tol:
                added_atom = True
                z_pos[z_pos0].append(atom.symbol)
                break
        if not added_atom:
            z_pos[atom.position[2]] = [atom.symbol]

    if centrosymmetric:
        z_pos = {k0: reduce_formula(transform_list_to_dict(v0)) for k0, v0 in z_pos.items()}
        z_pos_cs = {k0: v0 for k0, v0 in z_pos.items() if k0 < 0.5 * surf_1l.cell[:][2][2] - tol}
        for z_pos0, formula in z_pos.items():
            if z_pos0 not in z_pos_cs and formula not in list(z_pos_cs.values()):
                z_pos_cs[z_pos0] = formula
        z_pos = z_pos_cs
    return list(z_pos.keys())


def _determine_top_structure(bot_structure, rep_structure, tol):
    def create_sorted_z_positions(positions, tol):
        zpos = []
        for pos in positions:
            if all(abs(pos[2] - zpos0) > tol for zpos0 in zpos):
                zpos.append(pos[2])
        zpos.sort()
        return zpos

    def partition_pos_by_z(zpos, structure, tol):
        elements_part = []
        positions_part = []
        for el, pos in zip(structure["elements"], structure["positions"]):
            if abs(pos[2] - zpos) < tol:
                elements_part.append(el)
                positions_part.append(pos)
        return elements_part, positions_part

    zpositions_bot = create_sorted_z_positions(bot_structure["positions"], tol)
    zpositions_rep = create_sorted_z_positions(rep_structure["positions"], tol)
    bot_elements = []
    bot_positions = []
    bot_distances = []
    bot_z_distances = zpositions_bot
    top_elements = []
    top_positions = []
    top_distances = []
    for zpos_bot in zpositions_bot:
        bot_new_el, bot_new_pos = partition_pos_by_z(zpos_bot, bot_structure, tol)
        bot_dist = _calculate_interatomic_dist(bot_new_el, bot_new_pos, bot_structure["cell"])
        bot_elements.append(bot_new_el)
        bot_positions.append(bot_new_pos)
        bot_distances.append(bot_dist)
    for zpos_rep in zpositions_rep:
        rep_new_el, rep_new_pos = partition_pos_by_z(zpos_rep, rep_structure, tol)
        rep_dist = _calculate_interatomic_dist(rep_new_el, rep_new_pos, rep_structure["cell"])
        top_elements.append(rep_new_el)
        top_positions.append(rep_new_pos)
        top_distances.append(rep_dist)
        if zpos_rep + bot_structure["cell"][2][2] < rep_structure["cell"][2][2]:
            bot_elements.append(rep_new_el)
            bot_positions.append(rep_new_pos)
            bot_distances.append(rep_dist)
            bot_z_distances.append(zpos_rep + bot_structure["cell"][2][2])
    top_start_idx = len(top_elements)

    comp_layers = None
    for zpos_idx, zpos_rep in enumerate(zpositions_rep):
        top_z_distances = [
            zpos_rep - zpositions_rep[idx0] for idx0 in reversed(range(zpos_idx + 1))
        ]
        top_z_distances += [
            zpos_rep + rep_structure["cell"][2][2] - zpositions_rep[idx0]
            for idx0 in reversed(range(len(zpositions_rep)))
        ]
        top_new_el, top_new_pos = partition_pos_by_z(zpos_rep, rep_structure, tol)
        top_dists = _calculate_interatomic_dist(top_new_el, top_new_pos, rep_structure["cell"])
        top_elements.append(top_new_el)
        top_positions.append(top_new_pos)
        top_distances.append(top_dists)
        if any(abs(topd - botd) >= tol for topd, botd in zip(top_z_distances, bot_z_distances)):
            continue
        comp_layers = []
        for idx0 in range(len(bot_elements)):
            comp_layers.append(
                _compare_interatomic_distances(
                    bot_distances[idx0], top_distances[len(top_elements) - idx0 - 1], tol
                )
            )
        if all(comp_layers):
            break
    if comp_layers is None or not all(comp_layers):
        return None
    top_structure = {
        "positions": [],
        "elements": [],
        "cell": rep_structure["cell"],
        "is_cartesian": True,
    }
    for top_el, top_pos in zip(top_elements[top_start_idx:], top_positions[top_start_idx:]):
        top_structure["elements"] += top_el
        top_structure["positions"] += top_pos
    top_structure["cell"][2][2] = max([pos[2] for pos in top_structure["positions"]] + [0.01])
    return top_structure


def _calculate_interatomic_dist(
    elements: List[str], positions: List[List[float]], cell: List[List[float]]
) -> List[dict]:
    def backfold_position(position, cell_np, inv_cell_np):
        pos_scaled = np.transpose(inv_cell_np).dot(np.array(position))
        for direction in range(2):
            while pos_scaled[direction] < 0.0:
                pos_scaled[direction] += 1.0
            while pos_scaled[direction] >= 1.0:
                pos_scaled[direction] -= 1.0
        return np.transpose(cell_np).dot(np.array(pos_scaled))

    cell_np = np.array(cell)
    inv_cell_np = np.linalg.inv(cell_np)
    distances = []
    pref = [-1.0, 0.0, 1.0]
    for idx0, (el, pos) in enumerate(zip(elements, positions)):
        pos0 = backfold_position(pos, cell_np, inv_cell_np)
        for idx1 in range(idx0):
            pos1 = backfold_position(positions[idx1], cell_np, inv_cell_np)
            dists = []
            for dir0 in range(3):
                for dir1 in range(3):
                    dists.append(
                        np.linalg.norm(
                            pos0 - pos1 + pref[dir0] * cell_np[0] + pref[dir1] * cell_np[1]
                        )
                    )
            distances.append(
                {
                    "elements": (el, elements[idx1]),
                    "distance": min(dists),
                    "positions": (pos0, pos1),
                }
            )
    return distances


def _compare_interatomic_distances(
    distances1: List[dict], distances2: List[dict], tol: float
) -> bool:
    used_dist_items = []
    for dist1 in distances1:
        found_match = False
        for dist2_idx, dist2 in enumerate(distances2):
            if dist2_idx in used_dist_items:
                continue
            if (
                dist1["elements"][0] == dist2["elements"][0]
                and dist1["elements"][1] == dist2["elements"][1]
                or dist1["elements"][0] == dist2["elements"][1]
                and dist1["elements"][1] == dist2["elements"][0]
            ):
                if abs(dist1["distance"] - dist2["distance"]) < tol:
                    found_match = True
                    used_dist_items.append(dist2_idx)
                    break
        if not found_match:
            return False
    return True
