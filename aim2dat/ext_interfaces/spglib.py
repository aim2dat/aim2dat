"""Wrapper functions for spglib."""

# Standard library imports
from typing import Union

# Third party library imports
import numpy as np
import spglib
from scipy.spatial.transform import Rotation

# Internal library imports
from aim2dat.elements import get_atomic_number, get_element_symbol
from aim2dat.utils.maths import calc_angle, calc_reflection_matrix


CENTROSYMMETRIC_PG = [
    "-1",
    "2/m",
    "mmm",
    "4/m",
    "4/mmm",
    "-3",
    "-3m",
    "6/m",
    "6/mmm",
    "m-3",
    "m-3m",
]


def _transfrom_structure_to_cell(structure):
    if structure.cell is None:
        raise ValueError("`cell` needs to be set.")
    cell = (
        structure.cell,
        structure.scaled_positions,
        [get_atomic_number(el) for el in structure["elements"]],
    )
    return cell


def _transform_cell_to_structure(cell):
    return {
        "cell": cell[0],
        "positions": cell[1],
        "elements": [get_element_symbol(el) for el in cell[2]],
        "is_cartesian": False,
        "pbc": True,
    }


def _transform_spglib_output_to_dict(output, keys):
    if not isinstance(output, dict):
        output = {key: getattr(output, key) for key in keys}
    return output


def _get_space_group_details(space_group: Union[int, str], return_sym_operations: bool):
    str_keys = [
        "international_short",
        "international_full",
        "international",
        "schoenflies",
        "hall_symbol",
    ]
    is_int = True
    if isinstance(space_group, str):
        is_int = False
        space_group = space_group.lower().replace(" ", "")

    output_dict = {}
    for hall_nr in range(1, 531):
        output_dict = _get_space_group_type(hall_nr)
        if (is_int and space_group == output_dict["number"]) or (
            space_group in [output_dict[key].lower().replace(" ", "") for key in str_keys]
        ):
            if return_sym_operations:
                output_dict["symmetry_operations"] = _get_symmetry_from_database(hall_nr)
            break
        # elif space_group in [output_dict[key].lower().replace(" ", "") for key in str_keys]:
        #     if return_sym_operations:
        #         output_dict["symmetry_operations"] = _get_symmetry_from_database(hall_nr)
        #     break
    return output_dict


def _space_group_analysis(
    structure,
    symprec,
    angle_tolerance,
    hall_number,
    no_idealize=False,
    return_primitive_structure=False,
    return_standardized_structure=False,
    return_sym_operations=False,
):
    """Analyze the symmetry and space group using spglib (new enhanced interface)."""
    cell = _transfrom_structure_to_cell(structure)
    hall_number, output_dict = _get_symmetry_dataset(
        cell,
        symprec,
        angle_tolerance,
        hall_number,
        return_primitive_structure,
        return_sym_operations,
    )
    output_dict["space_group"] = _get_space_group_type(hall_number)
    output_dict["space_group"]["centrosymmetric"] = (
        output_dict["space_group"]["pointgroup_international"] in CENTROSYMMETRIC_PG
    )

    if return_primitive_structure:
        prim_cell = spglib.standardize_cell(
            cell,
            to_primitive=True,
            no_idealize=no_idealize,
            symprec=symprec,
            angle_tolerance=angle_tolerance,
        )
        output_dict["primitive_structure"] = _transform_cell_to_structure(prim_cell)
    if return_standardized_structure:
        std_cell = spglib.standardize_cell(
            cell,
            to_primitive=False,
            no_idealize=no_idealize,
            symprec=symprec,
            angle_tolerance=angle_tolerance,
        )
        output_dict["standardized_structure"] = _transform_cell_to_structure(std_cell)
    return output_dict


def _layer_group_analysis(
    structure,
    symprec=1e-5,
    aperiodic_dir=2,
    return_primitive_structure=False,
    return_standardized_structure=False,
    reorientate_prim_cell=False,
):
    cell = _transfrom_structure_to_cell(structure)
    spglib_sym = spglib.spglib.get_symmetry_layerdataset(
        cell, aperiodic_dir=aperiodic_dir, symprec=symprec
    )
    output_dict = {
        "layer_group": {
            "int_symbol": spglib_sym["international"],
            "lg_number": spglib_sym["number"],
            "hall_number": spglib_sym["hall_number"],
            "point_group_symbol": spglib_sym["pointgroup"],
        }
    }
    if return_standardized_structure:
        output_dict["standardized_structure"] = _transform_cell_to_structure(
            (spglib_sym["std_lattice"], spglib_sym["std_positions"], spglib["std_types"])
        )

    if return_primitive_structure:
        if "positions_cart_np" in structure:
            inp_pos_cart = np.array(structure["positions_cart_np"])
        elif "positions" in structure:
            if structure["is_cartesian"]:
                inp_pos_cart = np.array(structure["positions"])
            else:
                inp_pos_cart = np.array(
                    [np.transpose(cell).dot(np.array(pos)) for pos in structure["positions"]]
                )
        else:
            raise ValueError("Cartesian positions could not be retrieved.")

        prim_cell = spglib_sym["primitive_lattice"]
        reor_matrix = np.identity(3)
        periodic_directions = [idx0 for idx0 in range(3) if idx0 != aperiodic_dir]
        if reorientate_prim_cell:
            # Use aperiodic cell vector from original cell:
            prim_cell[aperiodic_dir] = cell[0][aperiodic_dir].copy()

            # Rotate first periodic vector to the original cell vector direction:
            aperiodic_v = cell[0][aperiodic_dir].copy()
            aperiodic_v /= np.linalg.norm(aperiodic_v)
            angle = calc_angle(prim_cell[0], cell[0][periodic_directions[0]])
            rot = Rotation.from_rotvec(angle * aperiodic_v)
            reor_matrix = rot.as_matrix()
            prim_cell = np.dot(reor_matrix, prim_cell.T).T

            # Mirror second cell vector if the system is right-handed:
            if np.linalg.det(prim_cell) < 0.0:
                refl_matrix = calc_reflection_matrix(
                    np.cross(prim_cell[periodic_directions[0]], prim_cell[aperiodic_dir])
                )
                prim_cell = np.dot(refl_matrix, prim_cell.T).T
                reor_matrix = np.dot(refl_matrix, reor_matrix)
            prim_cell[np.abs(prim_cell) < 1e-6] = 0.0

        used_sites = []
        prim_positions = []
        prim_elements = []
        inv_prim_cell = np.linalg.inv(prim_cell)
        for el, pos, mapping in zip(
            structure["elements"], inp_pos_cart, spglib_sym["mapping_to_primitive"]
        ):
            if mapping in used_sites:
                continue
            prim_pos = np.transpose(inv_prim_cell).dot((reor_matrix.dot(pos.T)).T)
            for direction in periodic_directions:
                while prim_pos[direction] < 0.0:
                    prim_pos[direction] += 1.0
                while prim_pos[direction] >= 1.0:
                    prim_pos[direction] -= 1.0
            prim_pos[np.abs(prim_pos) < 1e-6] = 0.0
            prim_elements.append(el)
            prim_positions.append(prim_pos)
            used_sites.append(mapping)
        output_dict["primitive_structure"] = {
            "cell": prim_cell,
            "positions": prim_positions,
            "elements": prim_elements,
            "is_cartesian": False,
            "pbc": structure["pbc"],
        }
    return output_dict


def _get_symmetry_from_database(hall_number):
    spglib_output = _transform_spglib_output_to_dict(
        spglib.get_symmetry_from_database(hall_number),
        ["rotations", "translations"],
    )
    return [
        [rot, trans]
        for rot, trans in zip(
            spglib_output["rotations"].tolist(), spglib_output["translations"].tolist()
        )
    ]


def _get_symmetry_dataset(
    cell, symprec, angle_tolerance, hall_number, return_primitive_structure, return_sym_operations
):
    spglib_output = _transform_spglib_output_to_dict(
        spglib.get_symmetry_dataset(
            cell, symprec=symprec, angle_tolerance=angle_tolerance, hall_number=hall_number
        ),
        [
            "hall_number",
            "equivalent_atoms",
            "wyckoffs",
            "site_symmetry_symbols",
            "mapping_to_primitive",
            "transformation_matrix",
            "origin_shift",
            "rotations",
            "translations",
        ],
    )

    # Get equivalent sites:
    eq_sites = list(set(spglib_output["equivalent_atoms"]))
    output_dict = {
        "equivalent_sites": [
            {
                "wyckoff": spglib_output["wyckoffs"][site_idx],
                "symmetry": spglib_output["site_symmetry_symbols"][site_idx],
                "sites": [],
            }
            for site_idx in eq_sites
        ]
    }
    for at_idx, eq_at in enumerate(spglib_output["equivalent_atoms"]):
        site_idx = eq_sites.index(eq_at)
        output_dict["equivalent_sites"][site_idx]["sites"].append(at_idx)

    # Get mapping if primitive structure is demanded:
    if return_primitive_structure:
        output_dict["mapping_to_primitive"] = spglib_output["mapping_to_primitive"].tolist()

    # Get symmetry operations:
    if return_sym_operations:
        output_dict["transformation_matrix"] = spglib_output["transformation_matrix"].tolist()
        output_dict["origin_shift"] = spglib_output["origin_shift"].tolist()
        output_dict["symmetry_operations"] = [
            [rot, trans]
            for rot, trans in zip(
                spglib_output["rotations"].tolist(), spglib_output["translations"].tolist()
            )
        ]

    return spglib_output["hall_number"], output_dict


def _get_space_group_type(hall_number):
    return _transform_spglib_output_to_dict(
        spglib.get_spacegroup_type(hall_number),
        [
            "number",
            "international_short",
            "international_full",
            "international",
            "schoenflies",
            "hall_number",
            "hall_symbol",
            "choice",
            "pointgroup_international",
            "pointgroup_schoenflies",
            "arithmetic_crystal_class_number",
            "arithmetic_crystal_class_symbol",
        ],
    )
