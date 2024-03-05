"""Wrapper functions for spglib."""

# Third party library imports
import numpy as np
import spglib
from scipy.spatial.transform import Rotation

# Internal library imports
from aim2dat.utils.element_properties import get_atomic_number, get_element_symbol
from aim2dat.utils.maths import calc_angle, calc_reflection_matrix
from aim2dat.strct.strct import Structure


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
    # if isinstance(structure, (Structure, dict)):
    #     if "cell" in structure:
    #         cell = np.array(structure["cell"]).reshape((3, 3))
    #     else:
    #         raise ValueError("Unit cell could not be retrieved.")
    #
    #     if "cell_inverse" in structure:
    #         inv_cell = structure["cell_inverse"]
    #     else:
    #         inv_cell = np.linalg.inv(cell)
    #     if "positions" in structure:
    #         if structure["is_cartesian"]:
    #             positions = [
    #                 np.transpose(inv_cell).dot(np.array(pos)) for pos in structure["positions"]
    #             ]
    #         else:
    #             positions = structure["positions"]
    #     elif "scaled_positions" in structure:
    #         positions = structure["scaled_positions"]
    #     else:
    #         raise ValueError("Scaled positions could not be retrieved.")
    #     cell = (
    #         cell,
    #         positions,
    #         [get_atomic_number(el) for el in structure["elements"]],
    #     )
    # elif isinstance(structure, (list, tuple)):
    #     cell = structure
    # else:
    #     # To-Do: make independent of ase:
    #     ase_structure = structure.get_ase()
    #     cell = (
    #         ase_structure.cell[:],
    #         ase_structure.get_scaled_positions(),
    #         ase_structure.get_atomic_numbers(),
    #     )
    return cell


def _transform_cell_to_structure(cell):
    structure = {
        "cell": cell[0],
        "positions": cell[1],
        "elements": [get_element_symbol(el) for el in cell[2]],
        "is_cartesian": False,
        "pbc": True,
    }
    return Structure(**structure)


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
    spglib_sym = spglib.get_symmetry_dataset(
        cell, symprec=symprec, angle_tolerance=angle_tolerance, hall_number=hall_number
    )
    spglib_type = spglib.get_spacegroup_type(spglib_sym["hall_number"])
    output_dict = {
        "space_group": {
            "int_symbol": spglib_sym["international"],
            "sg_number": spglib_sym["number"],
            "hall_number": spglib_sym["hall_number"],
            "point_group_symbol": spglib_sym["pointgroup"],
            "schoenflies_symbol": spglib_type["schoenflies"],
            "centrosymmetric": spglib_sym["pointgroup"] in CENTROSYMMETRIC_PG,
        }
    }
    eq_sites = list(set(spglib_sym["equivalent_atoms"]))
    output_dict["equivalent_sites"] = [
        {
            "wyckoff": spglib_sym["wyckoffs"][site_idx],
            "symmetry": spglib_sym["site_symmetry_symbols"][site_idx],
            "sites": [],
        }
        for site_idx in eq_sites
    ]
    for at_idx, eq_at in enumerate(spglib_sym["equivalent_atoms"]):
        site_idx = eq_sites.index(eq_at)
        output_dict["equivalent_sites"][site_idx]["sites"].append(at_idx)

    if return_primitive_structure:
        output_dict["mapping_to_primitive"] = spglib_sym["mapping_to_primitive"]
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
    if return_sym_operations:
        output_dict["transformation_matrix"] = spglib_sym["transformation_matrix"].tolist()
        output_dict["origin_shift"] = spglib_sym["origin_shift"].tolist()
        output_dict["symmetry_operations"] = [
            [rot, trans]
            for rot, trans in zip(
                spglib_sym["rotations"].tolist(), spglib_sym["translations"].tolist()
            )
        ]
    return spglib_sym["number"], output_dict


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
