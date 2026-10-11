"""Methods to add one or several atoms."""

# Third party library imports
import numpy as np
from scipy.spatial.transform import Rotation

# Internal library imports
from aim2dat.strct.structure import Structure
from aim2dat.strct.validation import _structure_validate_elements
from aim2dat.strct.ext_manipulation.decorator import external_manipulation_method
from aim2dat.strct.ext_manipulation.add_structure import add_structure_position
from aim2dat.chem_f import transform_list_to_str
from aim2dat.utils.maths import calc_angle


@external_manipulation_method
def add_atom_prototype(
    structure: Structure,
    indices: list,
    elements: list = ["H"],
    bond_lengths: list = [1.0],
    change_label: bool = False,
    **cn_kwargs,
):
    """
    Add atom(s) based on a tetrahdedral, planar or linear prototype.

    Parameters
    ----------
    structure : aim2dat.strct.Structure
        Structure to which the atom(s) is(are) added.
    indices : list
        Indices of the sites that the atoms are attached to.
    elements : list
        List of elements to add.
    bond_lengths : list
        Distance(s) to the site that the atoms are added.
    change_label : bool
        Add suffix to the label of the new structure highlighting the performed manipulation.
    cn_kwargs :
        Optional keyword arguments passed on to the ``calc_coordination`` function.

    Raises
    ------
    ValueError / TypeError
        Elments are given in the wrong format.
    ValueError
        Number of given bond lengths and elements differ.
    ValueError
        If no prototype is found for one of the sites.
    ValueError
        If adding the atoms fails.
    """
    elements = _structure_validate_elements(elements)
    if len(elements) != len(bond_lengths):
        raise ValueError("`elements` and `bond_lengths` must have the same length.")

    tetrahedral_dirs = np.array(
        [
            [1, -1, 1],
            [1, 1, -1],
            [-1, 1, 1],
            [-1, -1, -1],
        ]
    ) / np.sqrt(3)
    planar_dirs = np.array(
        [
            [1.0, 0.0, 0.0],
            [np.cos(2.0 * np.pi / 3.0), np.sin(2.0 * np.pi / 3.0), 0.0],
            [np.cos(4.0 * np.pi / 3.0), np.sin(4.0 * np.pi / 3.0), 0.0],
        ]
    )
    linear_dirs = np.array(
        [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
        ]
    )

    if isinstance(indices, int):
        indices = [indices]
    all_coords = structure.calc_coordination(indices=indices, **cn_kwargs)
    for site_idx, coord in zip(indices, all_coords["sites"]):
        positions = [
            np.array(neigh["position"]) - np.array(coord["position"])
            for neigh in coord["neighbours"]
        ]
        positions /= np.linalg.norm(positions, axis=1).reshape(
            -1, 1
        )  # [pos / np.linalg.norm(pos) for pos in positions]

        if coord["total_cn"] + len(elements) == 4:
            prot = tetrahedral_dirs
        elif coord["total_cn"] + len(elements) == 3:
            prot = planar_dirs
        elif coord["total_cn"] + len(elements) == 2:
            prot = linear_dirs
        else:
            raise ValueError(
                f"No prototype found for site {site_idx} with coord. {coord['total_cn']}."
            )

        if len(positions) == 0:
            rot = Rotation.from_rotvec([0.0, 0.0, 0.0], degrees=False)
        elif len(positions) == 1:
            angle = -1.0 * calc_angle(positions[0], prot[0])
            rot_dir = np.cross(positions[0], prot[0])
            rot_dir /= np.linalg.norm(rot_dir)
            rot = Rotation.from_rotvec(angle * rot_dir, degrees=False)
        else:
            rot, _ = Rotation.align_vectors(positions[:2], prot[:2], return_sensitivity=False)

        new_pos = rot.apply(prot)
        final_positions = []
        for pos in new_pos:
            dists = np.linalg.norm(positions - np.tile(pos, (len(positions), 1)), axis=1)
            if any(dists < 0.5):
                continue
            final_positions.append(pos)
        if len(final_positions) != len(elements):
            raise ValueError(f"Falied to add atoms at site {site_idx}.")

        structure_add = Structure(
            elements=elements,
            positions=[
                d * bl + np.array(coord["position"])
                for d, bl in zip(final_positions, bond_lengths)
            ],
            pbc=False,
        )
        structure = add_structure_position(
            structure, guest_structure=structure_add, position=structure_add.positions, wrap=True
        )
    return structure, "_added-" + transform_list_to_str(elements)
