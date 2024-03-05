"""Module that implements functions to change molecular or crystalline structures."""

# Standard library imports
import os

# Third party library imports
import numpy as np
from scipy.spatial.transform import Rotation

# Internal library imports
from aim2dat.strct.ext_manipulation.decorator import (
    external_manipulation_method,
)
from aim2dat.strct.strct import Structure
from aim2dat.utils.element_properties import get_element_symbol
from aim2dat.strct.strct_manipulation import _add_label_suffix
from aim2dat.utils.maths import calc_angle
from aim2dat.io.yaml import load_yaml_file


cwd = os.path.dirname(__file__)


@external_manipulation_method
def add_functional_group(
    structure: Structure,
    wrap: bool = False,
    host_index: int = 0,
    functional_group: str = "CH3",
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
    change_label: bool = False,
) -> Structure:
    """
    Add a functional group or an atom to a host site.

    Parameters
    ----------
    key : str, int, tuple or list
        Only used in the ``StructureOperations`` class. Specifies the key or list/tuple of
        keys of the underlying ``StructureCollection`` object.
    wrap : bool (optional)
            Wrap atomic positions back into the unit cell.
    host_index : int
        Index of the host site.
    functional_group : str
        Functional group or element symbol of the added functional group or atom.
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


    Returns
    -------
    aiida_scripst.strct.Structure
        Structure with attached functional group.
    """
    if host_index < len(structure):
        fct_group_dict = _check_functional_group(functional_group)

        # Calculate coordination:
        coord = structure.calculate_coordination(
            method=cn_method,
            min_dist_delta=min_dist_delta,
            n_nearest_neighbours=n_nearest_neighbours,
            econ_tolerance=econ_tolerance,
            econ_conv_threshold=econ_conv_threshold,
            okeeffe_weight_threshold=okeeffe_weight_threshold,
        )
        cn_details = coord["sites"][host_index]

        # Derive bond direction and first rotation matrix:
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
        rot_dir = np.cross(bond_dir, np.array([1.0, 0.0, 0.0]))
        rot_dir /= np.linalg.norm(rot_dir)
        rot_angle = -calc_angle(np.array([1.0, 0.0, 0.0]), bond_dir)
        rotation = Rotation.from_rotvec(rot_angle * rot_dir)
        rot_matrix = rotation.as_matrix()

        # Create updated structure:
        new_structure = {
            "label": structure["label"],
            "pbc": structure["pbc"],
            "is_cartesian": True,
            "wrap": wrap,
            "positions": list(structure["positions"]),
            "cell": structure["cell"],
            "elements": list(structure["elements"]),
        }
        for el, pos in zip(fct_group_dict["elements"], fct_group_dict["positions"]):
            pos = rot_matrix.dot(np.array(pos).T)
            new_structure["elements"].append(el)
            new_structure["positions"].append(pos + bond_dir * bond_length + host_pos_np)
        return _add_label_suffix(new_structure, "_added-" + functional_group, change_label)


def _check_functional_group(fct_group_str: str) -> dict:
    if not isinstance(fct_group_str, str):
        raise TypeError("Functional group needs to be of type str.")
    fct_group_dict = {}
    try:
        fct_group_dict["elements"] = [get_element_symbol(fct_group_str)]
        fct_group_dict["positions"] = [[0.0, 0.0, 0.0]]
    except ValueError:
        try:
            fct_group_dict = load_yaml_file(
                os.path.join(cwd, "functional_groups", fct_group_str + ".yaml")
            )
        except FileNotFoundError:
            raise ValueError(f"Functional group `{fct_group_str}` is not supported.")
    return fct_group_dict
