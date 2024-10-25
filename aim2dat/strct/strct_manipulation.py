"""Module that implements functions to change molecular or crystalline structures."""

# Standard library imports
from __future__ import annotations
import os
from typing import List, Tuple, Union, TYPE_CHECKING

# Third party library imports
import numpy as np

# Internal library imports
from aim2dat.utils.element_properties import get_atomic_radius, get_element_symbol

if TYPE_CHECKING:
    from aim2dat.strct.strct import Structure


cwd = os.path.dirname(__file__)


def _add_label_suffix(strct, label_suffix, change_label):
    if change_label:
        new_label = label_suffix if strct["label"] is None else strct["label"] + label_suffix
        if isinstance(strct, dict):
            strct["label"] = new_label
        else:
            strct.label = new_label
    return strct


def delete_atoms(
    structure: Structure,
    elements: Union[str, List[str]],
    site_indices: Union[int, List[int]],
    change_label: bool,
) -> Structure:
    """Delete atoms."""
    # Check elements:
    if not isinstance(elements, (list, tuple)):
        elements = [elements]
    elements = [get_element_symbol(el) for el in elements]
    # TODO handle slices?
    if not isinstance(site_indices, (list, tuple)):
        site_indices = [site_indices]
    if not all(isinstance(site_idx, int) for site_idx in site_indices):
        raise TypeError("All site indices need to be of type int.")

    # Create new structure dict:
    new_structure = structure.to_dict()
    new_structure["positions"] = []
    new_structure["elements"] = []
    new_structure["kinds"] = []
    site_attributes = list(new_structure["site_attributes"].keys())
    new_structure["site_attributes"] = {key: [] for key in site_attributes}
    has_del = False
    for idx, site_details in enumerate(
        structure.iter_sites(get_kind=True, get_cart_pos=True, site_attributes=site_attributes)
    ):
        el = site_details[0]
        kind = site_details[1]
        pos = site_details[2]
        site_attr_vals = site_details[3:]
        if el not in elements and idx not in site_indices:
            new_structure["elements"].append(el)
            new_structure["kinds"].append(kind)
            new_structure["positions"].append(pos)
            for key, val in zip(site_attributes, site_attr_vals):
                new_structure["site_attributes"][key].append(val)
        else:
            has_del = True
    if has_del:
        return _add_label_suffix(new_structure, "_del", change_label)


def substitute_elements(
    structure: Structure,
    elements: List[Tuple[Union[str, int]]],
    radius_type: Union[str, None],
    remove_kind: bool,
    change_label: bool,
) -> Tuple[Structure, str]:
    """Substitute all atoms of the same element by another element."""
    if isinstance(elements[0], (str, int)):
        elements = [elements]
    attributes2keep = ["space_group", "source"]
    str_el_pairs = None
    if any(el_pair[0] in structure["elements"] for el_pair in elements):
        new_structure = structure.to_dict(cartesian=False)
        new_structure["elements"] = list(new_structure["elements"])
        new_structure["kinds"] = list(new_structure["kinds"])
        for label, val in structure["attributes"].items():
            if label in attributes2keep:
                new_structure["attributes"][label] = val
        str_el_pairs = []

        scaling_factor = 0.0
        nr_sub_atoms = 0
        for el_pair in elements:
            if el_pair[0] in structure._element_dict:
                str_el_pairs.append(el_pair[0] + el_pair[1])
                site_indices = structure._element_dict[el_pair[0]]
                for site_idx in site_indices:
                    new_structure["elements"][site_idx] = el_pair[1]
                    if remove_kind:
                        new_structure["kinds"][site_idx] = None
                if radius_type is not None:
                    scaling_factor += (
                        get_atomic_radius(el_pair[1], radius_type=radius_type)
                        / get_atomic_radius(el_pair[0], radius_type=radius_type)
                        * len(site_indices)
                    )
                    nr_sub_atoms += len(site_indices)
        scaling_factor = (scaling_factor + len(new_structure["elements"]) - nr_sub_atoms) / len(
            new_structure["elements"]
        )
        if structure["cell"] is not None:
            new_structure["cell"] = np.array(structure["cell"])
            for dir_idx in range(3):
                new_structure["cell"][dir_idx] *= scaling_factor
        return _add_label_suffix(new_structure, "_subst-" + "-".join(str_el_pairs), change_label)


def scale_unit_cell(
    structure: Structure, 
    scaling_factors: Union[float, List[float], np.ndarray], 
    change_label: bool = True
) -> Structure:
    """
    Scale the unit cell of the structure. Supports anisotropic scaling when `scaling_factors` is an array.

    Parameters:
        structure (Structure): The structure to be scaled.
        scaling_factors (float, list, or array): Scaling factor(s) for the unit cell.
        change_label (bool): Whether to change the label of the structure.

    Returns:
        Structure: The scaled structure.
    """
    # Ensure scaling_factors is an array for consistent handling.
    scaling_factors = np.array(scaling_factors)

    # Check if any scaling is needed.
    if not np.all(scaling_factors == 1.0) and structure["cell"] is not None:
        new_structure = structure.to_dict(cartesian=False)
        new_structure["cell"] = np.array(structure["cell"]) * scaling_factors
        return _add_label_suffix(new_structure, f"_scaled-aniso-{scaling_factors}", change_label)

    # If no scaling is needed, return the original structure.
    return structure

def apply_pressure_or_strain(
    structure,
    pressure: float = None,
    bulk_modulus: float = None,
    strain: Union[float, np.ndarray, list] = None,
    change_label: bool = True
) -> Structure:
    """
    Adjust the structure by applying pressure or strain using a strain tensor.

    Parameters:
        structure (Structure): The structure to modify.
        pressure (float, optional): Hydrostatic pressure to apply in GPa.
        bulk_modulus (float, optional): Bulk modulus in GPa, required if `pressure` is provided.
        strain (float, list of 3 floats, or 3x3 matrix, optional): Strain to apply. Can be uniform (float), anisotropic (list of 3 values), or a 3x3 strain matrix.
        change_label (bool): Whether to change the label of the structure.

    Returns:
        Structure: The modified structure with adjusted lattice.
    """
    from aim2dat.strct import Structure

    def get_strain_tensor(strain):
        """Convert strain input into a 3x3 strain tensor."""
        if isinstance(strain, (float, int)):
            return np.eye(3) * (1 + strain)
        if isinstance(strain, list) and len(strain) == 3:
            return np.diag([1 + s for s in strain])
        if isinstance(strain, np.ndarray) and strain.shape == (3, 3):
            return np.eye(3) + strain
        raise ValueError("Strain must be a float (uniform), a list of 3 values, or a 3x3 matrix.")

    # Determine strain from pressure if specified.
    if pressure is not None:
        if bulk_modulus is None:
            raise ValueError("Bulk modulus must be provided when applying pressure.")
        strain = -pressure / bulk_modulus

    if strain is None:
        raise ValueError("Either `pressure` or `strain` must be specified.")
    
    # Generate the strain tensor and apply it to the structure's lattice.
    scaling_factors = get_strain_tensor(strain)
    scaled_cell = np.dot(structure["cell"], scaling_factors)

    # Update the structure with the new scaled cell.
    new_structure = structure.to_dict(cartesian=False)
    new_structure["cell"] = scaled_cell

    return Structure(**_add_label_suffix(new_structure, "_scaled-strain", change_label))