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
        if strct["label"] is None:
            strct["label"] = ""
        strct["label"] += label_suffix
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
    new_structure = {
        "label": structure["label"],
        "pbc": structure["pbc"],
        "is_cartesian": True,
        "positions": [],
        "cell": structure["cell"],
        "elements": [],
    }
    has_del = False
    for index, el in enumerate(structure["elements"]):
        if el not in elements and index not in site_indices:
            new_structure["elements"].append(el)
            new_structure["positions"].append(structure["positions"][index])
        else:
            has_del = True
    if has_del:
        return _add_label_suffix(new_structure, "_del", change_label)


def substitute_elements(
    structure: Structure,
    elements: List[Tuple[Union[str, int]]],
    radius_type: Union[str, None],
    change_label: bool,
) -> Tuple[Structure, str]:
    """Substitute all atoms of the same element by another element."""
    if isinstance(elements[0], (str, int)):
        elements = [elements]
    attributes2keep = ["space_group", "source"]
    str_el_pairs = None
    if any(el_pair[0] in structure["elements"] for el_pair in elements):
        new_structure = {
            "label": structure.label,
            "elements": list(structure["elements"]),
            "positions": structure["scaled_positions"],
            "is_cartesian": False,
            "pbc": structure["pbc"],
            "attributes": {},
        }
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


def scale_unit_cell(structure: Structure, scaling_factor: float, change_label: bool) -> Structure:
    """Scale unit cell of the structure."""
    if scaling_factor != 1.0:
        new_structure = {
            "label": structure["label"],
            "cell": np.array(structure["cell"]) * scaling_factor,
            "positions": structure["scaled_positions"],
            "elements": structure["elements"],
            "pbc": structure["pbc"],
            "is_cartesian": False,
        }
        return _add_label_suffix(new_structure, f"_scaled-{scaling_factor}", change_label)
