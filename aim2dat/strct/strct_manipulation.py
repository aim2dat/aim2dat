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
    has_del = False
    for idx, (el, kind, pos) in enumerate(structure.iter_sites(get_kind=True, get_cart_pos=True)):
        if el not in elements and idx not in site_indices:
            new_structure["elements"].append(el)
            new_structure["kinds"].append(kind)
            new_structure["positions"].append(pos)
        else:
            has_del = True
    if has_del:
        if all(kind is None for kind in new_structure["kinds"]):
            del new_structure["kinds"]
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
        if new_structure["kinds"] is not None:
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
                    if remove_kind and new_structure["kinds"] is not None:
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


def scale_unit_cell(structure: Structure, scaling_factor: float, change_label: bool) -> Structure:
    """Scale unit cell of the structure."""
    if scaling_factor != 1.0 and structure["cell"] is not None:
        new_structure = structure.to_dict(cartesian=False)
        new_structure["cell"] = np.array(structure["cell"]) * scaling_factor
        return _add_label_suffix(new_structure, f"_scaled-{scaling_factor}", change_label)
