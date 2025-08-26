"""Internal manipulation methods dealing with specific atomic sites."""

# Standard library imports
from typing import Tuple, List, TYPE_CHECKING, Union

# Internal library imports
from aim2dat.elements import get_element_symbol, get_atomic_radius
from aim2dat.strct.manipulation.utils import _add_label_suffix


if TYPE_CHECKING:
    from aim2dat.strct.structure import Structure


def delete_atoms(
    structure: "Structure",
    elements: Union[str, List[str]],
    site_indices: Union[int, List[int]],
    change_label: bool,
) -> Union["Structure", dict]:
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
    structure: "Structure",
    elements: List[Tuple[Union[str, int]]],
    radius_type: Union[str, None],
    remove_kind: bool,
    change_label: bool,
) -> Union["Structure", dict]:
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
            new_structure["cell"] = [
                [value * scaling_factor if i == dir_idx else value for i, value in enumerate(row)]
                for dir_idx, row in enumerate(structure["cell"])
            ]
        return _add_label_suffix(new_structure, "_subst-" + "-".join(str_el_pairs), change_label)
