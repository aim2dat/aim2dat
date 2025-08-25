"""Interface to h5py to store/import structures."""

# Standard library imports
from typing import TYPE_CHECKING, List

# Third party library imports
import numpy as np
import h5py

if TYPE_CHECKING:
    from aim2dat.strct.structure import Structure


def _store_in_hdf5_file(file_path: str, structures: List["Structure"]) -> None:
    """Store structures in file."""
    string_dt = h5py.string_dtype(encoding="utf-8")
    labels = []
    nr_atoms = []
    pbc = []
    cells = []
    elements = []
    kinds = []
    positions = []
    attributes = {}
    site_attributes = {}

    same_elements = False
    same_kinds = False
    same_pbc = False
    if all(structures[0].elements == strct.elements for strct in structures):
        same_elements = True
        if all(structures[0].kinds == strct.kinds for strct in structures):
            same_kinds = True
    if all(structures[0].pbc == strct.pbc for strct in structures):
        same_pbc = True

    for idx, structure in enumerate(structures):
        labels.append(structure.label)
        if not same_elements or idx == 0:
            nr_atoms.append(len(structure))
            elements += structure.numbers
        if not same_kinds or idx == 0:
            kinds += ["" if k is None else k for k in structure.kinds]
        if not same_pbc or idx == 0:
            pbc += structure.pbc
        if structure.cell is not None:
            cells += [v1 for dir0 in structure.cell for v1 in dir0]
        for pos in structure.positions:
            positions += list(pos)
        if len(structure.attributes) > 0:
            attributes[structure.label] = structure.attributes
        if len(structure.site_attributes) > 0:
            site_attributes[structure.label] = structure.site_attributes

    with h5py.File(file_path, "w") as fobj:
        fobj.create_dataset("labels", data=labels, dtype=string_dt)
        fobj.create_dataset("nr_atoms", data=nr_atoms, dtype="int16")
        fobj.create_dataset("elements", data=elements, dtype="int8")
        fobj.create_dataset("pbc", data=pbc, dtype="bool")
        fobj.create_dataset("cells", data=cells, dtype="float32")
        fobj.create_dataset("kinds", data=kinds, dtype=string_dt)
        fobj.create_dataset("positions", data=positions, dtype="float32")
        if len(attributes) > 0:
            _add_recursive_dict(fobj.create_group("attributes"), attributes)
        if len(site_attributes) > 0:
            _add_recursive_dict(fobj.create_group("site_attributes"), site_attributes)


def _import_from_hdf5_file(file_path: str) -> List[dict]:
    """Read structures from file."""
    with h5py.File(file_path, "r") as fobj:
        labels = fobj["labels"][:]
        all_nr_atoms = fobj["nr_atoms"][:]
        pbc = fobj["pbc"][:]
        cells = fobj["cells"][:]
        elements = fobj["elements"][:]
        if "kinds" in fobj:
            kinds = fobj["kinds"][:]
        else:
            kinds = fobj["elements"][:]
        positions = fobj["positions"][:]
        attributes = {"attributes": {}}
        if "attributes" in fobj.keys():
            _retrieve_recursive_dict(fobj, "attributes", attributes)
        site_attributes = {"site_attributes": {}}
        if "site_attributes" in fobj.keys():
            _retrieve_recursive_dict(fobj, "site_attributes", site_attributes)

    attributes = attributes["attributes"]
    site_attributes = site_attributes["site_attributes"]
    if len(all_nr_atoms) != len(labels):
        all_nr_atoms = np.tile(all_nr_atoms, len(labels))
        elements = np.tile(elements, len(labels))
    if len(kinds) != len(elements):
        kinds = np.tile(kinds, len(labels))
    if len(pbc) != len(labels) * 3:
        pbc = np.tile(pbc, len(labels))

    c_atoms = 0
    c_pbc = 0
    structures = []
    for strct_idx, (nr_atoms, label) in enumerate(zip(all_nr_atoms, labels)):
        label = label.decode()
        structure = {
            "label": label,
            "elements": [
                el.decode() if hasattr(el, "decode") else el
                for el in elements[c_atoms : c_atoms + nr_atoms]
            ],
            "positions": [
                [positions[idx], positions[idx + 1], positions[idx + 2]]
                for idx in range(c_atoms * 3, (c_atoms + nr_atoms) * 3, 3)
            ],
            "pbc": [pbc[strct_idx * 3], pbc[strct_idx * 3 + 1], pbc[strct_idx * 3 + 2]],
            "is_cartesian": True,
        }
        kinds0 = [kind.decode() for kind in kinds[c_atoms : c_atoms + nr_atoms]]
        if any(kind != "" for kind in kinds0):
            structure["kinds"] = kinds0
        if any(structure["pbc"]):
            structure["cell"] = [
                [cells[c_pbc], cells[c_pbc + 1], cells[c_pbc + 2]],
                [cells[c_pbc + 3], cells[c_pbc + 4], cells[c_pbc + 5]],
                [cells[c_pbc + 6], cells[c_pbc + 7], cells[c_pbc + 8]],
            ]
            c_pbc += 9
        if label in attributes and len(attributes[label]) > 0:
            structure["attributes"] = attributes[label]
        if label in site_attributes and len(site_attributes[label]) > 0:
            structure["site_attributes"] = site_attributes[label]
        structures.append(structure)
        c_atoms += nr_atoms
    return structures


def _add_recursive_dict(grp: h5py.Group, input_dict: dict) -> None:
    for key, value in input_dict.items():
        if isinstance(value, dict):
            new_grp = grp.create_group(key)
            _add_recursive_dict(new_grp, value)
        elif value is not None:
            grp[key] = value


def _retrieve_recursive_dict(grp: h5py.Group, key: str, output_dict: dict) -> None:
    value = grp[key]
    if isinstance(value, h5py.Group):
        output_dict[key] = {}
        for sub_key in value.keys():
            _retrieve_recursive_dict(value, sub_key, output_dict[key])
    else:
        if value.dtype == "object":
            value = value.asstr()[()]
        else:
            value = value[()]
        output_dict[key] = getattr(value, "tolist", lambda: value)()
