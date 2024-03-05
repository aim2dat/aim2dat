"""Interface to h5py to store/import structures."""

# Standard library imports
from typing import List

# Third party library imports
import h5py

# Internal library imports
from aim2dat.strct.strct import Structure


def _store_in_hdf5_file(file_path: str, structures: List[Structure]) -> None:
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

    for structure in structures:
        labels.append(structure.label)
        nr_atoms.append(len(structure["elements"]))
        pbc += structure["pbc"]
        if structure["cell"] is not None:
            cells += [v1 for dir0 in structure.cell for v1 in dir0]
        elements += structure["elements"]
        if structure.kinds is not None:
            kinds += structure["kinds"]
        else:
            kinds += [""] * len(structure)
        for pos in structure["positions"]:
            positions += list(pos)
        attributes[structure.label] = structure["attributes"]

    with h5py.File(file_path, "w") as fobj:
        # labels = np.array(labels, dtype="str")
        fobj.create_dataset("labels", data=labels, dtype=string_dt)
        fobj.create_dataset("nr_atoms", data=nr_atoms, dtype="int16")
        fobj.create_dataset("pbc", data=pbc, dtype="bool")
        fobj.create_dataset("cells", data=cells, dtype="float32")
        fobj.create_dataset("elements", data=elements, dtype=string_dt)
        fobj.create_dataset("kinds", data=kinds, dtype=string_dt)
        fobj.create_dataset("positions", data=positions, dtype="float32")
        attr_grp = fobj.create_group("attributes")
        _add_recursive_dict(attr_grp, attributes)


def _import_from_hdf5_file(file_path: str) -> List[Structure]:
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
        attributes = {}
        _retrieve_recursive_dict(fobj, "attributes", attributes)

    attributes = attributes["attributes"]
    c_atoms = 0
    c_pbc = 0
    structures = []
    for strct_idx, (nr_atoms, label) in enumerate(zip(all_nr_atoms, labels)):
        label = label.decode()
        structure = {
            "label": label,
            "elements": [el.decode() for el in elements[c_atoms : c_atoms + nr_atoms]],
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
        structure = Structure(**structure)
        structures.append(structure)
        c_atoms += nr_atoms
    return structures


def _add_recursive_dict(grp: h5py.Group, input_dict: dict) -> None:
    # if isintance(input_dict, dict):
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
