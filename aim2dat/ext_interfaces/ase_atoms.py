"""Wrapper functions for the ase Atoms object."""

# Standard library imports
import re

# Third party library imports
import numpy as np
from ase import Atoms
from ase.io import read, write


def _extract_structure_from_atoms(atoms):
    """Extract a dictionary with structural parameters from the ase Atoms object."""
    keys2neglect = ["numbers", "positions", "tags"]

    positions = []
    elements = []
    kinds = []
    tags_sum = 0
    for atom in atoms:
        elements.append(atom.symbol)
        positions.append([float(atom.position[idx]) for idx in range(3)])
        kinds.append(f"{atom.symbol}{atom.tag}")
        tags_sum += atom.tag
    structure_dict = {
        "elements": elements,
        "kinds": kinds if tags_sum != 0 else None,
        "positions": positions,
        "pbc": atoms.get_pbc().tolist(),
        "is_cartesian": True,
        "attributes": atoms.info,
        "site_attributes": {},
    }
    for key, val in atoms.arrays.items():
        if len(val) == len(structure_dict["elements"]) and key not in keys2neglect:
            structure_dict["site_attributes"][key] = val.tolist()
    if any(structure_dict["pbc"]):
        structure_dict["cell"] = [cell_v.tolist() for cell_v in atoms.cell.array]
    return structure_dict


def _create_atoms_from_structure(structure):
    """Create ase atoms object from structure dictionary."""
    tags = []
    for k in structure.kinds:
        tag = None if k is None else re.findall(r"\d+", k)
        tag = int(tag[0]) if tag else 0
        tags.append(tag)
    atoms = Atoms(
        structure.elements,
        positions=structure.positions,
        cell=structure.cell,
        pbc=structure.pbc,
        tags=tags,
        info=structure.attributes,
    )
    for key, val in structure.site_attributes.items():
        atoms.set_array(key, np.array(val))
    return atoms


def _load_structure_from_file(file_path, kwargs):
    """
    Load structure from file using the ase implementation.
    """
    atoms = read(file_path, **kwargs)
    if isinstance(atoms, Atoms):
        atoms = [atoms]
    return [_extract_structure_from_atoms(at) for at in atoms]


def _write_structure_to_file(file_path, structure):
    """Write structure to file using the ase implementation."""
    if isinstance(structure, list):
        atoms = [_create_atoms_from_structure(strct) for strct in structure]
    else:
        atoms = _create_atoms_from_structure(structure)
    write(file_path, atoms)
