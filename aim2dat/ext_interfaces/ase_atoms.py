"""Wrapper functions for the ase Atoms object."""

# Standard library imports
import re

# Third party library imports
from ase import Atoms
from ase.io import read, write


def _extract_structure_from_atoms(atoms):
    """Extract a dictionary with structural parameters from the ase Atoms object."""
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
    }
    if any(structure_dict["pbc"]):
        structure_dict["cell"] = [cell_v.tolist() for cell_v in atoms.cell.array]
    return structure_dict


def _create_atoms_from_structure(structure):
    """Create ase atoms object from structure dictionary."""
    tags = []
    for k in structure.kinds:
        tag = None if k is None else re.findall(r"\d+", k)
        if tag:
            tags.append(int(tag[0]))
        else:
            tags.append(0)

    return Atoms(
        structure.elements,
        positions=structure.positions,
        cell=structure.cell,
        pbc=structure.pbc,
        tags=tags,
    )


def _load_structure_from_file(file_path, kwargs):
    """
    Load structure from file using the ase implementation.

    As for cif-files a tempory
    """
    return [_extract_structure_from_atoms(read(file_path, **kwargs))]


def _write_structure_to_file(struct_dict, file_path):
    """Write structure to file using the ase implementation."""
    write(file_path, _create_atoms_from_structure(struct_dict))
