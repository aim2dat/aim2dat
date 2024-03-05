"""Ase surface interface."""

# Third party library imports
from ase.build import surface

# Internal library imports
from aim2dat.ext_interfaces.ase_atoms import _create_atoms_from_structure


def _create_ase_surface_from_structure(struct_dict, miller_indices, nr_layers, vacuum, periodic):
    return surface(
        _create_atoms_from_structure(struct_dict), miller_indices, nr_layers, vacuum, periodic
    )
