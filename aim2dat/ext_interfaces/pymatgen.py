"""Wrapper functions for the pymatgen interface."""

# Third party library imports
from pymatgen.core import Lattice, Structure, Molecule


def _create_pymatgen_obj(structure):
    """Create pymatgen Structure/Molecule object from Structure object."""
    if structure.cell is None:
        return Molecule(
            species=structure.elements,
            coords=structure.positions,
            labels=structure.kinds,
        )
    else:
        return Structure(
            lattice=Lattice(structure.cell, structure.pbc),
            species=structure.elements,
            coords=structure.positions,
            labels=structure.kinds,
            coords_are_cartesian=True,
        )


def _extract_structure_from_pymatgen(pymatgen_obj):
    """Extract structure from pymatgen Structure/Molecule object."""
    strct_dict = {"kinds": [], "elements": [], "positions": [], "pbc": False}
    has_kinds = False
    for site in pymatgen_obj.sites:
        if len(site.species) > 1:
            raise ValueError("Partial occupations of sites is not supported.")
        element = site.species.elements[0].name
        strct_dict["elements"].append(element)
        strct_dict["positions"].append(site.coords)
        strct_dict["kinds"].append(site.label)
        if site.label != element:
            has_kinds = True
    if not has_kinds:
        del strct_dict["kinds"]
    if hasattr(pymatgen_obj, "lattice"):
        strct_dict["cell"] = pymatgen_obj.lattice.matrix
        strct_dict["pbc"] = pymatgen_obj.lattice.pbc
    return strct_dict
