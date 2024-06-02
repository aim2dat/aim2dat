"""Test structure manipulation functions."""

# Standard library imports
import os

# Internal library imports
from aim2dat.strct import Structure
from aim2dat.strct.ext_manipulation import add_structure
from aim2dat.io.yaml import load_yaml_file

STRUCTURES_PATH = os.path.dirname(__file__) + "/structures/"
STRUCTURE_MANIPULATION_PATH = os.path.dirname(__file__) + "/structure_manipulation/"


def test_add_structure_molecules(structure_comparison):
    """Test add_structure method."""
    inputs = dict(load_yaml_file(STRUCTURES_PATH + "NH3.yaml"))
    new_strct = add_structure(
        Structure(**inputs),
        host_indices=[1, 2, 3],
        guest_structure="OH",
        min_dist_delta=0.5,
        bond_length=1.0,
        dist_constraints=[(1, 1, 0.9)],
    )
    ref_p = load_yaml_file(STRUCTURE_MANIPULATION_PATH + "NH3-OH_ref.yaml")
    structure_comparison(new_strct, ref_p)
