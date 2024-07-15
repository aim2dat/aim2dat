"""Test structure manipulation functions."""

# Standard library imports
import os

# Third party library imports
import pytest

# Internal library imports
from aim2dat.strct import Structure
from aim2dat.strct.ext_manipulation import add_structure_coord, add_structure_random
from aim2dat.io.yaml import load_yaml_file

STRUCTURES_PATH = os.path.dirname(__file__) + "/structures/"
STRUCTURE_MANIPULATION_PATH = os.path.dirname(__file__) + "/structure_manipulation/"


def test_add_structure_coord_molecules(structure_comparison):
    """Test add_structure_coord method."""
    inputs = dict(load_yaml_file(STRUCTURES_PATH + "NH3.yaml"))
    new_strct = add_structure_coord(
        Structure(**inputs),
        host_indices=[1, 2, 3],
        guest_structure="OH",
        min_dist_delta=0.5,
        bond_length=1.0,
        dist_constraints=[(1, 1, 0.9)],
    )
    ref_p = load_yaml_file(STRUCTURE_MANIPULATION_PATH + "NH3-OH_ref.yaml")
    structure_comparison(new_strct, ref_p)


def test_add_structure_random_molecule(structure_comparison):
    """Test add_structure_random method for molecules."""
    inputs = dict(load_yaml_file(STRUCTURES_PATH + "NH3.yaml"))
    new_strct = add_structure_random(
        Structure(**inputs),
        guest_structure="H2O",
        random_state=44,
        dist_threshold=3.4,
    )
    new_strct = add_structure_random(
        new_strct,
        guest_structure="OH",
        random_state=44,
        dist_threshold=1.0,
    )
    ref_p = load_yaml_file(STRUCTURE_MANIPULATION_PATH + "NH3-H2O-OH_ref.yaml")
    structure_comparison(new_strct, ref_p)


def test_add_structure_random_crystal(structure_comparison):
    """Test add_structure_random method for a crystal."""
    inputs = dict(load_yaml_file(STRUCTURES_PATH + "NaCl_225_conv.yaml"))
    new_strct = add_structure_random(
        Structure(**inputs),
        guest_structure="H",
        random_state=44,
        dist_threshold=1.0,
    )
    ref_p = load_yaml_file(STRUCTURE_MANIPULATION_PATH + "NaCl_225_conv-H_ref.yaml")
    structure_comparison(new_strct, ref_p)


def test_add_structure_random_molecules_error():
    """Test add_structure_random method errors."""
    inputs = dict(load_yaml_file(STRUCTURES_PATH + "GaAs_216_prim.yaml"))
    with pytest.raises(ValueError) as error:
        add_structure_random(
            Structure(**inputs),
            guest_structure="H2O",
            random_state=44,
            dist_threshold=10.0,
        )
    assert (
        str(error.value)
        == "Could not add guest structure, host structure seems to be too aggregated."
    )
