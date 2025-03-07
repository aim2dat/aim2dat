"""Test structure manipulation functions."""

# Standard library imports
import os
import numpy as np

# Third party library imports
import pytest

# Internal library imports
from aim2dat.strct import Structure
from aim2dat.strct.ext_manipulation import (
    add_structure_coord,
    add_structure_random,
    add_structure_position,
    rotate_structure,
)
from aim2dat.io import load_yaml_file

STRUCTURES_PATH = os.path.dirname(__file__) + "/structures/"
STRUCTURE_MANIPULATION_PATH = os.path.dirname(__file__) + "/structure_manipulation/"


def test_add_structure_coord_planar(structure_comparison):
    """Test add_structure_coord method for planar geometries."""
    strct = Structure.from_file(STRUCTURES_PATH + "MOF-5_prim.xsf")
    new_strct = add_structure_coord(
        strct,
        guest_structure="H2O",
        guest_dir=[1.0, 0.0, 0.0],
        min_dist_delta=0.2,
        host_indices=54,
        dist_threshold=None,
        bond_length=0.2,
    )
    ref_p = load_yaml_file(
        STRUCTURE_MANIPULATION_PATH + "MOF-5_prim_add_structure_coord_planar_ref.yaml"
    )
    structure_comparison(new_strct, ref_p)


def test_add_structure_coord_molecules(structure_comparison):
    """Test add_structure_coord method."""
    inputs = dict(load_yaml_file(STRUCTURES_PATH + "NH3.yaml"))
    new_strct = add_structure_coord(
        Structure(**inputs),
        host_indices=[1, 2, 3],
        guest_structure="OH",
        guest_dir=None,
        min_dist_delta=0.5,
        bond_length=1.0,
        dist_constraints=[(1, 1, 0.9)],
    )
    ref_p = load_yaml_file(STRUCTURE_MANIPULATION_PATH + "NH3-OH_ref.yaml")
    structure_comparison(new_strct, ref_p)


def test_add_structure_coord_molecules_2(structure_comparison):
    """Test add_structure_coord method."""
    inputs = dict(load_yaml_file(STRUCTURES_PATH + "NH3.yaml"))
    new_strct = add_structure_coord(
        Structure(**inputs),
        host_indices=[1, 2, 3],
        guest_index=0,
        guest_structure="H2O",
        guest_dir=None,
        min_dist_delta=0.1,
        bond_length=1.5,
        dist_constraints=[(2, 1, 1.5)],
    )
    ref_p = load_yaml_file(STRUCTURE_MANIPULATION_PATH + "NH3-H2O_ref.yaml")
    structure_comparison(new_strct, ref_p)


def test_add_structure_coord_molecules_3(structure_comparison):
    """
    Test corner cases of add_structure_coord method when bond_dir and
    guest_dir align or have opposite directions.
    """
    inputs = dict(load_yaml_file(STRUCTURES_PATH + "PCl5.yaml"))

    new_strct = add_structure_coord(
        Structure(**inputs),
        host_indices=0,
        guest_index=0,
        guest_structure="H2O",
        guest_dir=[-1.0, 0.0, 0.0],
        min_dist_delta=0.3,
        bond_length=1.5,
    )
    ref_p = load_yaml_file(STRUCTURE_MANIPULATION_PATH + "PCl5-H2O_1_ref.yaml")
    structure_comparison(new_strct, ref_p)

    new_strct = add_structure_coord(
        Structure(**inputs),
        host_indices=0,
        guest_index=0,
        guest_structure="H2O",
        guest_dir=[1.0, 0.0, 0.0],
        min_dist_delta=0.3,
        bond_length=1.5,
    )
    ref_p = load_yaml_file(STRUCTURE_MANIPULATION_PATH + "PCl5-H2O_2_ref.yaml")
    structure_comparison(new_strct, ref_p)

    new_strct = add_structure_coord(
        Structure(**inputs),
        host_indices=0,
        guest_index=0,
        guest_structure="H2O",
        min_dist_delta=0.3,
        bond_length=1.5,
        dist_constraints=[(0, 1, 2.2)],
    )
    ref_p = load_yaml_file(STRUCTURE_MANIPULATION_PATH + "PCl5-H2O_3_ref.yaml")
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


def test_add_structure_position(structure_comparison):
    """Test add_structure_random method for a crystal."""
    inputs = [
        dict(load_yaml_file(STRUCTURE_MANIPULATION_PATH + "PBI3.yaml")),
        dict(load_yaml_file(STRUCTURE_MANIPULATION_PATH + "CN2H5.yaml")),
    ]
    new_strct = add_structure_position(
        Structure(**inputs[0]),
        position=[2.88759377, 3.244215, 3.25149],
        guest_structure=Structure(**inputs[1]),
    )
    ref_p = load_yaml_file(STRUCTURE_MANIPULATION_PATH + "PBI3+CN2H5_ref.yaml")
    structure_comparison(new_strct, ref_p)

    with pytest.raises(ValueError) as error:
        new_strct = add_structure_position(
            Structure(**inputs[0]),
            position=[2.88759377, 3.244215, 3.25149],
            guest_structure=Structure(**inputs[1]),
            dist_threshold=10.0,
        )
    assert str(error.value) == "Atoms are too close to each other."


def test_scale_unit_cell_errors():
    """Test appropriate error rasing of scale_unit_cell function."""
    structure = Structure.from_file(STRUCTURES_PATH + "MOF-5_prim.xsf")
    with pytest.raises(TypeError) as error:
        structure.scale_unit_cell(scaling_factors=[0.0, 1.0, "c"])
    assert (
        str(error.value)
        == "`scaling_factors` must be of type float/int or a list of float/int values."
    )
    with pytest.raises(ValueError) as error:
        structure.scale_unit_cell(scaling_factors=[[0.0, 1.0, 1.0], [0.0, 0.0, 0.0]])
    assert (
        str(error.value)
        == "`scaling_factors` must be a single value, a list of 3 values, or a 3x3 nested list."
    )
    with pytest.raises(ValueError) as error:
        structure.scale_unit_cell(pressure=10.0)
    assert str(error.value) == "`bulk_modulus` must be provided when applying `pressure`."
    with pytest.raises(ValueError) as error:
        structure.scale_unit_cell()
    assert (
        str(error.value) == "Provide either `scaling_factors` or `pressure` (with `bulk_modulus`)."
    )


def test_scale_unit_cell_uniform_scaling():
    """Test scale_unit_cell with uniform scaling factors."""
    structure = Structure.from_file(STRUCTURES_PATH + "MOF-5_prim.xsf")
    scaling_factors = 1.1
    scaled_structure = structure.scale_unit_cell(scaling_factors=scaling_factors)
    expected_cell = np.array(structure["cell"]) * scaling_factors
    assert np.allclose(scaled_structure["cell"], expected_cell), "Uniform scaling failed"


def test_scale_unit_cell_anisotropic_scaling():
    """Test scale_unit_cell with anisotropic scaling factors."""
    structure = Structure.from_file(STRUCTURES_PATH + "MOF-5_prim.xsf")
    scaling_factors = [1.1, 1.2, 1.3]
    scaled_structure = structure.scale_unit_cell(scaling_factors=scaling_factors)
    expected_cell = np.dot(np.array(structure["cell"]), np.diag(scaling_factors))
    assert np.allclose(scaled_structure["cell"], expected_cell), "Anisotropic scaling failed"


def test_scale_unit_cell_pressure_based_scaling():
    """Test scale_unit_cell with pressure and bulk modulus."""
    structure = Structure.from_file(STRUCTURES_PATH + "MOF-5_prim.xsf")
    pressure = 10  # GPa
    bulk_modulus = 100  # GPa
    scaled_structure = structure.scale_unit_cell(pressure=pressure, bulk_modulus=bulk_modulus)
    strain = -pressure / bulk_modulus
    expected_cell = np.array(structure["cell"]) * (1 + strain)
    assert np.allclose(scaled_structure["cell"], expected_cell), "Pressure-based scaling failed"


def test_scale_unit_cell_uniform_strain():
    """Test scale_unit_cell with uniform strain."""
    structure = Structure.from_file(STRUCTURES_PATH + "MOF-5_prim.xsf")
    scaling_factors = 1.05  # 5% strain
    scaled_structure = structure.scale_unit_cell(scaling_factors=scaling_factors)
    expected_cell = np.array(structure["cell"]) * (scaling_factors)
    assert np.allclose(scaled_structure["cell"], expected_cell), "Uniform strain failed"


def test_scale_unit_cell_anisotropic_strain():
    """Test scale_unit_cell with anisotropic strain."""
    structure = Structure.from_file(STRUCTURES_PATH + "MOF-5_prim.xsf")
    scaling_factors = [1.02, 0.99, 1.03]
    scaled_structure = structure.scale_unit_cell(scaling_factors=scaling_factors)
    expected_cell = np.dot(np.array(structure["cell"]), np.diag(scaling_factors))
    assert np.allclose(scaled_structure["cell"], expected_cell), "Anisotropic strain failed"


def test_scale_unit_cell_full_strain_matrix():
    """Test scale_unit_cell with a 3x3 strain matrix."""
    structure = Structure.from_file(STRUCTURES_PATH + "MOF-5_prim.xsf")
    scaling_matrix = [[1.02, 0.01, 0.0], [0.01, 0.99, 0.0], [0.0, 0.02, 1.03]]
    scaled_structure = structure.scale_unit_cell(scaling_factors=scaling_matrix)
    expected_cell = np.dot(np.array(structure["cell"]), scaling_matrix)
    assert np.allclose(scaled_structure["cell"], expected_cell), "3x3 scaling matrix failed"


def test_rotate_structure(structure_comparison):
    """Test rotate_structure method for a crystal."""
    inputs = (
        dict(load_yaml_file(STRUCTURE_MANIPULATION_PATH + "PBI3+CN2H5_ref.yaml")),
        Structure.from_file(STRUCTURES_PATH + "MOF-5_prim.xsf"),
    )
    new_strct = rotate_structure(
        Structure(**inputs[0]),
        angles=[90, 0, 0],
        site_indices=[4, 5, 6, 7, 8, 9, 10, 11],
    )
    ref_p = load_yaml_file(STRUCTURE_MANIPULATION_PATH + "PBI3+CN2H5_rot_ref.yaml")
    structure_comparison(new_strct, ref_p)

    new_strct = rotate_structure(
        Structure(**inputs[1]),
        angles=90,
        site_indices=[44, 56, 76, 81, 57, 45, 77, 80, 104, 101, 105, 100],
        origin=[2.90097537349613, 6.52064699916667, 6.52064699916667],
        vector=[1.00000000e00, 2.44932357e-15, 2.44932357e-15],
    )
    ref_p = load_yaml_file(STRUCTURE_MANIPULATION_PATH + "MOF-5_prim.yaml")
    structure_comparison(new_strct, ref_p)
