"""Test manipulation functions of the Structure class."""

# Standard library imports
import os
import numpy as np

# Third party library imports
import pytest

# Internal library imports
from aim2dat.strct import Structure


STRUCTURES_PATH = os.path.dirname(__file__) + "/structures/"
STRUCTURE_MANIPULATION_PATH = os.path.dirname(__file__) + "/structure_manipulation/"


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
