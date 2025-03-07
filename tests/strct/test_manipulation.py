"""Test manipulation functions of the Structure class."""

# Standard library imports
import os
import numpy as np

# Third party library imports
import pytest

# Internal library imports
from aim2dat.strct import Structure, StructureCollection, StructureOperations
from aim2dat.io.yaml import load_yaml_file


STRUCTURES_PATH = os.path.dirname(__file__) + "/structures/"
STRUCTURE_MANIPULATION_PATH = os.path.dirname(__file__) + "/structure_manipulation/"


@pytest.mark.parametrize("structure", ["Benzene"])
def test_delete_atoms(structure_comparison, structure):
    """Test delete atoms method."""
    strct = Structure(
        **dict(load_yaml_file(STRUCTURES_PATH + structure + ".yaml")), label="Benzene"
    )
    ref_p = load_yaml_file(STRUCTURE_MANIPULATION_PATH + structure + "_ref.yaml")
    ref_p["structure"]["label"] = structure
    new_strct = strct.delete_atoms(**ref_p["function_args"], change_label=True)
    ref_p["structure"]["label"] += "_del"
    structure_comparison(new_strct, ref_p["structure"])


@pytest.mark.parametrize("structure", ["Cs2Te_62_prim", "GaAs_216_prim", "Cs2Te_19_prim_kinds"])
def test_element_substitution(structure_comparison, structure):
    """Test element substitution method."""
    inputs = dict(load_yaml_file(STRUCTURES_PATH + structure + ".yaml"))
    inputs["label"] = structure
    inputs2 = dict(load_yaml_file(STRUCTURES_PATH + "Al_225_conv.yaml"))
    inputs2["label"] = "Al_test"
    ref_p = load_yaml_file(STRUCTURE_MANIPULATION_PATH + structure + "_ref.yaml")
    strct_collect = StructureCollection()
    strct_collect.append(**inputs)
    strct_collect.append(**inputs2)
    strct_ops = StructureOperations(strct_collect)
    elements = ref_p["function_args"]["elements"]
    elements = elements if isinstance(elements[0], (list, tuple)) else [elements]
    subst_strct = strct_ops[0].substitute_elements(**ref_p["function_args"], change_label=True)
    assert len(strct_ops.structures) == 2
    structure_comparison(strct_ops.structures[0], inputs)
    structure_comparison(strct_ops.structures[1], inputs2)
    structure_comparison(subst_strct, ref_p["structure"])


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


@pytest.mark.parametrize("new_label", ["GaAs_216_prim", "GaAs_216_prim_scaled-0.7"])
def test_scale_unit_cell(structure_comparison, new_label):
    """Test scale unit cell function."""
    inputs = dict(load_yaml_file(STRUCTURES_PATH + "GaAs_216_prim.yaml"))
    ref = dict(
        load_yaml_file(STRUCTURE_MANIPULATION_PATH + "GaAs_216_prim_scale_unit_cell_ref.yaml")
    )
    ref["structure"]["label"] = new_label
    strct = Structure(**inputs, label="GaAs_216_prim")
    scaled_strct = strct.scale_unit_cell(
        **ref["function_args"], change_label="scaled" in new_label
    )
    structure_comparison(scaled_strct, ref["structure"])


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
