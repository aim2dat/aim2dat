"""Test the general interface of the StructuresOperations class."""

# Standard library imports
import os

# Third party library imports
import pandas as pd
import pytest

# Internal library imports
from aim2dat.strct import StructureOperations, StructureCollection
from aim2dat.io.yaml import load_yaml_file

STRUCTURES_PATH = os.path.dirname(__file__) + "/structures/"


def test_structure_ops_basics():
    """Test basic features of the StructuresOperations class."""
    strct_collect = StructureCollection()
    inputs = dict(load_yaml_file(STRUCTURES_PATH + "Cs2Te_62_prim" + ".yaml"))
    strct_collect.append("Cs2Te_62_prim", **inputs)

    strct_ops = StructureOperations(structures=strct_collect, append_to_coll=True)

    with pytest.raises(TypeError) as error:
        strct_ops.append_to_coll = "test"
    assert str(error.value) == "`append_to_coll` needs to be of type bool."

    assert isinstance(strct_ops.calculate_distance("Cs2Te_62_prim", 0, 1), float)
    assert isinstance(strct_ops.calculate_distance(["Cs2Te_62_prim"], 0, 1), dict)
    strct_ops.output_format = "DataFrame"
    assert isinstance(strct_ops.calculate_distance(["Cs2Te_62_prim"], 0, 1), pd.DataFrame)
    with pytest.raises(TypeError) as error:
        strct_ops.output_format = 10
    assert str(error.value) == "`output_format` needs to be of type str."
    with pytest.raises(ValueError) as error:
        strct_ops.output_format = "DataFramE"
    assert (
        str(error.value)
        == "`output_format` 'DataFramE' is not supported."
        + " It has to be one of the following options: ['dict', 'DataFrame']"
    )

    inputs = dict(load_yaml_file(STRUCTURES_PATH + "NaCl_225_prim" + ".yaml"))
    strct_collect.append("NaCl_225_prim", **inputs)

    with pytest.raises(ValueError) as error:
        strct_ops.compare_structures_via_ffingerprint(0, 1, distinguish_kinds=True)
    assert str(error.value) == "If `distinguish_kinds` is true `kinds` need to be set."

    with pytest.raises(ValueError) as error:
        strct_ops.compare_sites_via_ffingerprint(0, 1, 20, 20)
    assert str(error.value) == "Site index out of range for structure '0'."

    # strct_ops.output_format = "dict"
    # with pytest.raises(TypeError):
    #     strct_ops.calculate_distance("Cs2Te_62_prim", 0, 1, new_label="TEST_ERROR")
    #
    # assert type(strct_ops.scale_unit_cell("Cs2Te_62_prim", 2)) == Structure
    # assert strct_ops.structures.labels == ["Cs2Te_62_prim"]
    # strct_ops.change_label = True
    # assert type(strct_ops.scale_unit_cell(["Cs2Te_62_prim"], 2)) == StructureCollection
    # assert strct_ops.structures.labels == ["Cs2Te_62_prim", "Cs2Te_62_prim_scaled-2"]
    #
    # strct_ops_return = StructuresOperations(strct_collect, append_to_coll=False)
    # assert isinstance(strct_ops_return.scale_unit_cell("Cs2Te_62_prim", 2), Structure)
    # strct_c_scaled = strct_ops_return.scale_unit_cell(["Cs2Te_62_prim"], 2)
    # assert isinstance(strct_c_scaled, StructureCollection)
    # assert len(strct_ops_return.structures) == 1
    # assert len(strct_c_scaled) == 2
    # assert strct_c_scaled.labels == ["Cs2Te_62_prim", "Cs2Te_62_prim_scaled"]
    #
    # strct_ops_return_overw = StructuresOperations(
    #     strct_collect, append_to_coll=False, overwrite_manip_strcts=True
    # )
    # strct_c_scaled = strct_ops_return_overw.scale_unit_cell(["Cs2Te_62_prim"], 2)
    # assert isinstance(strct_c_scaled, StructureCollection)
    # assert len(strct_ops_return_overw.structures) == 1
    # assert len(strct_c_scaled) == 1
    # assert strct_c_scaled.labels == ["Cs2Te_62_prim"]
