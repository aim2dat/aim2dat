"""Test the general interface of the StructuresOperations class."""

# Standard library imports
import os

# Third party library imports
import pandas as pd
import pytest

# Internal library imports
from aim2dat.strct import StructureOperations, StructureCollection, Structure
from aim2dat.io import read_yaml_file

STRUCTURES_PATH = os.path.dirname(__file__) + "/structures/"
REF_PATH = os.path.dirname(__file__) + "/structure_op/"


def test_structure_op_basics():
    """Test basic features of the StructuresOperations class."""
    strct_collect = StructureCollection()
    inputs = dict(read_yaml_file(STRUCTURES_PATH + "Cs2Te_62_prim.yaml"))
    strct_collect.append("Cs2Te_62_prim", **inputs)

    inputs = dict(read_yaml_file(STRUCTURES_PATH + "NaCl_225_prim.yaml"))
    strct_collect.append("NaCl_225_prim", **inputs)

    strct_ops = StructureOperations(structures=strct_collect)
    strct_ops_inp_list = StructureOperations([Structure(label="Cs2Te_62_prim", **inputs)])
    assert isinstance(strct_ops_inp_list.structures, StructureCollection)

    with pytest.raises(TypeError) as error:
        StructureOperations([1, 2, 3])

    assert isinstance(strct_ops["Cs2Te_62_prim"].calc_distance(0, 1), float)
    assert isinstance(strct_ops[["Cs2Te_62_prim", "NaCl_225_prim"]].calc_distance(0, 1), dict)
    strct_ops.output_format = "DataFrame"
    assert isinstance(
        strct_ops[["Cs2Te_62_prim", "NaCl_225_prim"]].calc_distance(0, 1), pd.DataFrame
    )
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

    with pytest.raises(ValueError) as error:
        strct_ops.compare_structures_via_ffingerprint(0, 1, distinguish_kinds=True)
    assert str(error.value) == "If `distinguish_kinds` is true, all `kinds` must be unequal None."

    with pytest.raises(ValueError) as error:
        strct_ops.compare_sites_via_ffingerprint(0, 1, 20, 20)
    assert str(error.value) == "Site index out of range for structure '0'."


def test_structure_op_pipeline(structure_comparison):
    """Test pipeline implementation of the StructureOperations class."""
    ref = read_yaml_file(REF_PATH + "pipeline.yaml")
    strct_c = StructureCollection()
    for file_name, label in ref["input_structures"]:
        strct_c.append_structure(Structure.from_file(STRUCTURES_PATH + file_name), label=label)
    strct_op = StructureOperations(strct_c)
    strct_op.pipeline = ref["pipeline"]
    # TODO test output pipeline, return None if pipeline is not set and errors.
    structures = strct_op.run_pipeline()
    assert len(structures) == len(ref["output_structures"])
    for strct, ref_strct in zip(structures, ref["output_structures"]):
        structure_comparison(strct, ref_strct)
    assert len(strct_op.structures) == len(strct_c)
    for strct0, strct1 in zip(strct_op.structures, strct_c):
        structure_comparison(strct0, strct1)
