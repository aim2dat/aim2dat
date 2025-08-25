"""Test functions that generate input parameters for the cp2k core work chains."""

# Standard library imports
import os

# Third party library imports
import pytest
from aiida.orm import SinglefileData

# Internal library imports
from aim2dat.io import read_yaml_file
from aim2dat.aiida_workflows.cp2k.core_work_chain_inputs import _set_input_parameters
from aim2dat.aiida_workflows.cp2k.core_work_chain_scf import _initialize_scf_parameters

REF_PATH = os.path.dirname(__file__) + "/cp2k/inputs/"


@pytest.mark.parametrize(
    "structure,test_case",
    [
        ("Al_225_conv", "inp1"),
        ("Al_225_conv", "inp2"),
        ("Al_225_conv", "inp3"),
        ("Al_225_conv", "inp4"),
        ("Al_225_conv", "inp5"),
        ("Al_225_conv", "inp6"),
        ("Al_225_conv", "inp7"),
        ("Cs2Te_62_prim", "inp1"),
    ],
)
def test_inputs_general_features(
    structure, test_case, aiida_create_wc_inputs, nested_dict_comparison
):
    """Test function _set_input_parameters."""
    ref = dict(read_yaml_file(REF_PATH + structure + "_" + test_case + "_ref.yaml"))
    inputs, ctx, strct_node = aiida_create_wc_inputs(structure, ref)
    cp2k_code = ref.get("cp2k_code")
    if cp2k_code is not None:
        ctx.inputs.code.full_label = cp2k_code["full_label"]
        ctx.inputs.code.description = cp2k_code["description"]
    error = _set_input_parameters(inputs, ctx, **ref["add_args"])
    if "error" in ref:
        assert ref["error"][0] == error[0]
        assert ref["error"][1] == error[1]
    else:
        nested_dict_comparison(ctx.scf_m_info, ref["ctx"]["scf_m_info"])
        nested_dict_comparison(
            ctx.inputs.parameters.get_dict(), ref["ctx"]["inputs"]["parameters"]
        )
        assert ctx.inputs.structure == strct_node


def test_inputs_files(aiida_create_wc_inputs, nested_dict_comparison):
    """Test function _set_input_parameters."""
    ref = dict(read_yaml_file(REF_PATH + "Al_225_conv_inp1" + "_ref.yaml"))
    inputs, ctx, strct_node = aiida_create_wc_inputs("Al_225_conv", ref)
    single_file = SinglefileData(REF_PATH + "Al_225_conv_inp1" + "_ref.yaml")
    inputs.numerical_p.basis_file = single_file
    inputs.numerical_p.pseudo_file = single_file
    _set_input_parameters(inputs, ctx, **ref["add_args"])
    nested_dict_comparison(ctx.scf_m_info, ref["ctx"]["scf_m_info"])
    ref["ctx"]["inputs"]["parameters"]["FORCE_EVAL"]["DFT"]["BASIS_SET_FILE_NAME"] = ref["ctx"][
        "inputs"
    ]["parameters"]["FORCE_EVAL"]["DFT"]["BASIS_SET_FILE_NAME"] + ["Al_225_conv_inp1_ref.yaml"]
    ref["ctx"]["inputs"]["parameters"]["FORCE_EVAL"]["DFT"][
        "POTENTIAL_FILE_NAME"
    ] = "Al_225_conv_inp1_ref.yaml"
    nested_dict_comparison(ctx.inputs.parameters.get_dict(), ref["ctx"]["inputs"]["parameters"])
    assert ctx.inputs.structure == strct_node
    assert ctx.inputs.file["pseudo"] == single_file
    assert ctx.inputs.file["basis"] == single_file


def test_inputs_resources(aiida_create_wc_inputs, nested_dict_comparison):
    """Test function _set_input_parameters."""
    ref = dict(read_yaml_file(REF_PATH + "Al_225_conv_inp1" + "_ref.yaml"))
    inputs, ctx, strct_node = aiida_create_wc_inputs("Al_225_conv", ref)
    ctx.inputs.metadata["options"]["resources"] = {
        "coeff": 2.0,
        "exp": 2.5,
        "shift": 10.0,
        "num_mpiprocs_per_machine": 12,
        "num_cores_per_mpiproc": 2,
    }
    _set_input_parameters(inputs, ctx, **ref["add_args"])
    nested_dict_comparison(ctx.scf_m_info, ref["ctx"]["scf_m_info"])
    nested_dict_comparison(ctx.inputs.parameters.get_dict(), ref["ctx"]["inputs"]["parameters"])
    assert ctx.inputs.metadata == {
        "options": {
            "resources": {
                "exp": 2.5,
                "num_cores_per_mpiproc": 2,
                "num_machines": 42,
                "num_mpiprocs_per_machine": 12,
                "shift": 10.0,
            }
        }
    }
    assert ctx.inputs.structure == strct_node


@pytest.mark.parametrize(
    "structure,test_case",
    [
        ("Cs2Te_62_prim", "scf1"),
        ("Cs2Te_62_prim", "scf2"),
        ("H2O", "scf1"),
        ("H2O", "scf2"),
        ("H2O", "scf3"),
        ("Al_225_conv", "scf1"),
    ],
)
def test_scf_initialization(aiida_create_wc_inputs, nested_dict_comparison, structure, test_case):
    """Test SCF initialization."""
    ref = dict(read_yaml_file(REF_PATH + structure + "_" + test_case + "_ref.yaml"))
    inputs, ctx, strct_node = aiida_create_wc_inputs(structure, ref)
    _set_input_parameters(inputs, ctx, **ref["add_args"])
    error, reports = _initialize_scf_parameters(inputs, ctx)
    if "error" in ref:
        assert ref["error"][0] == error[0]
        assert ref["error"][1] == error[1]
        assert ref["reports"] == reports
    else:
        nested_dict_comparison(ctx.scf_m_info, ref["ctx"]["scf_m_info"])
        nested_dict_comparison(ctx.cur_scf_p, ref["ctx"]["cur_scf_p"])
        nested_dict_comparison(ctx.cur_scf_p_set, ref["ctx"]["cur_scf_p_set"])
        nested_dict_comparison(
            ctx.inputs.parameters.get_dict(), ref["ctx"]["inputs"]["parameters"]
        )
        assert reports == ref["reports"]
        assert ctx.inputs.structure == strct_node
