"""Test error handlers of the core work chains."""

# Standard library imports
import os
from collections import namedtuple

# Third party library imports
import pytest
from aiida.engine.processes.exit_code import ExitCodesNamespace
from aiida.common.extendeddicts import AttributeDict

# Internal library imports
from aim2dat.io import read_yaml_file
from aim2dat.aiida_workflows.utils import create_aiida_node
from aim2dat.utils.dict_tools import dict_set_parameter
from aim2dat.aiida_workflows.cp2k.core_work_chain_inputs import _set_input_parameters
from aim2dat.aiida_workflows.cp2k.core_work_chain_scf import _initialize_scf_parameters
from aim2dat.aiida_workflows.cp2k.core_work_chain_opt import _initialize_opt_parameters
from aim2dat.aiida_workflows.cp2k.core_work_chain_handlers import (
    _switch_scf_parameters,
    _switch_to_open_shell_ks,
    _switch_to_atomic_scf_guess,
    _switch_to_broyden_mixing,
    _resubmit_unconverged_geometry,
    _resubmit_calculation,
)


REF_PATH = os.path.dirname(__file__) + "/cp2k/handlers/"


@pytest.mark.parametrize(
    "structure,test_case,handler_function",
    [
        ("Al_225_conv", "switch_scf_1", _switch_scf_parameters),
        ("Al_225_conv", "switch_scf_2", _switch_scf_parameters),
        ("Al_225_conv", "switch_scf_3", _switch_scf_parameters),
        ("Al_225_conv", "switch_scf_4", _switch_scf_parameters),
        ("Al_225_conv", "open_shell_1", _switch_to_open_shell_ks),
        ("Al_225_conv", "open_shell_2", _switch_to_open_shell_ks),
        ("Al_225_conv", "atomic_guess_1", _switch_to_atomic_scf_guess),
        ("Al_225_conv", "atomic_guess_2", _switch_to_atomic_scf_guess),
        ("Al_225_conv", "atomic_guess_3", _switch_to_atomic_scf_guess),
        ("Al_225_conv", "pulay_1", _switch_to_broyden_mixing),
        ("Al_225_conv", "pulay_2", _switch_to_broyden_mixing),
        ("Al_225_conv", "pulay_3", _switch_to_broyden_mixing),
    ],
)
def test_scf_error_handlers(
    structure, test_case, handler_function, aiida_create_wc_inputs, nested_dict_comparison
):
    """Test 'switch to atomic scf guess' handler."""
    ExitCode = namedtuple("ExitCode", "exit_status message")
    ref = dict(read_yaml_file(REF_PATH + structure + "_" + test_case + "_ref.yaml"))
    inputs, ctx, strct_node = aiida_create_wc_inputs(structure, ref)
    _set_input_parameters(inputs, ctx, **ref["add_args"])
    _initialize_scf_parameters(inputs, ctx)
    exit_codes = ExitCodesNamespace(
        {"ERROR_CALCULATION_ABORTED": ExitCode(611, "Calculation did not finish properly.")}
    )
    ref["calc"]["outputs"]["output_parameters"] = create_aiida_node(
        ref["calc"]["outputs"]["output_parameters"]
    )
    error_handler, reports = handler_function(inputs, ctx, exit_codes, AttributeDict(ref["calc"]))
    if "reports" in ref:
        assert len(reports) == len(ref["reports"])
        for rep, rep_ref in zip(reports, ref["reports"]):
            assert rep == rep_ref
    if ref["error_handler"] is None:
        assert error_handler is None
    else:
        assert error_handler.do_break == ref["error_handler"]["do_break"]
        assert error_handler.exit_code[0] == ref["error_handler"]["exit_code"]["exit_status"]
        assert error_handler.exit_code[1] == ref["error_handler"]["exit_code"]["message"]
        nested_dict_comparison(ctx.scf_m_info, ref["updated_scf_m_info"])
        nested_dict_comparison(ctx.cur_scf_p, ref["updated_cur_scf_p"])
        nested_dict_comparison(ctx.inputs.parameters.get_dict(), ref["updated_cp2k_parameters"])


@pytest.mark.parametrize(
    "structure,test_case,handler_function",
    [
        ("Al_225_conv", "unconv_structure_1", _resubmit_unconverged_geometry),
        ("Al_225_conv", "unconv_structure_2", _resubmit_unconverged_geometry),
        ("Al_225_conv", "unfinished_structure_1", _resubmit_calculation),
        ("Al_225_conv", "unfinished_structure_2", _resubmit_calculation),
    ],
)
def test_opt_error_handlers(
    structure,
    test_case,
    handler_function,
    aiida_create_wc_inputs,
    aiida_create_structuredata,
    nested_dict_comparison,
):
    """Test optimization error handlers."""
    ExitCode = namedtuple("ExitCode", "exit_status message")
    ref = dict(read_yaml_file(REF_PATH + structure + "_" + test_case + "_ref.yaml"))
    inputs, ctx, strct_node = aiida_create_wc_inputs(structure, ref)
    _set_input_parameters(inputs, ctx, **ref["add_args"])
    exit_codes = ExitCodesNamespace(
        {"ERROR_CALCULATION_ABORTED": ExitCode(611, "Calculation did not finish properly.")}
    )
    _initialize_opt_parameters(inputs, ctx, exit_codes, ref["opt_type"])
    ref["calc"]["outputs"]["output_parameters"] = create_aiida_node(
        ref["calc"]["outputs"]["output_parameters"]
    )
    outp_structure = aiida_create_structuredata(structure)
    ref["calc"]["outputs"]["output_structure"] = outp_structure
    ctx.inputs.parent_calc_folder = None
    parameters = ctx.inputs.parameters.get_dict()
    dict_set_parameter(parameters, ["MOTION", "CELL_OPT", "STEP_START_VAL"], 20)
    error_handler, reports = handler_function(inputs, ctx, exit_codes, AttributeDict(ref["calc"]))
    if ref["error_handler"] is None:
        assert error_handler is None
    else:
        assert error_handler.do_break == ref["error_handler"]["do_break"]
        assert error_handler.exit_code[0] == ref["error_handler"]["exit_code"]["exit_status"]
        assert error_handler.exit_code[1] == ref["error_handler"]["exit_code"]["message"]
        assert ctx.opt_level == ref["updated_opt_level"]
        assert ctx.opt_iteration == ref["updated_opt_iteration"]
        if error_handler.exit_code[0] == 0:
            assert ctx.inputs.parent_calc_folder == "output_folder"
        nested_dict_comparison(ctx.inputs.parameters.get_dict(), ref["updated_cp2k_parameters"])
