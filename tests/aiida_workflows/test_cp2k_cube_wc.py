"""Test cp2k cube work chain."""

# Standard library imports
import os

# Third party library imports
import pytest

# Internal library import
from aim2dat.io import read_yaml_file
from aim2dat.aiida_workflows.utils import create_aiida_node
from aim2dat.aiida_workflows.cp2k.cube_work_chain import (
    _validate_cube_types,
    _setup_wc_specific_inputs,
)


REF_PATH = os.path.dirname(__file__) + "/cp2k/inputs/"


@pytest.mark.parametrize(
    "input_value,return_value",
    [
        (
            ["_"],
            "Only 'efield', 'elf', 'external_potential', 'e_density', "
            + "'tot_density', 'v_hartree', 'v_xc' are supported cube types.",
        ),
        (
            ["e_density", "e_density"],
            "Field types cannot be calculated twice.",
        ),
        (["e_density", "tot_density"], None),
    ],
)
def test_validate_cube_types(input_value, return_value):
    """Test cube types input validation for cp2k cube work chain."""
    assert _validate_cube_types(create_aiida_node(input_value), None) == return_value


def test_setup_wc_specific_inputs(aiida_create_wc_inputs, nested_dict_comparison):
    """Test wc_specific_inputs fucntion for cp2k cube work chain."""
    ref = dict(read_yaml_file(REF_PATH + "Al_225_conv_cube_ref.yaml"))
    inputs, ctx, strct_node = aiida_create_wc_inputs("Al_225_conv", ref)
    _setup_wc_specific_inputs(ctx, inputs)
    nested_dict_comparison(ctx.inputs.parameters.get_dict(), ref["ref"]["parameters"])
    nested_dict_comparison(ctx.inputs.settings.get_dict(), ref["ref"]["settings"])
