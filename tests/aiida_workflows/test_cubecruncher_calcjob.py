"""Test cubecruncher calcjob."""

# Standard library imports
import os

# Third party library imports
import pytest
from aiida.common import datastructures

# Internal library imports
from aim2dat.aiida_workflows.utils import create_aiida_node
from aim2dat.io import read_yaml_file


MAIN_PATH = os.path.dirname(__file__) + "/cubecruncher/"
INPUTS_PATH = MAIN_PATH + "inputs/"
OUTPUTS_PATH = MAIN_PATH + "outputs/"
INPUT_FILES = MAIN_PATH + "input_files/"


@pytest.mark.aiida
@pytest.mark.parametrize("system", ["charge_density_difference"])
def test_input(
    aiida_sandbox_folder, aiida_get_calcinfo, aiida_create_remote_data, aiida_create_code, system
):
    """Test 'prepare_for_submission' functions."""
    input_details = read_yaml_file(INPUTS_PATH + f"{system}.yaml")
    input_p = input_details["inputs"]
    for input_key, input_value in input_p.items():
        input_p[input_key] = create_aiida_node(input_value)

    input_p["charge_density_folder"] = aiida_create_remote_data(INPUT_FILES)
    input_p["code"] = aiida_create_code("aim2dat.cubecruncher", "cubecruncher_code")
    calcinfo = aiida_get_calcinfo("aim2dat.cubecruncher", input_p, aiida_sandbox_folder)

    # Check general info:
    assert isinstance(calcinfo, datastructures.CalcInfo)
    assert isinstance(calcinfo.codes_info[0], datastructures.CodeInfo)
