"""Test critic2 calcjob."""

# Standard library imports
import os

# Third party library imports
import pytest
from aiida.common import datastructures

# Internal library imports
from aim2dat.aiida_workflows.utils import create_aiida_node
from aim2dat.io import read_yaml_file


MAIN_PATH = os.path.dirname(__file__) + "/critic2/"
INPUTS_PATH = MAIN_PATH + "inputs/"
OUTPUTS_PATH = MAIN_PATH + "outputs/"
INPUT_FILES = MAIN_PATH + "input_files/"


@pytest.mark.aiida
@pytest.mark.parametrize("system", ["bader_charges", "rhodef_plane"])
def test_input(
    aiida_sandbox_folder, aiida_get_calcinfo, aiida_create_remote_data, aiida_create_code, system
):
    """Test 'prepare_for_submission' functions."""
    input_details = read_yaml_file(INPUTS_PATH + f"{system}.yaml")
    input_p = input_details["inputs"]
    for input_key, input_value in input_p.items():
        input_p[input_key] = create_aiida_node(input_value)

    input_p["charge_density_folder"] = aiida_create_remote_data(INPUT_FILES)
    input_p["code"] = aiida_create_code("aim2dat.critic2", "critic2_code")
    calcinfo = aiida_get_calcinfo("aim2dat.critic2", input_p, aiida_sandbox_folder)

    # Check general info:
    assert isinstance(calcinfo, datastructures.CalcInfo)
    assert isinstance(calcinfo.codes_info[0], datastructures.CodeInfo)
    assert calcinfo.retrieve_list == input_details["retrieve_list"]

    # Check input file:
    with aiida_sandbox_folder.open("aiida.in") as fobj:
        for line, ref_line in zip(fobj, input_details["input_file"]):
            assert line.strip() == ref_line


@pytest.mark.aiida
@pytest.mark.parametrize("system", ["bader_charges"])  # , "rhodef_plane"])
def test_parser(aiida_create_calcjob, aiida_create_remote_data, aiida_create_parser, system):
    """Test parser."""
    ref_output = read_yaml_file(OUTPUTS_PATH + system + ".yaml")
    node = aiida_create_calcjob("aim2dat.critic2", MAIN_PATH + f"output_files_{system}")
    parser = aiida_create_parser("aim2dat.critic2")
    results, _ = parser.parse_from_node(node, store_provenance=False)

    output_parameters = results["output_parameters"].get_dict()
    for output_p, rev_value in ref_output["output_parameters"].items():
        assert output_parameters[output_p] == rev_value, f"Value for {output_p} does not match."

    if "output_bader_populations" in ref_output:
        bader_populations = results["output_bader_populations"].get_list()
        for atom, atom_ref in zip(bader_populations, ref_output["output_bader_populations"]):
            assert (
                atom["element"] == atom_ref["element"]
            ), "Elements of the bader population analysis don't match."
            assert (
                abs(atom["population"] - atom_ref["population"]) < 1e-5
            ), "Populations of the bader population analysis don't match."

    # if "planes" in ref_output:
    #     planes = results["output_planes"]
    #     print(planes)
    # for plane_label, ref_plane in ref_output["planes"].items():
    #     assert plane_label in planes, f"Plane {plane_label} not parsed."
    #     assert (
    #         planes[plane_label]["coordinates_unit"] == ref_plane["coordinates_unit"]
    #     ), "Plane units don't match."
    #     for coord, coord_ref in zip(
    #         planes[plane_label]["coordinates"], ref_plane["coordinates"]
    #     ):
    #         for dir0 in range(2):
    #             assert (
    #                 abs(coord[dir0] - coord_ref[dir0]) < 1e-5
    #             ), "Plane coordinates don't match."
    #     for value, value_ref in zip(planes[plane_label]["values"], ref_plane["values"]):
    #         if isinstance(value_ref, float):
    #             assert abs(value - value_ref) < 1e-5, "Plane values don't match."
    #         else:
    #             for f_idx in range(len(value_ref)):
    #                 assert (
    #                     abs(value[f_idx] - value_ref[f_idx]) < 1e-5
    #                 ), "Plane values don't match."
