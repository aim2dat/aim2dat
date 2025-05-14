"""Test cp2k calcjob."""

# Standard library imports
import os

# Third party library imports
import pytest

# Internal library imports
from aim2dat.strct import Structure
from aim2dat.io import read_yaml_file

MAIN_PATH = os.path.dirname(__file__) + "/cp2k/"
OUTPUTS_PATH = MAIN_PATH + "outputs/"
IO_OUTPUT_PATH = os.path.dirname(__file__) + "/../io/cp2k_stdout/"


@pytest.mark.aiida
@pytest.mark.parametrize("system, version", [("geo_opt_low", "2025.1")])
def test_standard_parser(
    structure_comparison,
    nested_dict_comparison,
    aiida_create_calcjob,
    aiida_create_remote_data,
    aiida_create_parser,
    system,
    version,
):
    """Test cp2k parser."""
    ref_output = read_yaml_file(OUTPUTS_PATH + version + "_standard_" + system + ".yaml")
    node = aiida_create_calcjob("aim2dat.cp2k", IO_OUTPUT_PATH + f"cp2k-{version}/{system}/")
    parser = aiida_create_parser("aim2dat.cp2k.standard")
    results, _ = parser.parse_from_node(
        node,
        store_provenance=False,
        retrieved_temporary_folder=IO_OUTPUT_PATH + f"cp2k-{version}/{system}/",
    )
    nested_dict_comparison(
        results["output_parameters"].get_dict(), ref_output["output_parameters"]
    )
    structure_comparison(
        Structure.from_aiida_structuredata(results["output_structure"]),
        ref_output["output_structure"],
    )


@pytest.mark.aiida
@pytest.mark.parametrize(
    "system,version", [("Sc2BDC3_el_pdos", "9.1"), ("Si_at_pdos", "9.1"), ("Si_uks_pdos", "9.1")]
)
def test_pdos_standard_parser(
    nested_dict_comparison,
    aiida_create_calcjob,
    aiida_create_remote_data,
    aiida_create_parser,
    system,
    version,
):
    """Test cp2k pdos parser."""
    ref_output = read_yaml_file(OUTPUTS_PATH + version + "_standard_" + system + ".yaml")
    node = aiida_create_calcjob(
        "aim2dat.cp2k", MAIN_PATH + f"output_files_{version}_standard_{system}"
    )
    parser = aiida_create_parser("aim2dat.cp2k.standard")
    results, _ = parser.parse_from_node(
        node,
        store_provenance=False,
        retrieved_temporary_folder=MAIN_PATH + f"output_files_{version}_standard_{system}",
    )
    nested_dict_comparison(
        results["output_parameters"].get_dict(), ref_output["output_parameters"]
    )
    if "output_pdos" in ref_output:
        x_data = results["output_pdos"].get_x()
        y_data = results["output_pdos"].get_y()
        assert x_data[0] == ref_output["output_pdos"]["x"][0], "PDOS: Wrong data_label for x."
        assert x_data[2] == ref_output["output_pdos"]["x"][2], "PDOS: Wrong unit for x."
        assert [
            abs(val - ref_val) < 1e-5
            for val, ref_val in zip(x_data[1], ref_output["output_pdos"]["x"][1])
        ], "PDOS: Wrong values for x."
        for orb_data, orb_ref in zip(y_data, ref_output["output_pdos"]["y"]):
            assert orb_data[0] == orb_ref[0], "PDOS: Wrong data_label for y."
            assert orb_data[2] == orb_ref[2], "PDOS: Wrong unit for y."
            assert [
                abs(val - ref_val) < 1e-5 for val, ref_val in zip(orb_data[1], orb_ref[1])
            ], "PDOS: Wrong values for y."
