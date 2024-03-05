"""Tests for the cp2k structure-parser."""

# Standard library imports
import os

# Third party library imports
import pytest

# Internal library imports
from aim2dat.aiida_workflows.cp2k.parser_utils import RestartStructureParser
from aim2dat.io.yaml import load_yaml_file

FOLDER_DIR = os.path.dirname(__file__) + "/cp2k/structure_parser_files/"


@pytest.mark.parametrize(
    "restart_file,reference_file",
    [
        ("/geo_opt/aiida-1.restart", "/geo_opt_reference.yaml"),
        ("/cell_opt/aiida-1.restart", "/cell_opt_reference.yaml"),
        (
            "/cell_opt_incomplete_numbers/aiida-1.restart",
            "/cell_opt_incomplete_numbers_reference.yaml",
        ),
        ("/md-nvt/aiida-1.restart", "/md-nvt_reference.yaml"),
    ],
)
def test_structure(restart_file, reference_file):
    """
    Test the cp2k structure-parser for a geo_opt calculation.
    """
    reference_values = list(load_yaml_file(FOLDER_DIR + reference_file))

    file_content = open(FOLDER_DIR + restart_file, "r").read()
    parser = RestartStructureParser(file_content)
    structure = parser.retrieve_output_structure()[0]
    for ref_value in reference_values:
        assert structure[ref_value[0]] == ref_value[1]
