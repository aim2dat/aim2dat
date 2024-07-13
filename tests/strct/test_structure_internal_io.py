"""Test internalb io backend of the Structure object."""

# Standard library imports
import os

# Third party library imports
import pytest

# Internal library imports
from aim2dat.strct import Structure
from aim2dat.io.yaml import load_yaml_file

IO_PATH = os.path.dirname(__file__) + "/io/"


@pytest.mark.parametrize("system", ["imidazole"])
def test_read_qe_input_structure(structure_comparison, system):
    """Test read_input_structure function."""
    structure = Structure.from_file(IO_PATH + "qe_input/" + system + ".in", backend="internal")
    structure_ref = load_yaml_file(IO_PATH + "qe_input/" + system + "_ref.yaml")
    structure_comparison(structure, structure_ref)


@pytest.mark.parametrize("restart_file", ["aiida-1.restart"])
def test_cp2k_read_restart_structure(structure_comparison, restart_file):
    """
    Test read cp2k restart function for single calculations.
    """
    ref = load_yaml_file(IO_PATH + "cp2k_restart/ref.yaml")
    structure = Structure.from_file(IO_PATH + "cp2k_restart/" + restart_file, backend="internal")
    structure_comparison(structure, ref)
