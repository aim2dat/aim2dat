"""Test the qe module of the io sub-package."""

# Standard library imports
import os

# Third party library imports
import pytest

# Internal library imports
from aim2dat.io.yaml import load_yaml_file
from aim2dat.io.qe import (
    read_input_structure,
    read_band_structure,
    read_total_density_of_states,
    read_atom_proj_density_of_states,
)

INPUT_PATH = os.path.dirname(__file__) + "/qe_input/"
BAND_STRUCTURE_PATH = os.path.dirname(__file__) + "/qe_band_structure/"
TDOS_PATH = os.path.dirname(__file__) + "/qe_tdos/"
PDOS_PATH = os.path.dirname(__file__) + "/qe_pdos/"


@pytest.mark.parametrize("system", ["imidazole"])
def test_read_input_structure(nested_dict_comparison, system):
    """Test read_input_structure function."""
    structure = read_input_structure(INPUT_PATH + system + ".in")
    structure_ref = load_yaml_file(INPUT_PATH + system + "_ref.yaml")
    nested_dict_comparison(structure, structure_ref)


def test_read_band_structure(nested_dict_comparison):
    """Test read_band_structure function."""
    bands_data = read_band_structure(BAND_STRUCTURE_PATH + "bands.dat")
    bands_ref = load_yaml_file(BAND_STRUCTURE_PATH + "ref.yaml")
    nested_dict_comparison(bands_data, bands_ref)


def test_read_total_density_of_states(nested_dict_comparison):
    """Test read_total_density_of_states function."""
    tdos_data = read_total_density_of_states(TDOS_PATH + "dos.dat")
    tdos_ref = load_yaml_file(TDOS_PATH + "ref.yaml")
    nested_dict_comparison(tdos_data, tdos_ref)


def test_read_atom_proj_density_of_states(nested_dict_comparison):
    """Test read_atom_proj_density_of_states function."""
    pdos_data = read_atom_proj_density_of_states(PDOS_PATH)
    pdos_ref = load_yaml_file(PDOS_PATH + "ref.yaml")
    nested_dict_comparison(pdos_data, pdos_ref)
