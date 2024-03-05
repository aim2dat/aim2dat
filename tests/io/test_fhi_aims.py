"""Test the cp2k module of the io sub-package."""

# Standard library imports
import os

# Third party library imports
import pytest

# Internal library imports
from aim2dat.io.yaml import load_yaml_file
from aim2dat.io.fhi_aims import read_band_structure, read_atom_proj_density_of_states

cwd = os.path.dirname(__file__) + "/"
BAND_STRUCTURE_PATH = cwd + "fhi_aims_band_structure/"
PDOS_PATH = cwd + "fhi_aims_pdos/"


def test_errors():
    """Test errors."""
    with pytest.raises(ValueError) as error:
        read_band_structure(cwd + "empty_folder/")
    assert str(error.value) == "No band structure files found."

    with pytest.raises(ValueError) as error:
        read_atom_proj_density_of_states(cwd + "empty_folder/")
    assert str(error.value) == "No pDOS files found."

    with pytest.raises(ValueError) as error:
        read_band_structure(BAND_STRUCTURE_PATH + "no_soc", soc=True)
    assert (
        str(error.value)
        == "Spin-orbit coupling activated but the files don't have the right naming scheme."
    )

    with pytest.raises(ValueError) as error:
        read_atom_proj_density_of_states(PDOS_PATH + "no_soc", soc=True)
    assert (
        str(error.value)
        == "Spin-orbit coupling activated but the files don't have the right naming scheme."
    )


@pytest.mark.parametrize("system,soc", [("Cs3Sb_soc", False), ("Cs3Sb_soc", True)])
def test_read_band_structure(nested_dict_comparison, system, soc):
    """Test read_band_structure function."""
    bands_data = read_band_structure(BAND_STRUCTURE_PATH + system + "/", soc=soc)
    ref_label = "/ref"
    if soc:
        ref_label += "_soc"
    bands_ref = dict(load_yaml_file(BAND_STRUCTURE_PATH + system + ref_label + ".yaml"))
    nested_dict_comparison(bands_data, bands_ref)


@pytest.mark.parametrize(
    "system,soc,load_raw", [("Cs3Sb_soc", False, False), ("Cs3Sb_soc", True, True)]
)
def test_read_atom_proj_density_of_states(nested_dict_comparison, system, soc, load_raw):
    """Test read_atom_proj_density_of_states function."""
    pdos_data = read_atom_proj_density_of_states(
        PDOS_PATH + system + "/", soc=soc, load_raw=load_raw
    )
    ref_label = "/ref"
    if soc:
        ref_label += "_soc"
    if load_raw:
        ref_label += "_raw"
    pdos_ref = dict(load_yaml_file(PDOS_PATH + system + ref_label + ".yaml"))
    nested_dict_comparison(pdos_data, pdos_ref)
