"""Test the cp2k module of the io sub-package."""

# Standard library imports
import os

# Third party library imports
import pytest
import numpy as np
import h5py

# Internal library imports
from aim2dat.io import (
    read_yaml_file,
    read_fhiaims_band_structure,
    read_fhiaims_total_dos,
    read_fhiaims_proj_dos,
)

cwd = os.path.dirname(__file__) + "/"
BAND_STRUCTURE_PATH = cwd + "fhi_aims_band_structure/"
PDOS_PATH = cwd + "fhi_aims_pdos/"


def test_errors():
    """Test errors."""
    with pytest.raises(ValueError) as error:
        read_fhiaims_band_structure(cwd + "empty_folder/")
    assert str(error.value) == "No files with the correct naming scheme found."

    with pytest.raises(ValueError) as error:
        read_fhiaims_proj_dos(cwd + "empty_folder/")
    assert str(error.value) == "No files with the correct naming scheme found."

    with pytest.raises(ValueError) as error:
        read_fhiaims_band_structure(BAND_STRUCTURE_PATH + "Cs3Sb", soc=True)
    assert (
        str(error.value)
        == "Spin-orbit coupling activated but the files don't have the proper naming scheme."
    )

    with pytest.raises(ValueError) as error:
        read_fhiaims_proj_dos(PDOS_PATH + "g_qantum_number", soc=True)
    assert (
        str(error.value)
        == "Spin-orbit coupling activated but the files don't have the proper naming scheme."
    )


@pytest.mark.parametrize(
    "system,soc", [("Cs3Sb", False), ("Cs3Sb_soc", False), ("Cs3Sb_soc", True)]
)
def test_read_fhiaims_band_structure(nested_dict_comparison, system, soc):
    """Test read_fhiaims_band_structure function."""
    bands_data = read_fhiaims_band_structure(BAND_STRUCTURE_PATH + system + "/", soc=soc)
    assert bands_data["unit_y"] == "eV"
    ref_label = "/ref"
    if soc:
        ref_label += "_soc"
    with h5py.File(BAND_STRUCTURE_PATH + system + ref_label + ".h5", "r") as fobj:
        for key in ["kpoints", "bands", "occupations"]:
            np.testing.assert_allclose(bands_data[key], fobj[key][:], atol=1.0e-5)


def test_read_fhiaims_total_dos():
    """Test read_fhiaims_total_dos function."""
    tdos_data = read_fhiaims_total_dos(PDOS_PATH + "Cs3Sb_soc/KS_DOS_total_raw.dat")
    assert tdos_data["unit_x"] == "eV"
    with h5py.File(PDOS_PATH + "Cs3Sb_soc/ref_tdos.h5", "r") as fobj:
        for key in ["energy", "tdos"]:
            np.testing.assert_allclose(tdos_data[key], fobj[key][:], atol=1.0e-5)


@pytest.mark.parametrize(
    "system,soc,load_raw",
    [("Cs3Sb_soc", False, False), ("Cs3Sb_soc", True, True), ("g_qantum_number", False, True)],
)
def test_read_fhiaims_proj_dos(nested_dict_comparison, system, soc, load_raw):
    """Test read_fhiaims_proj_dos function."""
    pdos_data = read_fhiaims_proj_dos(PDOS_PATH + system + "/", soc=soc, load_raw=load_raw)
    ref_label = "/ref"
    if soc:
        ref_label += "_soc"
    if load_raw:
        ref_label += "_raw"
    pdos_ref = dict(read_yaml_file(PDOS_PATH + system + ref_label + ".yaml"))
    nested_dict_comparison(pdos_data, pdos_ref)
