"""Test the qe module of the io sub-package."""

# Standard library imports
import os

# Third party library imports
import pytest
import h5py
import numpy as np

# Internal library imports
from aim2dat.io.qe import (
    read_band_structure,
    read_total_density_of_states,
    read_atom_proj_density_of_states,
    read_xml,
)
from aim2dat.io.yaml import load_yaml_file

BAND_STRUCTURE_PATH = os.path.dirname(__file__) + "/qe_band_structure/"
TDOS_PATH = os.path.dirname(__file__) + "/qe_tdos/"
PDOS_PATH = os.path.dirname(__file__) + "/qe_pdos/"
XML_PATH = os.path.dirname(__file__) + "/qe_xml/"


def test_read_band_structure(nested_dict_comparison):
    """Test read_band_structure function."""
    bands_data = read_band_structure(BAND_STRUCTURE_PATH + "bands.dat")
    assert bands_data["unit_y"] == "eV"
    with h5py.File(BAND_STRUCTURE_PATH + "ref.h5", "r") as fobj:
        for key in ["kpoints", "bands"]:
            np.testing.assert_allclose(bands_data[key], fobj[key][:], atol=1.0e-5)


def test_read_total_density_of_states(nested_dict_comparison):
    """Test read_total_density_of_states function."""
    tdos_data = read_total_density_of_states(TDOS_PATH + "dos.dat")
    assert tdos_data["unit_x"] == "eV"
    assert tdos_data["e_fermi"] == 5.968
    with h5py.File(TDOS_PATH + "ref.h5", "r") as fobj:
        for key in ["energy", "tdos"]:
            np.testing.assert_allclose(tdos_data[key], fobj[key][:], atol=1.0e-5)


def test_read_atom_proj_density_of_states(nested_dict_comparison):
    """Test read_atom_proj_density_of_states function."""
    pdos_data = read_atom_proj_density_of_states(PDOS_PATH)
    assert pdos_data["unit_x"] == "eV"
    with h5py.File(PDOS_PATH + "ref.h5", "r") as fobj:
        np.testing.assert_allclose(pdos_data["energy"], fobj["energy"][:], atol=1.0e-5)
        for pdos in pdos_data["pdos"]:
            kind = pdos.pop("kind")
            for orb, pdos0 in pdos.items():
                label = kind + ":" + orb
                np.testing.assert_allclose(pdos0, fobj[label][:], atol=1.0e-5)


@pytest.mark.parametrize("system", ["SCF", "H2O"])
def test_read_xml(nested_dict_comparison, system):
    """Test read_xml function."""
    ref = load_yaml_file(XML_PATH + system + "_ref.yaml")
    outp_dict = read_xml(XML_PATH + system + ".xml", **ref["parameters"])
    ref["ref"]["input"]["atomic_species"] = {
        k: tuple(v) for k, v in ref["ref"]["input"]["atomic_species"].items()
    }
    for strct in ref["ref"]["structures"]:
        strct["positions"] = [tuple(v) for v in strct["positions"]]
        if "atomic_species" in strct["attributes"]:
            strct["attributes"]["atomic_species"] = {
                k: tuple(v) for k, v in strct["attributes"]["atomic_species"].items()
            }
        if "site_attributes" in strct:
            strct["site_attributes"]["forces"] = [
                tuple(v) for v in strct["site_attributes"]["forces"]
            ]
    nested_dict_comparison(outp_dict, ref["ref"])
