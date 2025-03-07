"""Test the phonopy module of the io sub-package."""

# Standard library imports
import os

# Third party library imports
import pytest

# Internal library imports
from aim2dat.io import load_yaml_file
from aim2dat.io.phonopy import (
    read_band_structure,
    read_total_density_of_states,
    read_atom_proj_density_of_states,
    read_thermal_properties,
    read_qha_properties,
)


cwd = os.path.dirname(__file__) + "/"


def test_read_band_structure(nested_dict_comparison):
    """Test read_band_structure function."""
    bands_ref = dict(load_yaml_file(cwd + "phonopy_ha/band_structure_ref.yaml"))
    for idx, label in enumerate(bands_ref["bands"]["path_labels"]):
        bands_ref["bands"]["path_labels"][idx] = tuple(label)
    bands, cell = read_band_structure(
        cwd + "phonopy_ha/phonopy_disp.yaml",
        [[[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.5, 0.5, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 0.0]]],
        15,
        force_sets_file_name=cwd + "phonopy_ha/FORCE_SETS",
        path_labels=["GAMMA", "X", "S", "Y", "GAMMA"],
    )
    nested_dict_comparison(bands, bands_ref["bands"], threshold=1e-3)
    for dir0, dir0_ref in zip(cell, bands_ref["cell"]):
        for val, val_ref in zip(dir0, dir0_ref):
            assert abs(val - val_ref) < 1e-5, "Reference cell is wrong."


def test_read_total_density_of_states(nested_dict_comparison):
    """Test read_total_density_of_states function."""
    tdos_ref = dict(load_yaml_file(cwd + "phonopy_ha/tdos_ref.yaml"))
    tdos = read_total_density_of_states(
        cwd + "phonopy_ha/phonopy_disp.yaml",
        force_sets_file_name=cwd + "phonopy_ha/FORCE_SETS",
        mesh=20,
    )
    nested_dict_comparison(tdos, tdos_ref, threshold=1e-3)


def test_read_atom_proj_density_of_states(nested_dict_comparison):
    """Test read_atom_proj_density_of_states function."""
    pdos_ref = dict(load_yaml_file(cwd + "phonopy_ha/pdos_ref.yaml"))
    pdos = read_atom_proj_density_of_states(
        cwd + "phonopy_ha/phonopy_disp.yaml",
        force_sets_file_name=cwd + "phonopy_ha/FORCE_SETS",
        mesh=[5, 5, 5],
    )
    nested_dict_comparison(pdos, pdos_ref, threshold=1e-3)


@pytest.mark.skip
def test_read_thermal_properties(nested_dict_comparison):
    """Test read_thermal_properties function."""
    thermal_properties_ref = dict(load_yaml_file(cwd + "phonopy_ha/thermal_properties_ref.yaml"))
    thermal_properties = read_thermal_properties(
        cwd + "phonopy_ha/phonopy_disp.yaml",
        force_sets_file_name=cwd + "phonopy_ha/FORCE_SETS",
        mesh=20,
        t_max=500.0,
        t_step=50.0,
    )
    nested_dict_comparison(thermal_properties, thermal_properties_ref, threshold=1e-3)


@pytest.mark.skip
def test_read_qha_properties(nested_dict_comparison):
    """Test read_qha_properties function."""
    qha_properties_ref = dict(load_yaml_file(cwd + "phonopy_qha/qha_properties_ref.yaml"))
    qha_properties = read_qha_properties(
        calculation_folders=[cwd + f"phonopy_qha/energy-{idx0}" for idx0 in range(-5, 6)],
        ev_file_name=cwd + "phonopy_qha/e-v.dat",
        mesh=[5, 5, 5],
        t_max=550.0,
        t_step=50.0,
    )
    nested_dict_comparison(qha_properties, qha_properties_ref, threshold=1e-3)
    qha_properties_ref = dict(load_yaml_file(cwd + "phonopy_qha/qha_properties_tp_ref.yaml"))
    qha_properties = read_qha_properties(
        thermal_properties_file_names=[
            cwd + f"phonopy_qha/thermal_properties.yaml-{idx0}" for idx0 in range(-5, 6)
        ],
        ev_file_name=cwd + "phonopy_qha/e-v.dat",
    )
    nested_dict_comparison(qha_properties, qha_properties_ref, threshold=1e-3)
