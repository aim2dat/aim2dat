"""Test the phonopy module of the io sub-package."""

# Standard library imports
import os

# Internal library imports
from aim2dat.io import (
    read_yaml_file,
    read_phonopy_band_structure,
    read_phonopy_total_dos,
    read_phonopy_proj_dos,
    read_phonopy_thermal_properties,
    read_phonopy_qha_properties,
)


cwd = os.path.dirname(__file__) + "/"


def test_read_phonopy_band_structure(nested_dict_comparison):
    """Test read_phonopy_band_structure function."""
    bands_ref = dict(read_yaml_file(cwd + "phonopy_ha/band_structure_ref.yaml"))
    for idx, label in enumerate(bands_ref["bands"]["path_labels"]):
        bands_ref["bands"]["path_labels"][idx] = tuple(label)
    bands, cell = read_phonopy_band_structure(
        cwd + "phonopy_ha/phonopy_disp.yaml",
        [[[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.5, 0.5, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 0.0]]],
        15,
        force_sets_file_path=cwd + "phonopy_ha/FORCE_SETS",
        path_labels=["GAMMA", "X", "S", "Y", "GAMMA"],
    )
    nested_dict_comparison(bands, bands_ref["bands"], threshold=1e-3)
    for dir0, dir0_ref in zip(cell, bands_ref["cell"]):
        for val, val_ref in zip(dir0, dir0_ref):
            assert abs(val - val_ref) < 1e-5, "Reference cell is wrong."


def test_read_total_dos(nested_dict_comparison):
    """Test read_phonopy_total_dos function."""
    tdos_ref = dict(read_yaml_file(cwd + "phonopy_ha/tdos_ref.yaml"))
    tdos = read_phonopy_total_dos(
        cwd + "phonopy_ha/phonopy_disp.yaml",
        force_sets_file_path=cwd + "phonopy_ha/FORCE_SETS",
        mesh=20,
    )
    nested_dict_comparison(tdos, tdos_ref, threshold=1e-3)


def test_read_phonopy_proj_dos(nested_dict_comparison):
    """Test read_phonopy_proj_dos function."""
    pdos_ref = dict(read_yaml_file(cwd + "phonopy_ha/pdos_ref.yaml"))
    pdos = read_phonopy_proj_dos(
        cwd + "phonopy_ha/phonopy_disp.yaml",
        force_sets_file_path=cwd + "phonopy_ha/FORCE_SETS",
        mesh=[5, 5, 5],
    )
    nested_dict_comparison(pdos, pdos_ref, threshold=1e-3)


def test_read_phonopy_thermal_properties(nested_dict_comparison):
    """Test read_phonopy_thermal_properties function."""
    thermal_properties_ref = dict(read_yaml_file(cwd + "phonopy_ha/thermal_properties_ref.yaml"))
    thermal_properties = read_phonopy_thermal_properties(
        cwd + "phonopy_ha/phonopy_disp.yaml",
        force_sets_file_path=cwd + "phonopy_ha/FORCE_SETS",
        mesh=20,
        t_max=500.0,
        t_step=50.0,
    )
    nested_dict_comparison(thermal_properties, thermal_properties_ref, threshold=1e-3)


def test_read_phonopy_qha_properties(nested_dict_comparison):
    """Test read_phonopy_qha_properties function."""
    qha_properties_ref = dict(read_yaml_file(cwd + "phonopy_qha/qha_properties_ref.yaml"))
    qha_properties = read_phonopy_qha_properties(
        calculation_folder_paths=[cwd + f"phonopy_qha/energy-{idx0}" for idx0 in range(-5, 6)],
        ev_file_path=cwd + "phonopy_qha/e-v.dat",
        mesh=[5, 5, 5],
        t_max=550.0,
        t_step=50.0,
    )
    nested_dict_comparison(qha_properties, qha_properties_ref, threshold=1.0)
    qha_properties_ref = dict(read_yaml_file(cwd + "phonopy_qha/qha_properties_tp_ref.yaml"))
    qha_properties = read_phonopy_qha_properties(
        thermal_properties_file_paths=[
            cwd + f"phonopy_qha/thermal_properties.yaml-{idx0}" for idx0 in range(-5, 6)
        ],
        ev_file_path=cwd + "phonopy_qha/e-v.dat",
    )
    nested_dict_comparison(qha_properties, qha_properties_ref, threshold=1.0)
