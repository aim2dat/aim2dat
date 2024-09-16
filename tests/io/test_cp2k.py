"""Test the cp2k module of the io sub-package."""

# Standard library imports
import os

# Third party library imports
import pytest

# Internal library imports
from aim2dat.io.yaml import load_yaml_file
from aim2dat.io.cp2k import (
    read_band_structure,
    read_atom_proj_density_of_states,
    read_optimized_structure,
)

cwd = os.path.dirname(__file__) + "/"
STRUCTURES_PATH = cwd + "cp2k_structures/"
BAND_STRUCTURE_PATH = cwd + "cp2k_band_structure/"
PDOS_PATH = cwd + "cp2k_pdos/"


@pytest.mark.parametrize("system", ["standard", "uks"])
def test_read_band_structure(system):
    """Test read_band_structure function."""
    bands_data = read_band_structure(BAND_STRUCTURE_PATH + system + "/bands.bs")
    bands_ref = dict(load_yaml_file(BAND_STRUCTURE_PATH + system + "/ref.yaml"))
    assert bands_data["unit_y"] == bands_ref["unit_y"], "Energy unit doesn't match."
    for kpt_idx, kpt0 in enumerate(bands_ref["kpoints"]):
        kpt1 = bands_data["kpoints"][kpt_idx]
        assert all(
            [abs(coord0 - coord1) < 1e-5 for coord0, coord1 in zip(kpt0, kpt1)]
        ), f"k-point with index {kpt_idx} doesn't match."
        if len(bands_ref["bands"]) == 2:
            for spin_idx in range(2):
                ev0 = bands_ref["bands"][spin_idx][kpt_idx]
                ev1 = bands_data["bands"][spin_idx][kpt_idx]
                assert all(
                    [abs(en0 - en1) < 1e-5 for en0, en1 in zip(ev0, ev1)]
                ), f"Energies on k-point with index {kpt_idx} for spin {spin_idx} don't match."
        else:
            ev0 = bands_ref["bands"][kpt_idx]
            ev1 = bands_data["bands"][kpt_idx]
            assert all(
                [abs(en0 - en1) < 1e-5 for en0, en1 in zip(ev0, ev1)]
            ), f"Energies on k-point with index {kpt_idx} don't match."


@pytest.mark.parametrize("system", ["standard", "uks", "lists"])
def test_read_atom_proj_density_of_states(system):
    """Test read_atom_proj_density_of_states function."""
    # Test empty folder
    with pytest.raises(ValueError) as error:
        read_atom_proj_density_of_states(cwd + "empty_folder/")
    assert str(error.value) == "No files with the correct naming scheme found."

    # Test different systems:
    pdos_data = read_atom_proj_density_of_states(PDOS_PATH + system + "/")
    pdos_ref = dict(load_yaml_file(PDOS_PATH + system + "/ref.yaml"))
    assert all(
        [abs(en0 - en1) < 1e-5 for en0, en1 in zip(pdos_data["energy"], pdos_ref["energy"])]
    ), "Energies don't match."
    assert abs(pdos_data["e_fermi"] - pdos_ref["e_fermi"]) < 1e-5, "Fermi energies don't match."
    assert pdos_data["unit_x"] == pdos_ref["unit_x"], "Energy unit doesn't match."
    for idx0, pdos0 in enumerate(pdos_ref["pdos"]):
        kind = pdos0.pop("kind")
        pdos1 = pdos_data["pdos"][idx0]
        assert kind == pdos1["kind"], f"Kind label doesn't match for pdos {idx0}"
        for orb_l0, orb_v0 in pdos0.items():
            orb_v1 = pdos1[orb_l0]
            assert all(
                [abs(val0 - val1) < 1e-5 for val0, val1 in zip(orb_v0, orb_v1)]
            ), f"Values don't match for kind {kind} and orbital {orb_l0}."


@pytest.mark.parametrize(
    "restart_file,reference_file",
    [
        ("/cell_opt/aiida-1.restart", "/cell_opt/ref.yaml"),
        (
            "/cell_opt_incomplete_numbers/aiida-1.restart",
            "/cell_opt_incomplete_numbers/ref.yaml",
        ),
        ("/md-nvt/aiida-1.restart", "/md-nvt/ref.yaml"),
    ],
)
def test_read_optimized_structure_single(restart_file, reference_file):
    """
    Test read_optimized_structure function for single calculations.
    """
    reference_values = list(load_yaml_file(STRUCTURES_PATH + reference_file))
    structure = read_optimized_structure(STRUCTURES_PATH + restart_file)
    for ref_value in reference_values:
        assert structure[ref_value[0]] == ref_value[1]


def test_read_optimized_structure_multiple():
    """Test read_optimized_structure function."""
    structures = read_optimized_structure(STRUCTURES_PATH + "multiple_structures/")
    ref_structures = dict(load_yaml_file(STRUCTURES_PATH + "multiple_structures/ref.yaml"))

    for str_label, str1 in ref_structures.items():
        for str0 in structures:
            if str0["label"] == str_label:
                break
        assert all(
            [pbc0 == pbc1 for pbc0, pbc1 in zip(str0["pbc"], str1["pbc"])]
        ), "Periodic boundary conditions don't match."
        for idx0, (cell0, cell1) in enumerate(zip(str0["cell"], str1["cell"])):
            assert all(
                [abs(coord0 - coord1) < 1e-5 for coord0, coord1 in zip(cell0, cell1)]
            ), f"Cell vectors {idx0} don't match."
        for idx0, (pos0, pos1) in enumerate(zip(str0["positions"], str1["positions"])):
            assert all(
                [abs(coord0 - coord1) < 1e-5 for coord0, coord1 in zip(pos0, pos1)]
            ), f"Positions {idx0} don't match."
        assert all(
            [sym0 == sym1 for sym0, sym1 in zip(str0["symbols"], str1["symbols"])]
        ), "Symbols don't match."
        assert all(
            [kind0 == kind1 for kind0, kind1 in zip(str0["kinds"], str1["kinds"])]
        ), "Kinds don't match."
