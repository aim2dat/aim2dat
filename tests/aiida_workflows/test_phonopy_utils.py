"""Tests for the phonopy_utils calcfunctions (finite-displacement phonon pipeline)."""

# Standard library imports
import io
import os

# Third party library imports
import numpy as np
import pytest
import aiida.orm as aiida_orm
from aiida.plugins import DataFactory

# Internal library imports
from aim2dat.io import read_yaml_file
from aim2dat.utils.phonopy_utils import (
    phonopy_generate_displacements,
    parse_cp2k_forces,
    phonopy_collect_phonons,
)

TEST_SYSTEMS_PATH = os.path.dirname(__file__) + "/cp2k/test_systems/"

StructureData = DataFactory("core.structure")

# Minimal Γ→X band path for Si (FCC reciprocal space)
_SI_BAND_PATH = [[[0.0, 0.0, 0.0], [0.5, 0.0, 0.5]]]
_SI_BAND_LABELS = ["GAMMA", "X"]

_SI_CRYSTAL_DICT = dict(
    read_yaml_file(TEST_SYSTEMS_PATH + "Si_crystal.yaml")
)["structure"]


def _make_cp2k_force_content(n_atoms, forces):
    """Return a minimal CP2K 2024-format force-output string for ``n_atoms`` atoms."""
    lines = [
        " ATOMIC FORCES in [a.u.]",
        "",
        " # Atom   Kind   Element          X              Y              Z",
    ]
    for idx, (fx, fy, fz) in enumerate(forces):
        lines.append(
            f"      {idx + 1}      1      Si    {fx:14.8f}   {fy:14.8f}   {fz:14.8f}"
        )
    return "\n".join(lines) + "\n"


def _si_folder_with_forces(forces):
    """Wrap ``forces`` (shape n_atoms×3) in a FolderData named 'aiida.out'."""
    content = _make_cp2k_force_content(len(forces), forces)
    folder = aiida_orm.FolderData()
    folder.put_object_from_filelike(io.StringIO(content), "aiida.out")
    return folder


# --------------------------------------------------------------------------- #
# Test 1 — phonopy_generate_displacements
# --------------------------------------------------------------------------- #
@pytest.mark.aiida
def test_phonopy_generate_displacements(aiida_create_structuredata):
    """Displaced supercells and setting info are generated correctly for Si."""
    si = aiida_create_structuredata(_SI_CRYSTAL_DICT)  # 2-atom FCC primitive cell

    parameters = aiida_orm.Dict(
        dict={
            "supercell_matrix": [[2, 0, 0], [0, 2, 0], [0, 0, 2]],
            "symprec": 1e-5,
            "displacement": 0.01,
        }
    )

    outputs = phonopy_generate_displacements(si, parameters)

    info = outputs["phonon_setting_info"].get_dict()

    # Check Si has one symmetry-reduced displacement in a 2×2×2 supercell
    assert info["number_of_displacements"] == 1
    assert "supercell_0000" in outputs
    assert "supercell_0001" not in outputs

    supercell = outputs["supercell_0000"]
    assert isinstance(supercell, StructureData)
    assert len(supercell.get_ase()) == 16  # 2 atoms × 2×2×2

    # Check setting info carries all keys forward to parse/collect steps
    assert info["supercell_n_atoms"] == 16
    assert info["supercell_matrix"] == [[2, 0, 0], [0, 2, 0], [0, 0, 2]]
    assert info["displacement"] == 0.01
    assert "displacement_dataset" in info


# --------------------------------------------------------------------------- #
# Test 2 — parse_cp2k_forces
# --------------------------------------------------------------------------- #
@pytest.mark.aiida
def test_parse_cp2k_forces(aiida_create_structuredata):
    """Forces are read from CP2K output files and returned as an ArrayData."""
    si = aiida_create_structuredata(_SI_CRYSTAL_DICT)

    parameters = aiida_orm.Dict(
        dict={
            "supercell_matrix": [[2, 0, 0], [0, 2, 0], [0, 0, 2]],
            "symprec": 1e-5,
            "displacement": 0.01,
        }
    )
    disp_outputs = phonopy_generate_displacements(si, parameters)
    phonon_setting_info = disp_outputs["phonon_setting_info"]

    n_atoms = 16
    # Synthetic restoring force on displaced atom; Newton's-3rd reaction on neighbours
    disp = phonon_setting_info.get_dict()["displacement_dataset"]["first_atoms"][0][
        "displacement"
    ]
    k = 5.0  # arbitrary spring constant [a.u.]
    forces = np.zeros((n_atoms, 3))
    forces[0] = [-k * d for d in disp]
    forces[1] = [k * d / 2 for d in disp]
    forces[2] = [k * d / 2 for d in disp]

    folder = _si_folder_with_forces(forces)
    outputs = parse_cp2k_forces(phonon_setting_info, supercell_0000=folder)

    #Check force sets output correct shape and finite 
    force_sets = outputs["force_sets"].get_array("force_sets")
    assert force_sets.shape == (1, n_atoms, 3)
    assert np.all(np.isfinite(force_sets))


# --------------------------------------------------------------------------- #
# Test 3 — phonopy_collect_phonons  (band structure + DOS, no thermal)
# --------------------------------------------------------------------------- #
@pytest.mark.aiida
def test_phonopy_collect_phonons_bands_dos(aiida_create_structuredata):
    """Band structure and total DOS are assembled from force constants."""
    si = aiida_create_structuredata(_SI_CRYSTAL_DICT)

    parameters = aiida_orm.Dict(
        dict={
            "supercell_matrix": [[2, 0, 0], [0, 2, 0], [0, 0, 2]],
            "symprec": 1e-5,
            "displacement": 0.01,
        }
    )
    disp_outputs = phonopy_generate_displacements(si, parameters)
    phonon_setting_info = disp_outputs["phonon_setting_info"]

    disp = phonon_setting_info.get_dict()["displacement_dataset"]["first_atoms"][0][
        "displacement"
    ]
    k = 5.0
    forces = np.zeros((16, 3))
    forces[0] = [-k * d for d in disp]
    forces[1] = [k * d / 2 for d in disp]
    forces[2] = [k * d / 2 for d in disp]

    force_outputs = parse_cp2k_forces(
        phonon_setting_info, supercell_0000=_si_folder_with_forces(forces)
    )

    collect_params = aiida_orm.Dict(
        dict={
            "band_path": _SI_BAND_PATH,
            "band_labels": _SI_BAND_LABELS,
            "band_npoints": 11,
            "dos_mesh": [4, 4, 4],
            "thermal_properties": False,
        }
    )

    outputs = phonopy_collect_phonons(
        si, phonon_setting_info, force_outputs["force_sets"], collect_params
    )

    #Check outputs for correct values
    assert "band_structure" in outputs
    assert "total_dos" in outputs
    assert "thermal_properties" not in outputs

    #Confirm band structure outputs are correct
    bs = outputs["band_structure"].get_dict()
    assert "distances" in bs
    assert "frequencies" in bs
    assert len(bs["distances"]) == 1          # one path segment
    assert len(bs["distances"][0]) == 11      # band_npoints
    assert len(bs["frequencies"][0][0]) == 6  # 2 atoms × 3 modes

    #Confirm DOS units and shape correct
    x_label, x_vals, x_unit = outputs["total_dos"].get_x()
    assert x_unit == "THz"
    assert len(x_vals) > 0
    y_vals = outputs["total_dos"].get_y()[0][1]
    assert len(y_vals) == len(x_vals)


# --------------------------------------------------------------------------- #
# Test 4 — phonopy_collect_phonons  (with thermal properties)
# --------------------------------------------------------------------------- #
@pytest.mark.aiida
def test_phonopy_collect_phonons_thermal(aiida_create_structuredata):
    """Thermal properties dict is included when thermal_properties=True."""
    si = aiida_create_structuredata(_SI_CRYSTAL_DICT)

    parameters = aiida_orm.Dict(
        dict={
            "supercell_matrix": [[2, 0, 0], [0, 2, 0], [0, 0, 2]],
            "symprec": 1e-5,
            "displacement": 0.01,
        }
    )
    disp_outputs = phonopy_generate_displacements(si, parameters)
    phonon_setting_info = disp_outputs["phonon_setting_info"]

    disp = phonon_setting_info.get_dict()["displacement_dataset"]["first_atoms"][0][
        "displacement"
    ]
    k = 5.0
    forces = np.zeros((16, 3))
    forces[0] = [-k * d for d in disp]
    forces[1] = [k * d / 2 for d in disp]
    forces[2] = [k * d / 2 for d in disp]

    force_outputs = parse_cp2k_forces(
        phonon_setting_info, supercell_0000=_si_folder_with_forces(forces)
    )

    collect_params = aiida_orm.Dict(
        dict={
            "band_path": _SI_BAND_PATH,
            "band_labels": _SI_BAND_LABELS,
            "band_npoints": 11,
            "dos_mesh": [4, 4, 4],
            "thermal_properties": True,
            "t_min": 0.0,
            "t_max": 100.0,
            "t_step": 50.0,
        }
    )

    outputs = phonopy_collect_phonons(
        si, phonon_setting_info, force_outputs["force_sets"], collect_params
    )

    #Confirm thermal properties in outputs
    assert "thermal_properties" in outputs

    #Confirm all thermal props are there 
    tp = outputs["thermal_properties"].get_dict()
    assert "temperatures" in tp
    assert "free_energy" in tp
    assert "entropy" in tp
    assert "heat_capacity" in tp
    # t_min=0, t_max=100, t_step=50 → temperatures [0, 50, 100]
    assert len(tp["temperatures"]) == 3
