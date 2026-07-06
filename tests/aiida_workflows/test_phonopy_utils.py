"""Tests for the phonopy_utils calcfunctions (finite-displacement phonon pipeline)."""

# Standard library imports
import os

# Third party library imports
import numpy as np
import pytest
import aiida.orm as aiida_orm
from aiida.plugins import DataFactory

# Internal library imports
from aim2dat.io import read_yaml_file
from aim2dat.aiida_workflows.utils import (
    phonopy_generate_displacements,
    phonopy_calculate_phonons,
)

TEST_SYSTEMS_PATH = os.path.dirname(__file__) + "/cp2k/test_systems/"

StructureData = DataFactory("core.structure")

# Minimal Γ→X band path for Si (FCC reciprocal space)
_SI_BAND_PATH = [[[0.0, 0.0, 0.0], [0.5, 0.0, 0.5]]]
_SI_BAND_LABELS = ["GAMMA", "X"]

_SI_CRYSTAL_DICT = dict(read_yaml_file(TEST_SYSTEMS_PATH + "Si_crystal.yaml"))["structure"]

_FORCE_SETS = np.array(
    [
        [
            [-3.21128624e-05, -1.86482524e-03, -1.86482524e-03],
            [-9.26690217e-07, 4.97394976e-05, 4.97394976e-05],
            [1.07862867e-04, -2.88768204e-04, -5.98669894e-05],
            [-1.12051502e-04, -6.04458325e-05, -2.86075835e-04],
            [1.07862867e-04, -5.98669894e-05, -2.88768204e-04],
            [-1.12051502e-04, -2.86075835e-04, -6.04458325e-05],
            [3.18110230e-06, -1.70083919e-04, -1.70083919e-04],
            [7.20316864e-07, 6.78304445e-05, 6.78304445e-05],
            [6.45820970e-04, 9.75747267e-04, 9.75747267e-04],
            [-6.09691676e-04, 9.48714428e-04, 9.48714428e-04],
            [-4.66818606e-07, 3.35606091e-04, 3.33439748e-04],
            [3.08734535e-07, -6.01494978e-05, -5.92009167e-05],
            [-4.66818606e-07, 3.33439748e-04, 3.35606091e-04],
            [3.08734533e-07, -5.92009167e-05, -6.01494978e-05],
            [-1.28001421e-04, 6.87889864e-05, 6.87889864e-05],
            [1.29656762e-04, 6.95567393e-05, 6.95567393e-05],
        ]
    ]
)


# --------------------------------------------------------------------------- #
# Test 1 — phonopy_generate_displacements
# --------------------------------------------------------------------------- #
@pytest.mark.aiida
def test_phonopy_generate_displacements(aiida_create_structuredata):
    """Displaced supercells and setting info are generated correctly for Si."""
    si = aiida_create_structuredata(_SI_CRYSTAL_DICT)  # 2-atom FCC primitive cell

    phonopy_parameters = aiida_orm.Dict(
        dict={
            "supercell_matrix": [[2, 0, 0], [0, 2, 0], [0, 0, 2]],
            "symprec": 1e-5,
        }
    )

    parameters = aiida_orm.Dict(
        dict={
            "displacement": 0.01,
        }
    )

    outputs = phonopy_generate_displacements(si, phonopy_parameters, parameters)

    # Check Si has one symmetry-reduced displacement in a 2×2×2 supercell
    assert "supercell_0000" in outputs
    assert "supercell_0001" not in outputs

    supercell = outputs["supercell_0000"]
    assert isinstance(supercell, StructureData)
    assert len(supercell.get_ase()) == 16  # 2 atoms × 2×2×2

    # Check outputs info carries displacement_dataset
    assert "displacement_dataset" in outputs


# --------------------------------------------------------------------------- #
# Test 2 — phonopy_collect_phonons  (band structure + DOS, no thermal)
# --------------------------------------------------------------------------- #
@pytest.mark.aiida
def test_phonopy_calculate_phonons_bands_dos(aiida_create_structuredata):
    """Band structure and total DOS are assembled from force constants."""
    si = aiida_create_structuredata(_SI_CRYSTAL_DICT)

    phonopy_parameters = aiida_orm.Dict(
        dict={
            "supercell_matrix": [[2, 0, 0], [0, 2, 0], [0, 0, 2]],
            "symprec": 1e-5,
        }
    )

    parameters = aiida_orm.Dict(
        dict={
            "displacement": 0.01,
        }
    )

    disp_outputs = phonopy_generate_displacements(si, phonopy_parameters, parameters)
    displacement_dataset = disp_outputs["displacement_dataset"].get_dict()

    parameters = aiida_orm.Dict(
        dict={
            "displacement_dataset": displacement_dataset,
            "force_sets": _FORCE_SETS,
            "band_path": _SI_BAND_PATH,
            "band_labels": _SI_BAND_LABELS,
            "band_npoints": 11,
            "dos_mesh": [4, 4, 4],
            "thermal_properties": False,
        }
    )

    outputs = phonopy_calculate_phonons(si, phonopy_parameters, parameters)

    # Check outputs for correct values
    assert "band_structure" in outputs
    assert "total_dos" in outputs
    assert "thermal_properties" not in outputs

    # Confirm band structure outputs are correct
    bs = outputs["band_structure"]
    assert len(bs.get_bands()) == len(bs.get_kpoints())
    assert len(bs.get_bands()[0]) == 6  # 2 atoms × 3 modes

    # Confirm DOS units and shape correct
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

    phonopy_parameters = aiida_orm.Dict(
        dict={
            "supercell_matrix": [[2, 0, 0], [0, 2, 0], [0, 0, 2]],
            "symprec": 1e-5,
        }
    )

    parameters = aiida_orm.Dict(
        dict={
            "displacement": 0.01,
        }
    )

    disp_outputs = phonopy_generate_displacements(si, phonopy_parameters, parameters)
    displacement_dataset = disp_outputs["displacement_dataset"].get_dict()

    parameters = aiida_orm.Dict(
        dict={
            "displacement_dataset": displacement_dataset,
            "force_sets": _FORCE_SETS,
            "band_path": _SI_BAND_PATH,
            "band_labels": _SI_BAND_LABELS,
            "band_npoints": 11,
            "dos_mesh": [4, 4, 4],
            "thermal_properties": True,
            "temp_range": [0.0, 100.0, 50.0],
        }
    )

    outputs = phonopy_calculate_phonons(si, phonopy_parameters, parameters)

    # Confirm thermal properties in outputs
    assert "thermal_properties" in outputs

    # Confirm all thermal props are there
    x_label, x_vals, x_unit = outputs["thermal_properties"].get_x()
    assert x_unit == "K"
    assert len(x_vals) > 0
    free_energy, entropy, heat_capacity = outputs["thermal_properties"].get_y()
    assert len(free_energy[1]) == len(x_vals)
    assert "helmholtz free energies" in free_energy[0]
    assert "entropies" in entropy[0]
    assert "heat capacities" in heat_capacity[0]
    # t_min=0, t_max=100, t_step=50 → temperatures [0, 50, 100]
    assert len(x_vals) == 3
