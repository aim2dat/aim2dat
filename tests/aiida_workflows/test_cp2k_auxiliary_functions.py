"""Test auxiliary functions used by the CP2K work chains."""

# Standard library imports
import os

# Third party library imports
import pytest

# Internal library imports
from aim2dat.io import read_yaml_file
import aim2dat.aiida_workflows.cp2k.auxiliary_functions as aux_functions


MAIN_PATH = os.path.dirname(__file__) + "/cp2k/test_systems/"


@pytest.mark.parametrize(
    "test_system,kind_parameters,result",
    [
        ("Si_crystal", {}, 8),
        (
            "Si_crystal",
            {"KIND": [{"_": "Si", "BASIS_SET": "DZVP-MOLOPT-GTH", "POTENTIAL": "ALL"}]},
            28,
        ),
        (
            "H2O_molecule",
            {
                "KIND": [
                    {"_": "H", "ELEMENT": "H", "BASIS_SET": "DZVP-MOLOPT-GTH", "POTENTIAL": "ALL"},
                    {"_": "O", "BASIS_SET": "DZVP-MOLOPT-GTH-q6", "POTENTIAL": "GTH-PBE"},
                ]
            },
            8,
        ),
        (
            "H2O_molecule",
            {"KIND": [{"_": "H1", "BASIS_SET": "DZVP-MOLOPT-GTH", "POTENTIAL": "ALL"}]},
            30,
        ),
    ],
)
def test_calc_nr_explicit_electrons(
    aiida_profile, aiida_create_structuredata, test_system, kind_parameters, result
):
    """Test ``calc_nr_explicit_electrons``-function."""
    test_system_dict = read_yaml_file(MAIN_PATH + f"/{test_system}.yaml")
    structure = aiida_create_structuredata(test_system_dict["structure"])
    parameters = test_system_dict["input_parameters"]
    parameters["FORCE_EVAL"]["SUBSYS"].update(kind_parameters)
    assert result == aux_functions.calc_nr_explicit_electrons(structure, parameters)
