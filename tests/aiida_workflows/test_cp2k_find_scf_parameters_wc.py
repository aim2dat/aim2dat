"""Test for the work chain FindSCFParametersWorkChain."""

# Standard library imports
import os

# Third party library imports
import pytest
from aiida.plugins import WorkflowFactory
import aiida.orm as aiida_orm
from aiida.engine import run_get_node
import aiida.orm.nodes.process.calculation.calcjob as cjob

# Internal library imports
from aim2dat.io import read_yaml_file


FindSCFParametersWC = WorkflowFactory("aim2dat.cp2k.find_scf_p")

THIS_DIR = os.path.dirname(__file__)


@pytest.mark.skip
def test_find_scf_parameters_fail(aiida_local_code_factory, aiida_create_structuredata):
    """Test no converging set of inputs."""
    code = aiida_local_code_factory("cp2k", "/usr/bin/cp2k.ssmp")

    # Load input parameters:
    parameters_h2o = read_yaml_file(THIS_DIR + "/cp2k/test_systems/H2O_molecule.yaml")
    scf_m_custom_p_list = read_yaml_file(
        THIS_DIR + "/cp2k/scf_method_parameters/custom_mixing_parameters_failed.yaml"
    )

    # Create input nodes:
    structure = aiida_create_structuredata(parameters_h2o["structure"])
    parameters = aiida_orm.Dict(dict=parameters_h2o["input_parameters"])
    custom_scf_method = aiida_orm.List(list=scf_m_custom_p_list)

    system_par = {
        "unkown": [
            (0, 0, 0.0),
            (1, 0, 0.0),
            (2, 0, 0.0),
            (0, 0, 250.0),
            (0, 0, 500.0),
            (0, 0, 1000.0),
            (0, 1, 0.0),
            (1, 1, 0.0),
            (2, 1, 0.0),
            (0, 1, 250.0),
            (0, 1, 500.0),
            (0, 1, 1000.0),
        ],
        "metallic": [
            (0, 0, 250.0),
            (0, 0, 500.0),
            (0, 0, 1000.0),
            (0, 1, 250.0),
            (0, 1, 500.0),
            (0, 1, 1000.0),
        ],
        "insulator": [
            (0, 0, 0.0),
            (1, 0, 0.0),
            (2, 0, 0.0),
            (0, 1, 0.0),
            (1, 1, 0.0),
            (2, 1, 0.0),
        ],
    }
    for sys_char, level_tuples in system_par.items():
        results, node = run_get_node(
            FindSCFParametersWC,
            structural_p=dict(
                structure=structure,
                system_character=aiida_orm.Str(sys_char),
            ),
            custom_scf_method=custom_scf_method,
            cp2k=dict(
                code=code,
                parameters=parameters,
                metadata={"options": {"resources": {"num_machines": 1}}},
            ),
        )

        # Check whether process returns the right exit-status:
        assert node.exit_status == 610

        # Check called processes:
        tuple_counter = 0
        for process in node.called:
            if isinstance(process, cjob.CalcJobNode):
                cur_tuple = level_tuples[tuple_counter]
                input_p = process.inputs["parameters"].get_dict()
                assert (
                    input_p["FORCE_EVAL"]["DFT"]["SCF"]["MIXING"]["ALPHA"]
                    == scf_m_custom_p_list[cur_tuple[0]]["parameters"][cur_tuple[1]]["MIXING"][
                        "ALPHA"
                    ]
                )
                if cur_tuple[2] == 0.0:
                    assert not input_p["FORCE_EVAL"]["DFT"]["SCF"]["SMEAR"]["_"]
                else:
                    assert (
                        input_p["FORCE_EVAL"]["DFT"]["SCF"]["SMEAR"]["ELECTRONIC_TEMPERATURE"]
                        == cur_tuple[2]
                    )
                tuple_counter += 1


@pytest.mark.skip
def test_find_scf_parameters_success(aiida_local_code_factory, aiida_create_structuredata):
    """Test successful converging set of inputs."""
    code = aiida_local_code_factory("cp2k", "/usr/bin/cp2k.ssmp")

    # Load input parameters:
    parameters_h2o = read_yaml_file(THIS_DIR + "/cp2k/test_systems/H2O_molecule.yaml")
    scf_m_custom_p_list = read_yaml_file(
        THIS_DIR + "/cp2k/scf_method_parameters/custom_mixing_parameters_failed.yaml"
    )
    scf_m_custom_p_list[0]["parameters"][1]["MAX_SCF"] = 50

    # Create input nodes:
    structure = aiida_create_structuredata(parameters_h2o["structure"])
    parameters = aiida_orm.Dict(dict=parameters_h2o["input_parameters"])
    custom_scf_method = aiida_orm.List(list=scf_m_custom_p_list)

    results, node = run_get_node(
        FindSCFParametersWC,
        structural_p=dict(
            structure=structure,
        ),
        custom_scf_method=custom_scf_method,
        cp2k=dict(
            code=code,
            parameters=parameters,
            metadata={"options": {"resources": {"num_machines": 1}}},
        ),
    )

    # Check whether process returns the right exit-status:
    assert node.exit_status == 0

    # Check called processes:
    tuple_counter = 0
    level_tuples = [
        (0, 0, 0.0),
        (1, 0, 0.0),
        (2, 0, 0.0),
        (0, 0, 250.0),
        (0, 0, 500.0),
        (0, 0, 1000.0),
        (0, 1, 0.0),
    ]
    for process in node.called:
        if isinstance(process, cjob.CalcJobNode):
            cur_tuple = level_tuples[tuple_counter]
            input_p = process.inputs["parameters"].get_dict()
            assert (
                input_p["FORCE_EVAL"]["DFT"]["SCF"]["MIXING"]["ALPHA"]
                == scf_m_custom_p_list[cur_tuple[0]]["parameters"][cur_tuple[1]]["MIXING"]["ALPHA"]
            )
            if cur_tuple[2] == 0.0:
                assert not input_p["FORCE_EVAL"]["DFT"]["SCF"]["SMEAR"]["_"]
            else:
                assert (
                    input_p["FORCE_EVAL"]["DFT"]["SCF"]["SMEAR"]["ELECTRONIC_TEMPERATURE"]
                    == cur_tuple[2]
                )
            tuple_counter += 1

    # Check resutls:
    scf_parameters = {
        "uks": False,
        "roks": False,
        "parameters": {
            "MIXING": {"BETA": 0.5, "ALPHA": 0.2, "METHOD": "BROYDEN_MIXING", "NBUFFER": 4},
            "MAX_SCF": 50,
        },
        "method_level": 0,
        "smearing_level": 0,
        "parameter_level": 1,
        "added_mos": 0,
        "smearing_temperature": 0.0,
    }
    assert results["scf_parameters"].get_dict() == scf_parameters
