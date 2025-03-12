"""Test for the work chain CellOptWorkChain."""

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


CellOptWC = WorkflowFactory("aim2dat.cp2k.cell_opt")

THIS_DIR = os.path.dirname(__file__)


@pytest.mark.skip
def test_cell_opt_fail(aiida_local_code_factory, aiida_create_structuredata):
    """Test failed unit cell optimization."""
    code = aiida_local_code_factory("cp2k", "/usr/bin/cp2k.ssmp")

    # Load input parameters:
    parameters_si = read_yaml_file(THIS_DIR + "/cp2k/test_systems/Si_crystal.yaml")
    scf_m_custom_p_list = read_yaml_file(
        THIS_DIR + "/cp2k/scf_method_parameters/custom_mixing_parameters_failed.yaml"
    )
    c_opt_custom_p_list = read_yaml_file(
        THIS_DIR + "/cp2k/cell_opt_parameters/custom_parameters_failed.yaml"
    )
    scf_m_custom_p_list[0]["parameters"][1]["MAX_SCF"] = 50

    # Load input parameters:
    structure = aiida_create_structuredata(parameters_si["structure"])
    parameters = aiida_orm.Dict(dict=parameters_si["input_parameters"])
    scf_method_custom_parameters = aiida_orm.List(list=scf_m_custom_p_list)
    opt_method_custom_parameters = aiida_orm.List(list=c_opt_custom_p_list)

    # Initial SCF-parameters:
    i_scf_p = {
        "parameters": scf_m_custom_p_list[0]["parameters"][0],
        "smearing_level": 1,
        "method_level": 0,
        "parameter_level": 0,
        "uks": False,
        "roks": False,
    }
    initial_scf_parameters = aiida_orm.Dict(dict=i_scf_p)

    results, node = run_get_node(
        CellOptWC,
        structural_p=dict(
            structure=structure,
            scf_parameters=initial_scf_parameters,
        ),
        custom_scf_method=scf_method_custom_parameters,
        custom_opt_method=opt_method_custom_parameters,
        adjust_scf_parameters=aiida_orm.Bool(True),
        numerical_p=dict(kpoints_ref_dist=aiida_orm.Float(0.5)),
        cp2k=dict(
            code=code,
            parameters=parameters,
            metadata={"options": {"resources": {"num_machines": 1}}},
        ),
    )

    # Check whether process returns the right exit-status:
    assert node.exit_status == 612

    # Check called processes:
    tuple_counter = 0
    scf_level_tuples = [(0, 0, 250.0), (0, 0, 250.0), (0, 1, 250.0), (0, 1, 250.0), (0, 1, 250.0)]
    c_opt_trust_r = [0.001, 0.001, 0.001, 0.0001, 0.35]
    for process in node.called:
        if isinstance(process, cjob.CalcJobNode):
            cur_tuple = scf_level_tuples[tuple_counter]
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
            assert (
                input_p["MOTION"]["CELL_OPT"]["BFGS"]["TRUST_RADIUS"]
                == c_opt_trust_r[tuple_counter]
            )
            tuple_counter += 1
    # Check if all levels are reached:
    assert tuple_counter == len(scf_level_tuples)

    # Check resutls:
    scf_parameters = {
        "uks": False,
        "roks": False,
        "parameters": {
            "MIXING": {"BETA": 0.5, "ALPHA": 0.2, "METHOD": "BROYDEN_MIXING", "NBUFFER": 4},
            "MAX_SCF": 50,
        },
        "method_level": 0,
        "parameter_level": 1,
        "smearing_level": 1,
        "added_mos": 25,
        "smearing_temperature": 250.0,
        "odd_kpoints": False,
    }
    assert results["scf_parameters"].get_dict() == scf_parameters


@pytest.mark.skip
def test_cell_opt_sucess(aiida_local_code_factory, aiida_create_structuredata):
    """Test successful unit cell optimization."""
    code = aiida_local_code_factory("cp2k", "/usr/bin/cp2k.ssmp")

    # Load input parameters:
    parameters_si = read_yaml_file(THIS_DIR + "/cp2k/test_systems/Si_crystal.yaml")
    scf_m_custom_p_list = read_yaml_file(
        THIS_DIR + "/cp2k/scf_method_parameters/custom_mixing_parameters_failed.yaml"
    )
    c_opt_custom_p_list = read_yaml_file(
        THIS_DIR + "/cp2k/cell_opt_parameters/custom_parameters_failed.yaml"
    )
    scf_m_custom_p_list[1]["parameters"].append(
        {
            "MIXING": {"BETA": 0.5, "ALPHA": 0.35, "METHOD": "BROYDEN_MIXING", "NBUFFER": 4},
            "MAX_SCF": 25,
        }
    )
    c_opt_custom_p_list[2]["MAX_ITER"] = 50
    parameters_si["input_parameters"]["MOTION"] = {
        "CELL_OPT": {
            "MAX_DR": 0.2,
            "MAX_FORCE": 0.05,
            "RMS_DR": 0.1,
            "RMS_FORCE": 0.025,
            "PRESSURE_TOLERANCE": 5000.0,
        }
    }

    # Load input parameters:
    structure = aiida_create_structuredata(parameters_si["structure"])
    parameters = aiida_orm.Dict(dict=parameters_si["input_parameters"])
    scf_method_custom_parameters = aiida_orm.List(list=scf_m_custom_p_list)
    opt_method_custom_parameters = aiida_orm.List(list=c_opt_custom_p_list)

    # Initial SCF-parameters:
    i_scf_p = {
        "parameters": scf_m_custom_p_list[1]["parameters"][1],
        "smearing_level": 0,
        "method_level": 1,
        "parameter_level": 1,
        "uks": False,
        "roks": False,
    }
    initial_scf_parameters = aiida_orm.Dict(dict=i_scf_p)

    results, node = run_get_node(
        CellOptWC,
        structural_p=dict(
            structure=structure,
            scf_parameters=initial_scf_parameters,
        ),
        custom_scf_method=scf_method_custom_parameters,
        custom_opt_method=opt_method_custom_parameters,
        adjust_scf_parameters=aiida_orm.Bool(True),
        numerical_p=dict(kpoints_ref_dist=aiida_orm.Float(0.5)),
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
    scf_level_tuples = [(1, 1, 0.0), (1, 1, 0.0), (1, 2, 0.0), (1, 2, 0.0), (1, 2, 0.0)]
    c_opt_trust_r = [0.001, 0.001, 0.001, 0.0001, 0.35]
    for process in node.called:
        if isinstance(process, cjob.CalcJobNode):
            cur_tuple = scf_level_tuples[tuple_counter]
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
            assert (
                input_p["MOTION"]["CELL_OPT"]["BFGS"]["TRUST_RADIUS"]
                == c_opt_trust_r[tuple_counter]
            )
            tuple_counter += 1
    # Check if all levels are reached:
    assert tuple_counter == len(scf_level_tuples)

    # Check resutls:
    scf_parameters = {
        "uks": False,
        "roks": False,
        "parameters": {
            "MIXING": {"BETA": 0.5, "ALPHA": 0.35, "METHOD": "BROYDEN_MIXING", "NBUFFER": 4},
            "MAX_SCF": 25,
        },
        "method_level": 1,
        "smearing_level": 0,
        "parameter_level": 2,
        "added_mos": 0,
        "smearing_temperature": 0.0,
        "odd_kpoints": False,
    }
    assert results["scf_parameters"].get_dict() == scf_parameters
