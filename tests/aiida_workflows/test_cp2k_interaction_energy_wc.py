"""Test for the work chain InteractionEnergyWorkChain."""

# Standard library imports
import os


# Third party library imports
import pytest
from aiida.plugins import WorkflowFactory
import aiida.orm as aiida_orm
from aiida.engine import run_get_node

# Internal library imports
from aim2dat.io import read_yaml_file


InteractionEnergyWC = WorkflowFactory("aim2dat.cp2k.interaction_energy")

THIS_DIR = os.path.dirname(__file__)


@pytest.mark.aiida
def test_interaction_energies(aiida_local_code_factory, aiida_create_structuredata):
    """Test interaction energy workchain with generating ghost atoms."""
    code = aiida_local_code_factory("cp2k", "/usr/bin/cp2k.ssmp")

    # Load input parameters:
    parameters_si = read_yaml_file(THIS_DIR + "/cp2k/test_systems/graphene_H2.yaml")
    structure = aiida_create_structuredata(parameters_si["structure"])
    parameters = aiida_orm.Dict(dict=parameters_si["input_parameters"])

    results, node = run_get_node(
        InteractionEnergyWC,
        structural_p=dict(
            structure=structure,
        ),
        cp2k=dict(
            code=code,
            parameters=parameters,
            metadata={"options": {"resources": {"num_machines": 1}}},
        ),
        fragments=aiida_orm.List([4, 5]),
    )

    # Check whether process returns the right exit-status:
    assert node.exit_status == 0
    assert "scf_parameters" in results
    assert "interaction_energy" in results["cp2k"]
    ghosts = [
        kind
        for kind in dict(node.inputs.parameters)["FORCE_EVAL"]["SUBSYS"]["KIND"]
        if "ghost" in kind["_"]
    ]
    kinds = [
        kind
        for kind in dict(node.inputs.parameters)["FORCE_EVAL"]["SUBSYS"]["KIND"]
        if "ghost" not in kind["_"]
    ]
    ghosts.sort(key=lambda x: x["_"])
    kinds.sort(key=lambda x: x["_"])
    for kind, ghost in zip(kinds, ghosts):
        assert kind["_"] == ghost["_"].replace("_ghost", "")
        assert kind["BASIS_SET"] == ghost["BASIS_SET"]
        assert ghost["GHOST"]
