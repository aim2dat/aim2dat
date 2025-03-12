"""Test for the work chain CellOptWorkChain."""

# Standard library imports
import os


# Third party library imports
import pytest
from aiida.plugins import WorkflowFactory
import aiida.orm as aiida_orm
from aiida.engine import run_get_node

# Internal library imports
from aim2dat.io import read_yaml_file


BandStructureWC = WorkflowFactory("aim2dat.cp2k.band_structure")

THIS_DIR = os.path.dirname(__file__)


@pytest.mark.skip
def test_band_structure_incl_seekpath(aiida_local_code_factory, aiida_create_structuredata):
    """Test band structure calculation including seekpath to generate the k-path."""
    code = aiida_local_code_factory("cp2k", "/usr/bin/cp2k.ssmp")

    # Load input parameters:
    parameters_si = read_yaml_file(THIS_DIR + "/cp2k/test_systems/Si_crystal.yaml")
    structure = aiida_create_structuredata(parameters_si["structure"])
    parameters = aiida_orm.Dict(dict=parameters_si["input_parameters"])

    results, node = run_get_node(
        BandStructureWC,
        structural_p=dict(
            structure=structure,
        ),
        numerical_p=dict(kpoints_ref_dist=aiida_orm.Float(0.5)),
        seekpath_parameters=aiida_orm.Dict(dict={"reference_distance": 0.015, "symprec": 0.005}),
        cp2k=dict(
            code=code,
            parameters=parameters,
            metadata={"options": {"resources": {"num_machines": 1}}},
        ),
    )

    # Check whether process returns the right exit-status:
    assert node.exit_status == 0
    assert "scf_parameters" in results
    assert "output_bands" in results["cp2k"]
    seekpath_nodes = ["primitive_structure", "path_parameters"]
    assert all([node in results["seekpath"] for node in seekpath_nodes])
    # TO-DO: Further checks of input parameters etc.


@pytest.mark.skip
def test_band_structure_without_seekpath(aiida_local_code_factory, aiida_create_structuredata):
    """Test band structure calculation without using seekpath."""
    code = aiida_local_code_factory("cp2k", "/usr/bin/cp2k.ssmp")

    # Load input parameters:
    parameters_si = read_yaml_file(THIS_DIR + "/cp2k/test_systems/Si_crystal.yaml")
    structure = aiida_create_structuredata(parameters_si["structure"])
    parameters = aiida_orm.Dict(dict=parameters_si["input_parameters"])
    path_parameters = aiida_orm.Dict(
        dict={
            "point_coords": {
                "K": [0.375, 0.375, 0.75],
                "L": [0.5, 0.5, 0.5],
                "U": [0.625, 0.25, 0.625],
                "W": [0.5, 0.25, 0.75],
                "X": [0.5, 0.0, 0.5],
                "W_2": [0.75, 0.25, 0.5],
                "GAMMA": [0.0, 0.0, 0.0],
            },
            "explicit_segments": [
                [0, 77],
                [76, 103],
                [103, 185],
                [184, 251],
                [250, 305],
                [304, 342],
            ],
            "path": [
                ["GAMMA", "X"],
                ["X", "U"],
                ["K", "GAMMA"],
                ["GAMMA", "L"],
                ["L", "W"],
                ["W", "X"],
            ],
        }
    )

    results, node = run_get_node(
        BandStructureWC,
        structural_p=dict(
            structure=structure,
        ),
        path_parameters=path_parameters,
        cp2k=dict(
            code=code,
            parameters=parameters,
            metadata={"options": {"resources": {"num_machines": 1}}},
        ),
    )

    # Check whether process returns the right exit-status:
    assert node.exit_status == 0
    assert "scf_parameters" in results
    assert "output_bands" in results["cp2k"]
    # TO-DO: Further checks of input parameters etc.
