"""Test utils functions for CP2K combined work chains."""

# Standard library imports
import os

# Third party library imports
import pytest
import aiida.orm as aiida_orm
from aiida.common.extendeddicts import AttributeDict

# Internal library imports
from aim2dat.aiida_workflows.cp2k.surface_opt_utils import (
    surfopt_setup,
    surfopt_should_run_slab_conv,
    surfopt_should_run_add_calc,
)
from aim2dat.aiida_workflows.utils import obtain_value_from_aiida_node
from aim2dat.aiida_workflows.cp2k.el_properties_utils import elprop_setup
from aim2dat.io import read_yaml_file
from aim2dat.strct import Structure, SurfaceGeneration

STRUCTURES_PATH = os.path.dirname(__file__) + "/../strct/structures/"
REF_PATH = os.path.dirname(__file__) + "/cp2k/combined_wc_utils/"


@pytest.mark.aiida
def test_surfopt(nested_dict_comparison, structure_comparison):
    """Test surface opt utils functions."""

    def _parse_to_aiida_dict(value):
        return aiida_orm.Dict(dict={"energy": value})

    def _parse_to_aiida_float(value):
        return aiida_orm.Float(value.get_dict()["energy"])

    inputs_dict, ref = read_yaml_file(REF_PATH + "surfopt_ref.yaml")
    inputs_dict["slab_conv"]["criteria"] = aiida_orm.Str(inputs_dict["slab_conv"]["criteria"])
    inputs_dict["slab_conv"]["threshold"] = aiida_orm.Float(inputs_dict["slab_conv"]["threshold"])
    inputs_dict["structural_p"]["minimum_slab_size"] = aiida_orm.Float(
        inputs_dict["structural_p"]["minimum_slab_size"]
    )
    inputs_dict["structural_p"]["maximum_slab_size"] = aiida_orm.Float(
        inputs_dict["structural_p"]["maximum_slab_size"]
    )

    surf_gen = SurfaceGeneration(
        Structure(**read_yaml_file(STRUCTURES_PATH + "Cs2Te_62_prim.yaml"))
    )
    inputs_dict["structural_p"]["surface"] = surf_gen.to_aiida_surfacedata((1, 0, 0), 1, 0.005)

    assert surfopt_setup(AttributeDict(), AttributeDict(inputs_dict)) == "ERROR_INPUT_WRONG_VALUE"

    surface_slab_ref = ref["ctx1"].pop("surface_slab")
    for parse_f in [_parse_to_aiida_dict, _parse_to_aiida_float]:
        for key, val in inputs_dict["bulk_reference"].items():
            inputs_dict["bulk_reference"][key] = parse_f(val)
        inputs = AttributeDict(inputs_dict)
        ctx = AttributeDict()
        assert surfopt_setup(ctx, inputs) is None

        surface_slab = ctx.pop("surface_slab")
        structure_comparison(Structure.from_aiida_structuredata(surface_slab), surface_slab_ref)

        ctx.initial_opt_parameters = ctx.initial_opt_parameters.value
        ctx.slab_parameters = ctx.slab_parameters.get_dict()
        nested_dict_comparison(ctx, ref["ctx1"])
        ctx.slab_parameters = aiida_orm.Dict(dict=ctx.slab_parameters)

    ctx.surface_slab = surface_slab
    assert surfopt_should_run_slab_conv(ctx, inputs) == (True, [])

    # Check first geo_opt run:
    ctx.find_scf_p = AttributeDict(
        {"outputs": {"cp2k": {"output_parameters": {"energy": -300.0}}}}
    )
    ctx.geo_opt = AttributeDict({"outputs": {"cp2k": {"output_parameters": {"energy": -400.0}}}})
    should_run, reports = surfopt_should_run_slab_conv(ctx, inputs)
    assert should_run
    assert reports == ref["reports2"]
    surface_slab_ref = ref["ctx2"].pop("surface_slab")
    surface_slab = ctx.pop("surface_slab")
    structure_comparison(Structure.from_aiida_structuredata(surface_slab), surface_slab_ref)
    del ctx.find_scf_p
    del ctx.geo_opt
    ctx.slab_parameters = ctx.slab_parameters.get_dict()
    nested_dict_comparison(ctx, ref["ctx2"])
    ctx.slab_parameters = aiida_orm.Dict(dict=ctx.slab_parameters)
    ctx.surface_slab = surface_slab

    # Check converged geo_opt run:
    ctx.find_scf_p = AttributeDict(
        {"outputs": {"cp2k": {"output_parameters": {"energy": -300.0}}}}
    )
    ctx.geo_opt = AttributeDict({"outputs": {"cp2k": {"output_parameters": {"energy": -550.0}}}})
    should_run, reports = surfopt_should_run_slab_conv(ctx, inputs)
    assert not should_run
    assert reports == ref["reports3"]
    surface_slab_ref = ref["ctx3"].pop("surface_slab")
    surface_slab = ctx.pop("surface_slab")
    structure_comparison(Structure.from_aiida_structuredata(surface_slab), surface_slab_ref)
    del ctx.find_scf_p
    del ctx.geo_opt
    ctx.slab_parameters = ctx.slab_parameters.get_dict()
    nested_dict_comparison(ctx, ref["ctx3"])
    ctx.slab_parameters = aiida_orm.Dict(dict=ctx.slab_parameters)
    ctx.surface_slab = surface_slab

    # Check additional calc:
    assert surfopt_should_run_add_calc(ctx, inputs)
    surface_slab_ref = ref["ctx4"].pop("surface_slab")
    surface_slab = ctx.pop("surface_slab")
    structure_comparison(Structure.from_aiida_structuredata(surface_slab), surface_slab_ref)
    surface_slab_ref = ref["ctx4"].pop("prim_slab")
    surface_slab = ctx.pop("prim_slab")
    structure_comparison(Structure.from_aiida_structuredata(surface_slab), surface_slab_ref)
    ctx.slab_parameters = ctx.slab_parameters.get_dict()
    ctx.k_path_parameters = ctx.k_path_parameters.get_dict()
    nested_dict_comparison(ctx, ref["ctx4"])


def test_elprop(nested_dict_comparison):
    """Test utils functions for el prop work chain."""
    ref = read_yaml_file(REF_PATH + "elprop_ref.yaml")
    strct = Structure(**read_yaml_file(STRUCTURES_PATH + "Cs2Te_62_prim.yaml"))
    strct_node = strct.to_aiida_structuredata()

    inputs = AttributeDict(
        {
            "scf_extended_system": False,
            "structural_p": {"structure": strct_node},
            "numerical_p": {
                "xc_functional": aiida_orm.Str("PBE"),
                "cutoff_values": aiida_orm.Dict({"cutoff": 600.0, "rel_cutoff": 100.0}),
                "basis_sets": aiida_orm.Str("STANDARD-TZV2P"),
            },
            "cp2k": {"code": "test", "metadata": aiida_orm.Dict(dict={"test": "test"})},
            "critic2": {"code": "test", "metadata": aiida_orm.Dict(dict={"test": "test"})},
            "chargemol": {"code": "test", "metadata": aiida_orm.Dict(dict={"test": "test"})},
            "workflow": {
                "protocol": aiida_orm.Str("cp2k-crystal-standard"),
                "run_cell_optimization": aiida_orm.Bool(True),
                "calc_band_structure": aiida_orm.Bool(True),
                "calc_eigenvalues": aiida_orm.Bool(True),
                "calc_pdos": aiida_orm.Bool(True),
                "calc_partial_charges": aiida_orm.Bool(True),
            },
        }
    )
    ctx = AttributeDict()
    elprop_setup(ctx, AttributeDict(inputs))
    # ctx = ctx.get_dict()
    for task in [
        "find_scf_parameters",
        "unit_cell_opt",
        "band_structure",
        "eigenvalues",
        "pdos",
        "partial_charges",
    ]:
        for key, value in ctx[task].items():
            try:
                value = obtain_value_from_aiida_node(value)
            except ValueError:
                pass
            assert value == ref[task][key]
