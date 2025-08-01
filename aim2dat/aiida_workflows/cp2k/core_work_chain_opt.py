"""Function for the optimization workflows."""

# Standard library imports
import os

# Third party library imports
import aiida.orm as aiida_orm

# Internal library imports
from aim2dat.io import read_yaml_file
from aim2dat.utils.dict_tools import (
    dict_set_parameter,
    dict_retrieve_parameter,
    dict_create_tree,
    dict_merge,
)


cwd = os.path.dirname(__file__)


def _initialize_opt_parameters(inputs, ctx, exit_codes, opt_type):
    ctx.opt_level = 0
    ctx.opt_iteration = 0
    ctx.opt_type = opt_type.upper()

    # Read optimization parameters
    if "custom_opt_method" in inputs:
        ctx.opt_method_p = inputs.custom_opt_method.get_list()
    else:
        file_path = cwd + f"/opt_parameter_files/{opt_type}_p.yaml"
        ctx.opt_method_p = read_yaml_file(file_path)

    if "initial_opt_parameters" in inputs and inputs.initial_opt_parameters.value < len(
        ctx.opt_method_p
    ):
        ctx.opt_level = inputs.initial_opt_parameters.value

    # Adjust parameters:
    parameters = ctx.inputs.parameters.get_dict()
    dict_merge(parameters["GLOBAL"], {"RUN_TYPE": ctx.opt_type})
    dict_create_tree(parameters, ["MOTION", ctx.opt_type])
    dict_merge(parameters["MOTION"][ctx.opt_type], ctx.opt_method_p[ctx.opt_level])
    if "ignore_convergence_failure" in inputs and inputs.ignore_convergence_failure.value:
        dict_set_parameter(
            parameters, ["FORCE_EVAL", "DFT", "SCF", "IGNORE_CONVERGENCE_FAILURE"], True
        )
    if "optimization_p" in inputs:
        inputs_opt_p = inputs.optimization_p
        for keyword in [
            "max_force",
            "max_dr",
            "rms_force",
            "rms_dr",
            "keep_space_group",
            "eps_symmetry",
        ]:
            if keyword in inputs_opt_p:
                dict_set_parameter(
                    parameters,
                    ["MOTION", ctx.opt_type, keyword.upper()],
                    inputs_opt_p[keyword].value,
                )
        if "fixed_atoms" in inputs_opt_p:
            atoms_list = inputs_opt_p["fixed_atoms"].get_list()
            atoms_list = [i + 1 for i in atoms_list]
            atoms_list_sorted = sorted(atoms_list)
            if len(atoms_list) > 4 and atoms_list_sorted[-1] - atoms_list_sorted[0] + 1 == len(
                atoms_list_sorted
            ):
                fixed_atoms = f"{atoms_list_sorted[0]}..{atoms_list_sorted[-1]}"
            else:
                atoms_string = [str(s) for s in atoms_list]
                fixed_atoms = " ".join(atoms_string)
            dict_set_parameter(
                parameters,
                ["MOTION", "CONSTRAINT", "FIXED_ATOMS", "LIST"],
                fixed_atoms,
            )
        if opt_type == "cell_opt":
            _set_additional_cell_opt_p(inputs.optimization_p, ctx, exit_codes, parameters)
    ctx.inputs.parameters = aiida_orm.Dict(dict=parameters)


def _set_additional_cell_opt_p(inputs_opt_p, ctx, exit_codes, parameters):
    # inputs_opt_p = inputs.optimization_p
    if (
        "keep_symmetry" in inputs_opt_p
        and inputs_opt_p.keep_symmetry.value
        and "cell_symmetry" not in inputs_opt_p
    ):
        return exit_codes.ERROR_INPUT_DEPENDENCY.format(
            parameter1="keep_symmetry", parameter2="lattice_symmetry"
        )
    for keyword in ["keep_symmetry", "keep_angles", "pressure_tolerance"]:
        if keyword in inputs_opt_p:
            dict_set_parameter(
                parameters,
                ["MOTION", "CELL_OPT", keyword.upper()],
                inputs_opt_p[keyword].value,
            )
    if "cell_symmetry" in inputs_opt_p:
        dict_create_tree(parameters, ["FORCE_EVAL", "SUBSYS", "CELL"])
        dict_set_parameter(
            parameters,
            ["FORCE_EVAL", "SUBSYS", "CELL", "SYMMETRY"],
            inputs_opt_p.cell_symmetry,
        )
    if "ref_cell_scaling_factor" in inputs_opt_p:
        structure = ctx.inputs.structure
        scaling_factor = inputs_opt_p.ref_cell_scaling_factor.value
        ref_cell_dict = {
            letter: " ".join(str(coord * scaling_factor) for coord in cell_v)
            for letter, cell_v in zip(["A", "B", "C"], structure.cell)
        }
        pbc = dict_retrieve_parameter(parameters, ["FORCE_EVAL", "SUBSYS", "CELL", "PERIODIC"])
        if pbc is not None:
            ref_cell_dict["PERIODIC"] = pbc
        dict_create_tree(parameters, ["FORCE_EVAL", "SUBSYS", "CELL"])
        dict_set_parameter(parameters, ["FORCE_EVAL", "SUBSYS", "CELL", "CELL_REF"], ref_cell_dict)
