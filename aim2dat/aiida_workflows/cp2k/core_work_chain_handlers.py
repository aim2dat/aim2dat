"""Error handlers for core work chains."""

# Third party library imports
import aiida.orm as aiida_orm
from aiida.engine import ProcessHandlerReport

# Internal library imports
from aim2dat.utils.dict_tools import (
    dict_retrieve_parameter,
    dict_set_parameter,
    dict_merge,
)
from aim2dat.aiida_workflows.cp2k.core_work_chain_scf import (
    _update_scf_parameters,
    _iterate_scf_parameters,
)


def _switch_scf_parameters(inputs, ctx, exit_codes, calc):
    output_p_dict = calc.outputs["output_parameters"].get_dict()
    if "scf_converged" not in output_p_dict:
        return ProcessHandlerReport(exit_code=exit_codes.ERROR_CALCULATION_ABORTED), []
    adjust_scf_p = True
    if "adjust_scf_parameters" in inputs and not inputs.adjust_scf_parameters.value:
        adjust_scf_p = False
    if adjust_scf_p and not output_p_dict["scf_converged"]:
        parameters = ctx.inputs.parameters.get_dict()
        successful_upd = _iterate_scf_parameters(
            # structure, parameters, scf_method_p, cur_scf_p, cur_scf_p_set, scf_m_info
            ctx.inputs.structure,
            parameters,
            ctx.scf_method_p,
            ctx.cur_scf_p,
            ctx.cur_scf_p_set,
            ctx.scf_m_info,
        )
        if successful_upd:
            ctx.inputs.parameters = aiida_orm.Dict(dict=parameters)
            return ProcessHandlerReport(do_break=True), []
        else:
            return ProcessHandlerReport(exit_code=exit_codes.ERROR_SCF_CONVERGENCE_NOT_REACHED), []
    return None, []


def _switch_to_open_shell_ks(inputs, ctx, exit_codes, calc):
    if inputs.enable_roks.value:
        ctx.cur_scf_p["roks"] = True
    else:
        ctx.cur_scf_p["uks"] = True
    parameters = ctx.inputs.parameters.get_dict()
    _update_scf_parameters(
        ctx.inputs.structure,
        parameters,
        ctx.cur_scf_p,
        ctx.cur_scf_p_set.get("allow_smearing", True),
        ctx.cur_scf_p_set.get("allow_uks", True),
        ctx.cur_scf_p_set.get("disable_cholesky", True),
        ctx.scf_m_info,
    )
    ctx.inputs.parameters = aiida_orm.Dict(dict=parameters)
    return ProcessHandlerReport(do_break=True), []


def _switch_to_atomic_scf_guess(inputs, ctx, exit_codes, calc):
    output_p_dict = calc.outputs["output_parameters"].get_dict()
    if "scf_converged" not in output_p_dict:
        return ProcessHandlerReport(exit_code=exit_codes.ERROR_CALCULATION_ABORTED), []
    if not output_p_dict["scf_converged"]:
        ctx.scf_m_info["scf_guess"] = "ATOMIC"
        parameters = ctx.inputs.parameters.get_dict()
        _update_scf_parameters(
            ctx.inputs.structure,
            parameters,
            ctx.cur_scf_p,
            ctx.cur_scf_p_set.get("allow_smearing", True),
            ctx.cur_scf_p_set.get("allow_uks", True),
            ctx.cur_scf_p_set.get("disable_cholesky", True),
            ctx.scf_m_info,
        )
        ctx.inputs.parameters = aiida_orm.Dict(dict=parameters)
        return ProcessHandlerReport(do_break=True), []
    return None, []


def _switch_to_broyden_mixing(inputs, ctx, exit_codes, calc):
    def set_to_standard_alpha_beta(alpha, beta):
        if alpha is None:
            alpha = 0.4
        if beta is None:
            beta = 0.5
        return alpha, beta

    reports = []
    mixing_scheme = dict_retrieve_parameter(ctx.cur_scf_p, ["parameters", "MIXING", "METHOD"])
    if mixing_scheme is not None and mixing_scheme == "PULAY_MIXING":
        alpha = dict_retrieve_parameter(ctx.cur_scf_p, ["parameters", "MIXING", "ALPHA"])
        beta = dict_retrieve_parameter(ctx.cur_scf_p, ["parameters", "MIXING", "BETA"])
        alpha, beta = set_to_standard_alpha_beta(alpha, beta)
        found_p = False
        for m_lvl, method in enumerate(ctx.scf_method_p):
            if not method["allow_uks"] and ctx.cur_scf_p["uks"]:
                continue
            if (
                not method["allow_smearing"]
                and ctx.scf_m_info["smearing_levels"][ctx.cur_scf_p["smearing_level"]] > 0.0
            ):
                continue

            for p_lvl, mixing_p in enumerate(method["parameters"]):
                mixing_scheme_new = dict_retrieve_parameter(mixing_p, ["MIXING", "METHOD"])
                if mixing_scheme_new is None:
                    continue
                if mixing_scheme_new == "PULAY_MIXING":
                    continue
                alpha_new = dict_retrieve_parameter(mixing_p, ["MIXING", "ALPHA"])
                beta_new = dict_retrieve_parameter(mixing_p, ["MIXING", "BETA"])
                alpha_new, beta_new = set_to_standard_alpha_beta(alpha_new, beta_new)
                if beta == beta_new and alpha == alpha_new:
                    ctx.cur_scf_p["method_level"] = m_lvl
                    ctx.cur_scf_p["parameter_level"] = p_lvl
                    ctx.cur_scf_p["parameters"] = mixing_p
                    ctx.cur_scf_p["method_level"] = m_lvl
                    ctx.cur_scf_p["parameter_level"] = p_lvl
                    ctx.cur_scf_p_set = method
                    ctx.max_parameter_level = len(method["parameters"])
                    found_p = True
                    break
            if found_p:
                break
        parameters = ctx.inputs.parameters.get_dict()
        if found_p and _update_scf_parameters(
            ctx.inputs.structure,
            parameters,
            ctx.cur_scf_p,
            ctx.cur_scf_p_set.get("allow_smearing", True),
            ctx.cur_scf_p_set.get("allow_uks", True),
            ctx.cur_scf_p_set.get("disable_cholesky", True),
            ctx.scf_m_info,
        ):
            reports.append("Switching back to broyden-mixing.")
            ctx.inputs.parameters = aiida_orm.Dict(dict=parameters)
            ctx.scf_m_info["allow_pulay"] = False
            return ProcessHandlerReport(do_break=True), reports
        else:
            return ProcessHandlerReport(exit_code=exit_codes.ERROR_CALCULATION_ABORTED), reports
    return None, reports


def _resubmit_unconverged_geometry(inputs, ctx, exit_Codes, calc):
    ctx.opt_level += 1
    ctx.opt_iteration = 0
    if ctx.opt_level < len(ctx.opt_method_p):
        parameters = ctx.inputs.parameters.get_dict()
        dict_merge(
            parameters["MOTION"][ctx.opt_type],
            ctx.opt_method_p[ctx.opt_level],
        )
        ctx.inputs.parameters = aiida_orm.Dict(dict=parameters)
        if "parent_calc_folder" in ctx.inputs:
            ctx.inputs.parent_calc_folder = calc.outputs["remote_folder"]
        _update_structure(ctx, calc.outputs, set_start_iteration=False)
        return ProcessHandlerReport(do_break=True), []
    return None, []


def _resubmit_calculation(inputs, ctx, exit_codes, calc):
    if "parent_calc_folder" in ctx.inputs:
        ctx.inputs.parent_calc_folder = calc.outputs["remote_folder"]
    if "opt_type" in ctx:
        _update_structure(ctx, calc.outputs, set_start_iteration=True)
    return ProcessHandlerReport(do_break=True), []


def _update_structure(ctx, outputs, set_start_iteration=False):
    """Update the structure taken from the last calculation."""
    result_dict = outputs["output_parameters"].get_dict()
    parameters = ctx.inputs.parameters.get_dict()
    nr_steps = dict_retrieve_parameter(result_dict, ["nr_steps"])
    if set_start_iteration and nr_steps is not None:
        if nr_steps > ctx.opt_method_p[ctx.opt_level]["MAX_ITER"] - 1:
            ctx.opt_iteration = ctx.opt_method_p[ctx.opt_level]["MAX_ITER"] - 2
        else:
            ctx.opt_iteration = nr_steps
        dict_set_parameter(
            parameters,
            ["MOTION", ctx.opt_type, "STEP_START_VAL"],
            ctx.opt_iteration,
        )
    elif (
        dict_retrieve_parameter(parameters, ["MOTION", ctx.opt_type, "STEP_START_VAL"]) is not None
    ):
        del parameters["MOTION"][ctx.opt_type]["STEP_START_VAL"]

    ctx.inputs.parameters = aiida_orm.Dict(dict=parameters)
    if "output_structure" in outputs:
        ctx.inputs.structure = outputs["output_structure"]
