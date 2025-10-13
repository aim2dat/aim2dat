"""Routines to choose and update SCF parameters of the cp2k core work chains."""

# Standard library imports
import os

# Third party library imports
import aiida.orm as aiida_orm
from aiida.plugins import DataFactory

# Internal library imports
from aim2dat.aiida_workflows.cp2k.auxiliary_functions import calculate_added_mos
from aim2dat.utils.dict_tools import (
    dict_create_tree,
    dict_set_parameter,
    dict_retrieve_parameter,
    dict_merge,
)
from aim2dat.io import read_yaml_file

cwd = os.path.dirname(__file__)


def _set_scf_method_factors(scf_method, scf_extended_system):
    """Scale parameters for large systems."""
    for p_set in scf_method:
        factors = p_set.pop("factor_extended_system", [])
        if scf_extended_system:
            for kw_list, factor in factors:
                for parameters in p_set["parameters"]:
                    orig_value = dict_retrieve_parameter(parameters, kw_list)
                    orig_type = type(orig_value)
                    dict_set_parameter(parameters, kw_list, orig_type(orig_value * factor))


def _set_numerical_p_kpoints_ref_dist(structure, cp2k_dict, kpoints_ref_dist, use_odd_nrs):
    """
    Set Monkhorst-Pack k-points from reciprocal reference distance.
    """
    kpoints_type = DataFactory("core.array.kpoints")
    kpoints_grid = kpoints_type()
    kpoints_grid.set_cell_from_structure(structure)
    kpoints_grid.set_kpoints_mesh_from_density(kpoints_ref_dist)
    kpoints = []
    for pbc, kpt in zip(structure.pbc, kpoints_grid.get_kpoints_mesh()[0]):
        if use_odd_nrs:
            kpt_add = 1 if kpt % 2 == 0 else 0
        else:
            kpt_add = 1 if kpt % 2 != 0 else 0
        if pbc:
            kpoints.append(str(kpt_add + kpt))
        else:
            kpoints.append("1")
    dict_set_parameter(
        cp2k_dict,
        ["FORCE_EVAL", "DFT", "KPOINTS", "SCHEME"],
        "MONKHORST-PACK " + " ".join(kpoints),
    )


# def _set_smearing(parameters, temperature):
#     """Set the SMEAR section in parameters."""
#     if temperature == 0.0:
#         smear_dict = {"_": False}
#     else:
#         smear_dict = {
#             "ELECTRONIC_TEMPERATURE": temperature,
#             "METHOD": "FERMI_DIRAC",
#         }
#     dict_set_parameter(parameters, ["FORCE_EVAL", "DFT", "SCF", "SMEAR"], smear_dict)


def _set_added_mos(structure, parameters, factor_unocc_states):
    """Set the ADDED_MOS keyword in parameters."""
    n_unocc_states = calculate_added_mos(structure, parameters, factor_unocc_states)
    added_mos = dict_retrieve_parameter(parameters, ["FORCE_EVAL", "DFT", "SCF", "ADDED_MOS"])
    if added_mos is None:
        added_mos = 0
    new_added_mos = max(added_mos, n_unocc_states)
    dict_set_parameter(parameters, ["FORCE_EVAL", "DFT", "SCF", "ADDED_MOS"], new_added_mos)
    return new_added_mos


def _update_scf_parameters(
    structure, parameters, cur_scf_p, allow_smearing, allow_uks, disable_cholesky, scf_m_info
):
    """Update the input parameters for the calculation."""
    if cur_scf_p["smearing_level"] > 0 and not allow_smearing:
        return False
    if cur_scf_p["uks"] and not allow_uks:
        return False
    if cur_scf_p["cholesky"] and not disable_cholesky:
        return False
    mixing_scheme = dict_retrieve_parameter(cur_scf_p, ["parameters", "MIXING", "METHOD"])
    if (
        mixing_scheme is not None
        and mixing_scheme == "PULAY_MIXING"
        and not scf_m_info["allow_pulay"]
    ):
        return False
    dict_create_tree(parameters, ["FORCE_EVAL", "DFT", "SCF"])
    for keyw in ["MIXING", "OT", "OUTER_SCF", "MAX_SCF"]:
        if keyw in parameters["FORCE_EVAL"]["DFT"]["SCF"]:
            del parameters["FORCE_EVAL"]["DFT"]["SCF"][keyw]
    dict_merge(parameters["FORCE_EVAL"]["DFT"]["SCF"], cur_scf_p["parameters"])

    # Set restart:
    # if dict_retrieve_parameter(parameters, ["FORCE_EVAL", "DFT", "SCF", "SCF_GUESS"]) is None:
    dict_set_parameter(
        parameters, ["FORCE_EVAL", "DFT", "SCF", "SCF_GUESS"], scf_m_info["scf_guess"]
    )

    # Set UKS or ROKS:
    if cur_scf_p["uks"]:
        dict_set_parameter(parameters, ["FORCE_EVAL", "DFT", "UKS"], True)
    elif cur_scf_p["roks"]:
        dict_set_parameter(parameters, ["FORCE_EVAL", "DFT", "ROKS"], True)

    # Set CHOLESKY:
    if cur_scf_p["cholesky"]:
        dict_set_parameter(parameters, ["FORCE_EVAL", "DFT", "SCF", "CHOLESKY"], "OFF")

    # Set smearing:
    if scf_m_info["smearing_levels"][cur_scf_p["smearing_level"]] == 0.0:
        smear_dict = {"_": False}
    else:
        smear_dict = {
            "ELECTRONIC_TEMPERATURE": scf_m_info["smearing_levels"][cur_scf_p["smearing_level"]],
            "METHOD": "FERMI_DIRAC",
        }
    dict_set_parameter(parameters, ["FORCE_EVAL", "DFT", "SCF", "SMEAR"], smear_dict)
    cur_scf_p["smearing_temperature"] = scf_m_info["smearing_levels"][cur_scf_p["smearing_level"]]

    # Set unoccupied states:
    if (
        cur_scf_p["smearing_level"] > 0 or scf_m_info["always_add_unocc_states"]
    ) and "OT" not in cur_scf_p["parameters"]:
        cur_scf_p["added_mos"] = _set_added_mos(
            structure, parameters, scf_m_info["factor_unocc_states"]
        )
    else:
        cur_scf_p["added_mos"] = 0
        dict_set_parameter(parameters, ["FORCE_EVAL", "DFT", "SCF", "ADDED_MOS"], 0)

    # Set k-points:
    if scf_m_info["kpoints_ref_dist"] is not None:
        _set_numerical_p_kpoints_ref_dist(
            structure, parameters, scf_m_info["kpoints_ref_dist"], cur_scf_p["odd_kpoints"]
        )

    # Set EPS_SCF in OUTER_SCF:
    if "OT" in cur_scf_p["parameters"]:
        eps_scf = dict_retrieve_parameter(parameters, ["FORCE_EVAL", "DFT", "SCF", "EPS_SCF"])
        dict_set_parameter(
            parameters, ["FORCE_EVAL", "DFT", "SCF", "OUTER_SCF", "EPS_SCF"], eps_scf
        )
    return True


def _initialize_scf_parameters(inputs, ctx):
    reports = []
    # Load scf method parameters:
    if "custom_scf_method" in inputs:
        ctx.scf_method_p = inputs.custom_scf_method.get_list()
    else:
        ctx.scf_method_p = list(
            read_yaml_file(cwd + f"/scf_parameter_files/{inputs.scf_method.value}_p.yaml")
        )
    _set_scf_method_factors(ctx.scf_method_p, inputs.scf_extended_system.value)

    # parameters = ctx.inputs.parameters.get_dict()
    cur_scf_p = {
        "added_mos": 0,
        "method_level": 0,
        "parameter_level": 0,
        "smearing_level": 0,
        "parameters": ctx.scf_method_p[0]["parameters"][0],
        "roks": False,
        "uks": False,
        "cholesky": False,
    }
    if ctx.scf_m_info["system_character"] == "metallic":
        cur_scf_p["smearing_level"] = 1
    cur_scf_p_set = ctx.scf_method_p[cur_scf_p["method_level"]]
    if (
        all(not pbc for pbc in ctx.inputs.structure.pbc)
        and ctx.scf_m_info["kpoints_ref_dist"] is not None
    ):
        return (
            "ERROR_INPUT_LOGICAL",
            {
                "parameter1": "numerical_p.kpoints_ref_dist",
                "parameter2": "structural_p.structure.pbc",
            },
        ), reports
    if ctx.scf_m_info["kpoints_ref_dist"] is not None:
        cur_scf_p["odd_kpoints"] = False

    # Check for initial parameters
    if "scf_parameters" in inputs.structural_p:
        scf_parameters = inputs.structural_p.scf_parameters.get_dict()
        cur_scf_p["method_level"] = scf_parameters.get("method_level", 0)
        cur_scf_p["parameter_level"] = scf_parameters.get("parameter_level", 0)
        cur_scf_p["smearing_level"] = scf_parameters.get("smearing_level", 0)
        smearing_temp = scf_parameters.get("smearing_temperature", None)
        cur_scf_p["added_mos"] = scf_parameters.get("added_mos", 0)
        cur_scf_p["uks"] = scf_parameters.get("uks", False)
        cur_scf_p["roks"] = scf_parameters.get("roks", False)
        cur_scf_p["cholesky"] = scf_parameters.get("cholesky", False)
        if "odd_kpoints" in scf_parameters:
            cur_scf_p["odd_kpoints"] = scf_parameters["odd_kpoints"]
        if (cur_scf_p["uks"] and cur_scf_p["roks"]) or (
            cur_scf_p["uks"] and inputs.enable_roks.value
        ):
            return (
                "ERROR_INPUT_LOGICAL",
                {"parameter1": "scf_parameters-uks", "parameter2": "scf_parameters-roks"},
            ), reports

        # Check method level:
        if cur_scf_p["method_level"] >= len(ctx.scf_method_p):
            # Add report here?
            cur_scf_p["method_level"] = 0
        cur_scf_p_set = ctx.scf_method_p[cur_scf_p["method_level"]]
        cur_scf_p["parameters"] = scf_parameters.get("parameters", None)

        # Check parameter level:
        if cur_scf_p["parameters"] is not None:
            if cur_scf_p["parameters"] in cur_scf_p_set["parameters"]:
                cur_scf_p["parameter_level"] = cur_scf_p_set["parameters"].index(
                    cur_scf_p["parameters"]
                )
            else:
                # Add report
                cur_scf_p_set["parameters"].insert(0, cur_scf_p["parameters"])
                cur_scf_p["parameter_level"] = 0
        elif cur_scf_p["parameter_level"] >= len(cur_scf_p_set["parameters"]):
            cur_scf_p["parameter_level"] = 0

        # Check smearing level:
        orig_level = cur_scf_p["smearing_level"]
        if cur_scf_p["smearing_level"] >= len(ctx.scf_m_info["smearing_levels"]):
            cur_scf_p["smearing_level"] = len(ctx.scf_m_info["smearing_levels"]) - 1
        if smearing_temp is not None:
            if smearing_temp != ctx.scf_m_info["smearing_levels"][cur_scf_p["smearing_level"]]:
                if smearing_temp in ctx.scf_m_info["smearing_levels"]:
                    cur_scf_p["smearing_level"] = ctx.scf_m_info["smearing_levels"].index(
                        smearing_temp
                    )
                else:
                    for level_idx, temp_list in enumerate(ctx.scf_m_info["smearing_levels"]):
                        if temp_list > smearing_temp:
                            cur_scf_p["smearing_level"] = level_idx
                            break
        if orig_level != cur_scf_p["smearing_level"]:
            reports.append(
                f"Changed smearing level from {orig_level} to " f"{cur_scf_p['smearing_level']}."
            )

    # Set scf parameters:
    while cur_scf_p["method_level"] < len(ctx.scf_method_p):
        parameters = ctx.inputs.parameters.get_dict()
        successful_upd = _update_scf_parameters(
            ctx.inputs.structure,
            parameters,
            cur_scf_p,
            cur_scf_p_set.get("allow_smearing", True),
            cur_scf_p_set.get("allow_uks", True),
            cur_scf_p_set.get("disable_cholesky", True),
            ctx.scf_m_info,
        )
        if successful_upd or not inputs.adjust_scf_parameters.value:
            ctx.inputs.parameters = aiida_orm.Dict(dict=parameters)
            break
        else:
            cur_scf_p["method_level"] += 1
            cur_scf_p_set = ctx.scf_method_p[cur_scf_p["method_level"]]
    if not successful_upd:
        return (
            "ERROR_SCF_CONVERGENCE_NOT_REACHED",
            {},
        ), reports  # self.exit_codes.ERROR_SCF_CONVERGENCE_NOT_REACHED

    ctx.cur_scf_p = cur_scf_p
    ctx.cur_scf_p_set = cur_scf_p_set
    ctx.scf_m_info["max_method"] = len(ctx.scf_method_p)
    ctx.scf_m_info["max_parameter"] = len(ctx.cur_scf_p_set["parameters"])
    ctx.scf_m_info["max_smearing"] = len(ctx.scf_m_info["smearing_levels"])
    return None, reports


def _iterate_scf_parameters(
    structure, parameters, scf_method_p, cur_scf_p, cur_scf_p_set, scf_m_info
):
    successful_upd = False
    while not successful_upd:
        if (
            not scf_m_info["fix_scf_m"]
            and cur_scf_p["method_level"] < scf_m_info["max_method"] - 1
        ):
            cur_scf_p["method_level"] += 1
            cur_scf_p_set = scf_method_p[cur_scf_p["method_level"]]
            cur_scf_p["parameters"] = cur_scf_p_set["parameters"][cur_scf_p["parameter_level"]]
            scf_m_info["max_parameter"] = len(cur_scf_p_set["parameters"])
            successful_upd = _update_scf_parameters(
                structure,
                parameters,
                cur_scf_p,
                cur_scf_p_set.get("allow_smearing", True),
                cur_scf_p_set.get("allow_uks", True),
                cur_scf_p_set.get("disable_cholesky", True),
                scf_m_info,
            )
        elif (
            not scf_m_info["fix_smearing"]
            and scf_m_info["system_character"] != "insulator"
            and cur_scf_p["smearing_level"] < scf_m_info["max_smearing"] - 1
        ):
            if not scf_m_info["fix_scf_m"]:
                cur_scf_p["method_level"] = 0
                cur_scf_p_set = scf_method_p[cur_scf_p["method_level"]]
                cur_scf_p["parameters"] = cur_scf_p_set["parameters"][cur_scf_p["parameter_level"]]
                scf_m_info["max_parameter"] = len(cur_scf_p_set["parameters"])
            cur_scf_p["smearing_level"] += 1
            successful_upd = _update_scf_parameters(
                structure,
                parameters,
                cur_scf_p,
                cur_scf_p_set.get("allow_smearing", True),
                cur_scf_p_set.get("allow_uks", True),
                cur_scf_p_set.get("disable_cholesky", True),
                scf_m_info,
            )
        elif "odd_kpoints" in cur_scf_p and not cur_scf_p["odd_kpoints"]:
            if not scf_m_info["fix_scf_m"]:
                cur_scf_p["method_level"] = 0
                cur_scf_p_set = scf_method_p[cur_scf_p["method_level"]]
                cur_scf_p["parameters"] = cur_scf_p_set["parameters"][cur_scf_p["parameter_level"]]
                scf_m_info["max_parameter"] = len(cur_scf_p_set["parameters"])
            if not scf_m_info["fix_smearing"]:
                if scf_m_info["system_character"] == "metallic":
                    cur_scf_p["smearing_level"] = 1
                else:
                    cur_scf_p["smearing_level"] = 0
            cur_scf_p["odd_kpoints"] = not cur_scf_p["odd_kpoints"]
            successful_upd = _update_scf_parameters(
                structure,
                parameters,
                cur_scf_p,
                cur_scf_p_set.get("allow_smearing", True),
                cur_scf_p_set.get("allow_uks", True),
                cur_scf_p_set.get("disable_cholesky", True),
                scf_m_info,
            )
        elif cur_scf_p["parameter_level"] < scf_m_info["max_parameter"] - 1:
            if not scf_m_info["fix_scf_m"]:
                cur_scf_p["method_level"] = 0
                cur_scf_p_set = scf_method_p[cur_scf_p["method_level"]]
                scf_m_info["max_parameter"] = len(cur_scf_p_set["parameters"])
            if not scf_m_info["fix_smearing"]:
                if scf_m_info["system_character"] == "metallic":
                    cur_scf_p["smearing_level"] = 1
                else:
                    cur_scf_p["smearing_level"] = 0
            if "odd_kpoints" in cur_scf_p:
                cur_scf_p["odd_kpoints"] = not cur_scf_p["odd_kpoints"]
            cur_scf_p["parameter_level"] += 1
            cur_scf_p["parameters"] = cur_scf_p_set["parameters"][cur_scf_p["parameter_level"]]
            successful_upd = _update_scf_parameters(
                structure,
                parameters,
                cur_scf_p,
                cur_scf_p_set.get("allow_smearing", True),
                cur_scf_p_set.get("allow_uks", True),
                cur_scf_p_set.get("disable_cholesky", True),
                scf_m_info,
            )
        else:
            return False
    return successful_upd


def _compare_scf_p(dict_input, dict_output):
    """Compare initial and final scf-parameters."""
    same_parameters = True
    if "odd_kpoints" not in dict_input:
        dict_input["odd_kpoints"] = False
    for keyword in [
        "roks",
        "uks",
        "cholesky",
        "method_level",
        "parameter_level",
        "smearing_level",
        "parameters",
        "odd_kpoints",
    ]:
        if keyword not in dict_input or keyword not in dict_output:
            same_parameters = False
            break
        if dict_input[keyword] != dict_output[keyword]:
            same_parameters = False
            break
    return same_parameters
