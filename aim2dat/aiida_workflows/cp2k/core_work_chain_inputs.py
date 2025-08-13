"""Functions that create or update input parameters for the cp2k core work chains."""

# Standard library imports
import os
import re

# Third party library imports
import aiida.orm as aiida_orm

# Internal library imports
from aim2dat.aiida_workflows.cp2k import _supported_versions
from aim2dat.utils.dict_tools import (
    dict_set_parameter,
    dict_retrieve_parameter,
)
from aim2dat.aiida_workflows.cp2k.auxiliary_functions import estimate_comp_resources
from aim2dat.io import read_yaml_file


cwd = os.path.dirname(__file__)


def _get_version(cp2k_code):
    """
    Extract CP2K version from code label or description.
    """
    pattern = re.compile(r"\d+\.\d+(.\d+)?")
    matches = [
        match for match in pattern.finditer(cp2k_code.full_label + " " + cp2k_code.description)
    ]
    for m in matches:
        if m.group(0) in _supported_versions:
            return m.group(0)
    return "2024.1"


def _set_numerical_p_xc_functional(cp2k_dict, input_p, cp2k_version):
    """
    Set the parameters for the exchange-correlation functional in the input-parameters.
    """
    xc_keyword_dict = read_yaml_file(cwd + "/parameter_files/xc_functionals_p.yaml")
    xc_functional = input_p.value.upper()
    xc_parameters = xc_keyword_dict.get(xc_functional, [])
    for xc_par in xc_parameters:
        sup_versions = xc_par.pop("versions")
        if sup_versions == "all" or cp2k_version in sup_versions:
            dict_set_parameter(cp2k_dict, ["FORCE_EVAL", "DFT", "XC"], xc_par)
            return True, None
    return False, {"parameter": "numerical_p.xc_functional"}


def _set_numerical_p_basis_sets(cp2k_dict, input_p, structure, xc_functional):
    """Validate and set the basis set parameters."""
    bs_type = ""
    if isinstance(input_p, aiida_orm.Str):
        if xc_functional is None:
            return False, {
                "parameter1": "numerical_p.basis_sets",
                "parameter2": "numerical_p.xc_functional",
            }
        bs_type = input_p.value.split("-")[0].upper()
        bs_size = input_p.value.split("-")[1].upper()
        xc_funct = xc_functional.value.split("-")[0].upper()
        pred_basis_sets = read_yaml_file(cwd + "/parameter_files/basis_sets.yaml")
        if bs_type not in pred_basis_sets:
            return False, {"parameter": "numerical_p.basis_sets"}
        basis_sets = {
            "basis_set_file_name": pred_basis_sets[bs_type][xc_funct].pop("BASIS_SET_FILE_NAME"),
            "potential_file_name": pred_basis_sets[bs_type][xc_funct].pop("POTENTIAL_FILE_NAME"),
        }
        for el, bs_details in pred_basis_sets[bs_type][xc_funct].items():
            if bs_size not in bs_details:
                return False, {"parameter": "numerical_p.basis_sets"}
            basis_sets[el] = bs_details[bs_size]
    else:
        basis_sets = input_p.get_dict()

    cp2k_kinds = []
    for kind in structure.kinds:
        bs_entry = basis_sets[kind.symbol]
        kind_dict = {
            "_": kind.name,
            "BASIS_SET": bs_entry[0],
            "POTENTIAL": bs_entry[1],
        }
        if len(bs_entry) > 2 and bs_type == "ADMM":
            kind_dict["BASIS_SET AUX_FIT"] = bs_entry[2]
        if kind.name != kind.symbol:
            kind_dict["ELEMENT"] = kind.symbol
        cp2k_kinds.append(kind_dict)
    dict_set_parameter(cp2k_dict, ["FORCE_EVAL", "SUBSYS", "KIND"], cp2k_kinds)

    if bs_type == "ADMM":
        dict_set_parameter(
            cp2k_dict,
            ["FORCE_EVAL", "DFT", "AUXILIARY_DENSITY_MATRIX_METHOD"],
            {"ADMM_TYPE": "ADMMS"},
        )

    if "basis_set_file_name" in basis_sets:
        basis_filenames = dict_retrieve_parameter(
            cp2k_dict, ["FORCE_EVAL", "DFT", "BASIS_SET_FILE_NAME"]
        )
        if basis_filenames is None:
            basis_filenames = []
        new_file_names = basis_sets["basis_set_file_name"]
        if isinstance(new_file_names, str):
            new_file_names = [new_file_names]
        if isinstance(basis_filenames, str):
            basis_filenames = [basis_filenames]
        for bs_fn in new_file_names:
            if bs_fn not in basis_filenames:
                basis_filenames.append(bs_fn)
        dict_set_parameter(
            cp2k_dict,
            ["FORCE_EVAL", "DFT", "BASIS_SET_FILE_NAME"],
            basis_filenames,
        )

    if "potential_file_name" in basis_sets:
        dict_set_parameter(
            cp2k_dict,
            ["FORCE_EVAL", "DFT", "POTENTIAL_FILE_NAME"],
            basis_sets["potential_file_name"],
        )
    return True, None


def _set_input_parameters(
    inputs, ctx, fix_scf_m, fix_smearing, smearing_levels, initial_scf_guess
):
    """Set the self.inputs and self.ctx parameter of the core work chains."""
    ctx.inputs.structure = inputs.structural_p.structure
    # General parameters:
    ctx.scf_m_info = {
        "kpoints_ref_dist": None,
        "factor_unocc_states": inputs.factor_unocc_states.value,
        "fix_smearing": fix_smearing,
        "fix_scf_m": fix_scf_m,
        "smearing_levels": smearing_levels,
        "system_character": "unknown",
        "always_add_unocc_states": False,
        "allow_pulay": True,
        "scf_guess": initial_scf_guess,
    }
    if "always_add_unocc_states" in inputs and inputs.always_add_unocc_states.value:
        ctx.scf_m_info["always_add_unocc_states"] = True
    if "parent_calc_folder" in ctx.inputs:
        ctx.scf_m_info["scf_guess"] = "RESTART"

    # Set numerical p:
    if "numerical_p" in inputs:
        cp2k_p = ctx.inputs.parameters.get_dict()
        if "kpoints_ref_dist" in inputs.numerical_p:
            ctx.scf_m_info["kpoints_ref_dist"] = inputs.numerical_p.kpoints_ref_dist.value
        if "xc_functional" in inputs.numerical_p:
            cp2k_version = _get_version(ctx.inputs.code)
            ret, message = _set_numerical_p_xc_functional(
                cp2k_p, inputs.numerical_p.xc_functional, cp2k_version
            )
            if not ret:
                return "ERROR_INPUT_WRONG_VALUE", message

        if "basis_sets" in inputs.numerical_p:
            ret, message = _set_numerical_p_basis_sets(
                cp2k_p,
                inputs.numerical_p.basis_sets,
                inputs.structural_p.structure,
                inputs.numerical_p.get("xc_functional", None),
            )
            if not ret:
                if len(message) == 2:
                    return "ERROR_INPUT_DEPENDENCY", message
                else:
                    return "ERROR_INPUT_WRONG_VALUE", message

        if "cutoff_values" in inputs.numerical_p:
            cutoff_values = inputs.numerical_p.cutoff_values.get_dict()
            dict_set_parameter(
                cp2k_p, ["FORCE_EVAL", "DFT", "MGRID", "CUTOFF"], cutoff_values["cutoff"]
            )
            dict_set_parameter(
                cp2k_p,
                ["FORCE_EVAL", "DFT", "MGRID", "REL_CUTOFF"],
                cutoff_values["rel_cutoff"],
            )
            dict_set_parameter(
                cp2k_p, ["FORCE_EVAL", "DFT", "MGRID", "NGRIDS"], cutoff_values["ngrids"]
            )

        if "basis_file" in inputs.numerical_p:
            if "file" not in ctx.inputs:
                ctx.inputs.file = {}
            ctx.inputs.file["basis"] = inputs.numerical_p.basis_file
            basis_filenames = dict_retrieve_parameter(
                cp2k_p, ["FORCE_EVAL", "DFT", "BASIS_SET_FILE_NAME"]
            )
            if isinstance(basis_filenames, list):
                basis_filenames.append(inputs.numerical_p.basis_file.filename)
            elif isinstance(basis_filenames, str):
                basis_filenames = [
                    basis_filenames,
                    inputs.numerical_p.basis_file.filename,
                ]
            else:
                basis_filenames = inputs.numerical_p.basis_file.filename
            dict_set_parameter(
                cp2k_p, ["FORCE_EVAL", "DFT", "BASIS_SET_FILE_NAME"], basis_filenames
            )
        if "pseudo_file" in inputs.numerical_p:
            if "file" not in ctx.inputs:
                ctx.inputs.file = {}
            ctx.inputs.file["pseudo"] = inputs.numerical_p.pseudo_file
            dict_set_parameter(
                cp2k_p,
                ["FORCE_EVAL", "DFT", "POTENTIAL_FILE_NAME"],
                inputs.numerical_p.pseudo_file.filename,
            )
        if "cutoff_radius" in inputs.numerical_p:
            cutoff_radius = inputs.numerical_p.cutoff_radius.value
            interaction_potential = dict_retrieve_parameter(
                cp2k_p, ["FORCE_EVAL", "DFT", "XC", "HF", "INTERACTION_POTENTIAL"]
            )
            interaction_potential["cutoff_radius"] = cutoff_radius
            dict_set_parameter(
                cp2k_p,
                ["FORCE_EVAL", "DFT", "XC", "HF", "INTERACTION_POTENTIAL"],
                interaction_potential,
            )
        if "max_memory" in inputs.numerical_p:
            max_memory = inputs.numerical_p.max_memory.value
            dict_set_parameter(
                cp2k_p, ["FORCE_EVAL", "DFT", "XC", "HF", "MEMORY"], {"MAX_MEMORY": max_memory}
            )
        ctx.inputs.parameters = aiida_orm.Dict(dict=cp2k_p)

    # Set system-dependent resources:
    resources = dict_retrieve_parameter(ctx.inputs.metadata, ["options", "resources"])
    if resources is not None and "coeff" in resources:
        res_input_p = {
            "coeff": resources.pop("coeff"),
            "parameters": ctx.inputs.parameters.get_dict(),
            "structure": ctx.inputs.structure,
            "resources_dict": ctx.inputs.metadata["options"]["resources"],
        }
        for keyw in ["exp", "shift"]:
            if keyw in resources:
                res_input_p[keyw] = resources[keyw]
        estimate_comp_resources(**res_input_p)
