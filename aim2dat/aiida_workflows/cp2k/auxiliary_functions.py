"""Auxiliary functions for AiiDA work chains."""

# Standard library imports
import math
import os
import re

# Third party library imports
import aiida.orm as aiida_orm
from aiida.engine import calcfunction

# Internal library imports
from aim2dat.elements import get_atomic_number
from aim2dat.units import energy
from aim2dat.utils.dict_tools import dict_set_parameter, dict_retrieve_parameter
from aim2dat.io import read_yaml_file


@calcfunction
def return_scf_parameters(scf_parameters):  # , smearing_temp):
    """
    Aiida calcfuntion that creates a dictionary of the mixing parameters.
    """
    scf_p_output = scf_parameters.get_dict()
    # scf_p_output["smearing_temperature"] = smearing_temp.value
    return aiida_orm.Dict(dict=scf_p_output)


@calcfunction
def return_runtime_stats(**calcjob_output_parameters):
    """
    Return runtime statistics of the work chain.
    """
    fin_calcs = []
    unf_calcs = []
    run_times = []
    for output_p in calcjob_output_parameters.values():
        run_time = output_p.get_dict().get("runtime", None)
        if run_time is not None:
            fin_calcs.append(run_time)
        else:
            unf_calcs.append(None)
        run_times.append(run_time)
    run_times_dict = {
        "nr_calculations": len(run_times),
        "nr_unf_calculations": len(unf_calcs),
        "total_runtime": sum(fin_calcs),
        "avg_runtime_per_calc": sum(fin_calcs) / len(fin_calcs),
        "runtimes": run_times,
    }
    return aiida_orm.Dict(dict=run_times_dict)


@calcfunction
def return_work_chain_info(cp2k_output, structure, **kwargs):
    """
    Return general information of the work chain.
    """
    output_parameters = cp2k_output.get_dict()
    work_chain_info = {}
    work_chain_info["total_energy"] = output_parameters["energy"] * energy.Hartree
    work_chain_info["energy_units"] = "eV"
    work_chain_info["natoms"] = output_parameters["natoms"]
    work_chain_info["total_energy_per_atom"] = work_chain_info["total_energy"] / float(
        work_chain_info["natoms"]
    )
    work_chain_info["chem_formula"] = structure.get_composition()
    for label, value in kwargs.items():
        if isinstance(value, aiida_orm.Dict):
            value = value.get_dict()
        elif isinstance(value, aiida_orm.List):
            value = value.get_list()
        else:
            value = value.value
        work_chain_info[label] = value
    return aiida_orm.Dict(dict=work_chain_info)


@calcfunction
def return_rec_space_eigenvalues(cp2k_output):
    """
    Calcfuntion that summarizes information on the eigenvalues and 1st Brillouin zone.
    """
    output_parameters = cp2k_output.get_dict()
    eigenvalue_info = {
        "fermi_energy": output_parameters.get("fermi_energy"),
        "energy_units": output_parameters["energy_units"],
        "kpoint_units": "2pi/bohr",
        "direct_band_gap": output_parameters["eigenvalue_info"]["direct_gap"],
        "band_gap": output_parameters["eigenvalue_info"]["gap"],
    }
    eigenvalue_info["eigenvalues"] = output_parameters["eigenvalue_info"]["eigenvalues"]

    return aiida_orm.Dict(dict=eigenvalue_info)


def estimate_comp_resources(structure, parameters, resources_dict, coeff, exp=3.0, shift=12.0):
    """
    Estimate the number of nodes based on the number of electrons.

    The following equation is used: ntasks = coeff*nelec^exp + shift

    Parameters
    ----------
    structure : aiida.structure
        AiiDA structure node.
    parameters : dict
        CP2K input parameters.
    resources_dict : dict
        Dictionary giving the parameters ``'num_mpiprocs_per_machine'``.
    coeff : float
        Coefficient of the leading term.
    exp : float
        Exponent of the leading term.
    shift : float
        Constant shift of the  equation. The default value is ``12.0``.
    """
    if "num_mpiprocs_per_machine" in resources_dict:
        num_mpiprocs_per_machine = resources_dict["num_mpiprocs_per_machine"]
        if "num_cores_per_mpiproc" in resources_dict:
            num_cores_per_mpiproc = resources_dict["num_cores_per_mpiproc"]
        else:
            num_cores_per_mpiproc = 1
        nelectrons = calc_nr_explicit_electrons(structure, parameters)
        num_tasks = round(coeff * nelectrons**exp + shift)
        num_mpiprocs = round(num_tasks / num_cores_per_mpiproc)
        num_machines = math.ceil(num_tasks / (num_cores_per_mpiproc * num_mpiprocs_per_machine))
        num_mpiprocs_per_machine = math.floor(num_mpiprocs / num_machines)
        resources_dict["num_machines"] = num_machines
        resources_dict["num_mpiprocs_per_machine"] = num_mpiprocs_per_machine


def set_xc_functional(input_dict, xc_functional):
    """
    Set the parameters for the exchange-correlation functional in the input-paramters.

    Parameters
    ----------
    input_dict : dict
        Input parmaters for the cp2k calculation.
    xc_functional : str
        Exchange-correlation functional used for the calculation.

    Returns
    -------
    input_dict : dict
        Input parmaters for the cp2k calculation.
    """
    file_path = os.path.dirname(__file__) + "/parameter_files/xc_functionals_p.yaml"
    xc_keyword_dict = read_yaml_file(file_path)

    xc_functional = xc_functional.upper()
    if xc_functional in xc_keyword_dict:
        dict_set_parameter(input_dict, ["FORCE_EVAL", "DFT", "XC"], xc_keyword_dict[xc_functional])
    else:
        raise ValueError(f"{xc_functional} not supported.")


def calc_nr_explicit_electrons(structure, parameters):
    """
    Determine the number of valence electrons based on the structure and
    the pseudopotential name.

    Parameters
    ----------
    structure : aiida.structure
        The input structure.
    parameters : dict
        Dictionary of the input parameters for the cp2k calculation.

    Returns
    -------
    nelectrons : (int)
        Number of valence electrons.
    """
    nelectrons = 0
    nelec_atoms = None
    fallback_nelec_atom = 10
    pattern = re.compile("q([0-9]*)")

    # Retrieve valence electrons:
    basis_sets = dict_retrieve_parameter(parameters, ["FORCE_EVAL", "SUBSYS", "KIND"])
    if isinstance(basis_sets, list):
        nelec_atoms = {}
        for kind in basis_sets:
            # 1. Try to get valence electrons from basis set name.
            # 2. Try pp name
            # 3. Try to get explicit electrons from element (assuming all-electron calculation)
            # 4. Try to get explicit electrons from kind (assuming all-electron calculation)
            if len(pattern.findall(kind["BASIS_SET"])) > 0:
                nelec_atoms[kind["_"]] = int(pattern.findall(kind["BASIS_SET"])[0])
            elif len(pattern.findall(kind["POTENTIAL"])) > 0:
                nelec_atoms[kind["_"]] = int(pattern.findall(kind["POTENTIAL"])[0])
            elif "ELEMENT" in kind:
                nelec_atoms[kind["_"]] = get_atomic_number(kind["ELEMENT"])
            else:
                try:
                    nelec_atoms[kind["_"]] = get_atomic_number(kind["_"])
                except ValueError:
                    # Hard coded fall-back value:
                    nelec_atoms[kind["_"]] = fallback_nelec_atom

    # Sum up electrons of each site:
    for site in structure.sites:
        if nelec_atoms is None:
            nelectrons += fallback_nelec_atom
        else:
            try:
                nelectrons += nelec_atoms[site.kind_name]
            except KeyError:
                nelectrons += fallback_nelec_atom
    return nelectrons


def calculate_added_mos(structure, parameters, factor_unocc_states=0.3, return_n_electrons=False):
    """
    Calculate the number of unoccupied states based on the number of electrons and the smearing
    temperature.

    The following formula is used to calculate the number of states:
    max(10, nelectrons*factor_unocc_states*(1.0 + electronic_temperature/1000.0))

    Parameters
    ----------
    structure : aiida.structure
        The input structure.
    parameters : dict
        Dictionary of the input parameters for the cp2k calculation.
    factor_unocc_states : float (optional)
        Empirical factor. The default value is ``0.3``.

    Returns
    -------
    added_mos : int
        Number of unoccupied states
    """
    # spin_nondeg_keywords = [
    #     "UKS",
    #     "UNRESTRICTED_KOHN_SHAM",
    #     "LSD",
    #     "SPIN_POLARIZED",
    #     "ROKS",
    #     "RESTRICTED_OPEN_KOHN_SHAM",
    # ]

    temperature = 0.0
    nelectrons = calc_nr_explicit_electrons(structure, parameters)
    spin_factor = 1.0
    try:
        temperature = parameters["FORCE_EVAL"]["DFT"]["SCF"]["SMEAR"]["ELECTRONIC_TEMPERATURE"]
    except KeyError:
        pass
    # for keyword in spin_nondeg_keywords:
    #     try:
    #         if parameters["FORCE_EVAL"]["DFT"][keyword]:
    #             print(keyword, parameters["FORCE_EVAL"]["DFT"][keyword])
    #             spin_factor = 0.5
    #     except KeyError:
    #         pass
    added_mos = max(
        25, int(spin_factor * nelectrons * factor_unocc_states * (1.0 + temperature / 1000.0))
    )
    if return_n_electrons:
        return added_mos, nelectrons
    else:
        return added_mos


def create_aiida_node(value, node_type=None):
    """
    Create AiiDA data node from standard python variable.

    Parameters
    ----------
    value : variable
        Input variable.
    node_type : str (optional)
        AiiDA node type. The default value is ``None``.

    Returns
    -------
    aiida_node : variable
        AiiDA data node.
    """
    # TODO inlcude more node types, use DataFactory?
    check_node_type = False
    if node_type is None:
        check_node_type = True

    if node_type == "int" or (check_node_type and isinstance(value, int)):
        aiida_node = aiida_orm.Int(value)
    elif node_type == "float" or (check_node_type and isinstance(value, float)):
        aiida_node = aiida_orm.Float(value)
    elif node_type == "str" or (check_node_type and isinstance(value, str)):
        aiida_node = aiida_orm.Str(value)
    elif node_type == "bool" or (check_node_type and isinstance(value, bool)):
        aiida_node = aiida_orm.Bool(value)
    elif node_type == "list" or (check_node_type and isinstance(value, list)):
        aiida_node = aiida_orm.List(list=value)
    elif node_type == "dict" or (check_node_type and isinstance(value, dict)):
        aiida_node = aiida_orm.Dict(dict=value)
    else:
        raise ValueError(f"{type(value)} is not supported.")
    return aiida_node


# def dict_set_parameter(dictionary, parameter_tree, value):
#     """
#     Set parameter in a nested dictionary.
#
#     Parameters
#     ----------
#     dictionary : dict
#         Input dictionary.
#     paramter_tree : list
#         List of dictionary key words.
#     value : str, float or int
#         Value of the parameter.
#
#     Returns
#     -------
#     dictionary : dict
#         Output dictionary.
#     """
#     helper_dict = dictionary
#     can_add = True
#
#     for parameter in parameter_tree[:-1]:
#         if parameter in helper_dict:
#             helper_dict = helper_dict[parameter]
#         elif isinstance(helper_dict, dict):
#             helper_dict[parameter] = {}
#             helper_dict = helper_dict[parameter]
#         else:
#             can_add = False
#
#     if can_add:
#         helper_dict[parameter_tree[-1]] = value
#     else:
#         raise ValueError("Cannot add value to dictionary.")
#     return dictionary
#
#
# def dict_retrieve_parameter(dictionary, parameter_tree):
#     """
#     Retrieve value from nested dictionary.
#
#     Parameters
#     ----------
#     dictionary : dict
#         Input dictionary.
#     parameter_tree : list
#         List of dictionary key words.
#
#     Returns
#     -------
#     value :
#         The value of the parameter or ``None`` if the key word cound not be found.
#     """
#     helper_dict = dictionary
#
#     for parameter in parameter_tree:
#         if parameter in helper_dict:
#             helper_dict = helper_dict[parameter]
#         else:
#             helper_dict = None
#             break
#     return helper_dict
#
#
# def dict_create_tree(dictionary, parameter_tree):
#     """
#     Create a nested dictionary.
#
#     Parameters
#     ----------
#     dictionary : dict
#         Input dictionary.
#     parameter_tree : list
#         List of dictionary key words.
#     """
#     helper_dict = dictionary
#     for parameter in parameter_tree:
#         if isinstance(helper_dict, dict):
#             if parameter in helper_dict:
#                 helper_dict = helper_dict[parameter]
#             else:
#                 helper_dict[parameter] = {}
#                 helper_dict = helper_dict[parameter]
#         else:
#             raise ValueError("Cannot create nested dictionary.")
#     return dictionary
