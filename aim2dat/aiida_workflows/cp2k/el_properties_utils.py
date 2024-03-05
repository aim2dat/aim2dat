"""
Auxiliary functions for the ElectronicPropertiesWorkChain.
"""

# Standard library imports
import os

# Internal library imports
from aim2dat.aiida_workflows._workflow_builder_utils import _load_protocol
from aim2dat.aiida_workflows.utils import (
    create_aiida_node,
    obtain_value_from_aiida_node,
)


cwd = os.path.dirname(__file__)


def elprop_setup(ctx, inputs):
    """Set up calculation parameters."""
    if "custom_protocol" in inputs:
        protocol_dict = inputs.workflow.custom_protocol.get_dict()
    else:
        protocol_dict = _load_protocol(inputs.workflow.protocol.value, cwd + "/../protocols/")

    general_parameters = {}
    if "general_input" in protocol_dict:
        for key, value in protocol_dict["general_input"].items():
            if value["aiida_node"]:
                general_parameters[key] = create_aiida_node(value["value"])
            else:
                general_parameters[key] = value["value"]

    set_ctx_parameters_from_protocol(
        ctx, inputs, protocol_dict, general_parameters, "find_scf_parameters"
    )
    if inputs.workflow.run_cell_optimization.value:
        set_ctx_parameters_from_protocol(
            ctx, inputs, protocol_dict, general_parameters, "unit_cell_opt"
        )
    if inputs.workflow.calc_band_structure.value:
        set_ctx_parameters_from_protocol(
            ctx, inputs, protocol_dict, general_parameters, "band_structure"
        )
    if inputs.workflow.calc_eigenvalues.value:
        set_ctx_parameters_from_protocol(
            ctx, inputs, protocol_dict, general_parameters, "eigenvalues"
        )
    if inputs.workflow.calc_pdos.value:
        set_ctx_parameters_from_protocol(ctx, inputs, protocol_dict, general_parameters, "pdos")
    if inputs.workflow.calc_partial_charges.value:
        set_ctx_parameters_from_protocol(
            ctx, inputs, protocol_dict, general_parameters, "partial_charges"
        )


def set_ctx_parameters_from_protocol(ctx, inputs, protocol_dict, general_parameters, task_label):
    """Set ctx parameters from protocol."""

    def check_parameter(p_label, p_details, task_label):
        add_p = False
        p_splitted = p_label.split("->")
        if len(p_splitted) == 2 and p_splitted[1] == task_label:
            p_label = p_splitted[0]
            add_p = True
        elif task_label in p_details.get("tasks", []):
            add_p = True
        return add_p, p_label

    task_dict = {}
    for p_label, p_details in protocol_dict["general_input"].items():
        add_p, p_label = check_parameter(p_label, p_details, task_label)
        if add_p:
            if p_details["aiida_node"]:
                task_dict[p_label] = create_aiida_node(p_details["value"])
            else:
                task_dict[p_label] = p_details["value"]
    for p_label, p_details in protocol_dict["user_input"].items():
        add_p, p_label = check_parameter(p_label, p_details, task_label)
        if add_p:
            input_path = p_label.split(".")
            input_set = True
            value = inputs
            for input_key in input_path:
                if input_key not in value:
                    input_set = False
                    break
                else:
                    value = value[input_key]
            if not input_set:
                continue
            if p_details["aiida_node"]:
                task_dict[p_label] = value
            else:
                task_dict[p_label] = obtain_value_from_aiida_node(value)
    ctx[task_label] = task_dict
