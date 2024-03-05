"""Additional functions to test AiiDA-related parts of the package."""

# Third party library imports
import aiida.orm as aiida_orm
from aiida.common import LinkType


def generate_work_function_node(entry_point, process_state, inputs=None):
    """Generate work function node."""
    process = aiida_orm.WorkFunctionNode(process_type=entry_point)
    process.set_process_state(process_state)
    if inputs:
        for input_label, input_value in inputs.items():
            process.base.links.add_incoming(
                input_value, LinkType.INPUT_WORK, link_label=input_label
            )
    return process
