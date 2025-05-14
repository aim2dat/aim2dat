"""Utils for workflow builder objects."""

# Standard library imports
import os
import re

# Internal library imports
from aim2dat.io import read_yaml_file


def _load_protocol(inp_protocol, folder_path):
    """
    Load the protocol dictionary either directly by passing a dictionary or collecting it from
    the yaml-file.
    """
    if isinstance(inp_protocol, dict):
        protocol_dict = inp_protocol
    elif isinstance(inp_protocol, str):
        protocol_pattern = re.compile(r"^([\s\S]*)?\_v(\d+(\.\d+)?)?")
        protocol = inp_protocol.lower()
        match = protocol_pattern.match(protocol)
        if match is not None:
            file_path = folder_path + f"{match.groups()[0]}_v{float(match.groups()[1])}.yaml"
        else:
            file_path = None
            latest_version = 0.0
            for file in os.listdir(folder_path):
                if os.path.isfile(os.path.join(folder_path, file)) and protocol in file:
                    version = float(protocol_pattern.match(file).groups()[1])
                    if version > latest_version:
                        latest_version = version
                        file_path = folder_path + file
            if file_path is None:
                raise ValueError(f"No version of protocol {inp_protocol} could be found.")
        protocol_dict = read_yaml_file(file_path)
    else:
        raise TypeError("protocol needs to be of type `str` or `dict`.")
    return protocol_dict


def _wf_states_color_map(val):
    """Color map for workflow states pandas data frame."""
    if isinstance(val, str):
        if val == "completed":
            return "color: green"
        elif "failed" in val:
            return "color: red"
        elif val == "missing deps.":
            return "color: orange"
    return None
