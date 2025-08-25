"""
Input and output operations for yaml-files formated according to YAML 1.2 based on ruamel.yaml.
"""

# Standard library imports
from typing import Any

# Third party library imports
import ruamel.yaml

# Internal library imports
from aim2dat.io.utils import read_structure


@read_structure(r".*\.ya?ml")
def read_yaml_file(file_path: str, typ: str = "safe"):
    """
    Load a yaml-file and returns the content.

    Parameters
    ----------
    file_path : str
        Path to the yaml-file.
    typ : str (optional)
        Typ used to load the yaml-file.

    Returns
    -------
    data :
        Content of the file.
    """
    with open(file_path, "r") as f_obj:
        yaml = ruamel.yaml.YAML(typ=typ, pure=True)
        data = yaml.load(f_obj)
    return data


def write_yaml_file(file_path: str, content):
    """
    Write content to a yaml-file.

    Parameters
    ----------
    file_path : str
        Path to the yaml-file.
    content :
        Content of the file.
    """
    yaml = ruamel.yaml.YAML()
    yaml.version = (1, 2)
    yaml.default_flow_style = None
    with open(file_path, "w") as f_obj:
        yaml.dump(content, f_obj)


def load_yaml_file(file_path: str, typ: str = "safe"):
    """
    Load a yaml-file and returns the content.

    Notes
    -----
        This function is deprecated and will be removed, please use `aim2dat.io.read_yaml_file`
        instead.

    Parameters
    ----------
    file_path : str
        Path to the yaml-file.
    typ : str (optional)
        Typ used to load the yaml-file.

    Returns
    -------
    data :
        Content of the file.
    """
    from warnings import warn

    warn(
        "This function will be removed, please use `aim2dat.io.read_yaml_file` instead.",
        DeprecationWarning,
        2,
    )
    return read_yaml_file(file_path=file_path, typ=typ)


def store_in_yaml_file(file_name: str, content: Any):
    """
    Write conttent to a yaml-file.

    Notes
    -----
        This function is deprecated and will be removed, please use `aim2dat.io.write_yaml_file`
        instead.

    Parameters
    ----------
    file_name : str
        Path to the yaml-file.
    content :
        Content of the file.
    """
    from warnings import warn

    warn(
        "This function will be removed, please use `aim2dat.io.write_yaml_file` instead.",
        DeprecationWarning,
        2,
    )
    write_yaml_file(file_path=file_name, content=content)
