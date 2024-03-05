"""
Input and output operations for yaml-files formated according to YAML 1.2 based on ruamel.yaml.
"""

# Third party library imports
import ruamel.yaml


def load_yaml_file(file_path, typ="safe"):
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
    with open(file_path, "r") as file:
        yaml = ruamel.yaml.YAML(typ=typ, pure=True)
        data = yaml.load(file)
    return data


def store_in_yaml_file(file_name, content):
    """
    Load a yaml-file and returns the content.

    Parameters
    ----------
    file_name : str
        Path to the yaml-file.
    content :
        Content of the file.
    """
    yaml = ruamel.yaml.YAML()
    yaml.version = (1, 2)
    yaml.default_flow_style = None
    with open(file_name, "w") as open_f:
        yaml.dump(content, open_f)
