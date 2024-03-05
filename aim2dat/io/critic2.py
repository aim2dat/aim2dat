"""
Functions to read output-files of critic2.
"""

# Internal library imports
from aim2dat.aiida_workflows.critic2.parser_utils import (
    _parse_plane_file,
    _parse_stdout_file,
)


def read_stdout(file_name):
    """
    Read standard output file.

    Parameters
    ----------
    file_name : str
        Path to the file.

    Returns
    -------
    dict
        Results.
    """
    with open(file_name, "r") as stdout_file:
        return _parse_stdout_file(stdout_file)


def read_plane(file_name):
    """
    Read output plane file.

    Parameters
    ----------
    file_name : str
        Path to the file.

    Returns
    -------
    dict
        plane details.
    """
    with open(file_name, "r") as plane_file:
        return _parse_plane_file(plane_file)
