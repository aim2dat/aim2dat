"""Test the critic2 module of the io sub-package."""

# Standard library imports
import os

# Third party library imports
import pytest

# Internal library imports
from aim2dat.io import read_yaml_file, read_critic2_plane, read_critic2_stdout

cwd = os.path.dirname(__file__) + "/"


def test_stdout(nested_dict_comparison):
    """Test read_partial_charge function."""
    nested_dict_comparison(
        read_critic2_stdout(cwd + "critic2_stdout/critic2.out"),
        dict(read_yaml_file(cwd + "critic2_stdout/ref.yaml")),
    )


def test_stdout_error():
    """Test read_partial_charge error."""
    with pytest.raises(ValueError) as error:
        read_critic2_stdout(cwd + "critic2_stdout/critic2_error.out", raise_error=True)
    assert (
        str(error.value) == "Calculation did not finish properly, error message: "
        "'ERROR (fopen_read): error opening file: cube-ELECTRON_DENSITY-1_0.cube\n"
        "'. To obtain output, set `raise_error` to False."
    )


def test_planes(nested_dict_comparison):
    """Test read_critic2_plane function."""
    plane_ref = dict(read_yaml_file(cwd + "critic2_planes/ref.yaml"))
    plane_ref["coordinates"] = [tuple(val) for val in plane_ref["coordinates"]]
    plane = read_critic2_plane(cwd + "critic2_planes/rhodef")
    nested_dict_comparison(plane, plane_ref)
