"""Test the critic2 module of the io sub-package."""

# Standard library imports
import os

# Internal library imports
from aim2dat.io.yaml import load_yaml_file
from aim2dat.io.critic2 import read_plane, read_stdout

cwd = os.path.dirname(__file__) + "/"


def test_stdout(nested_dict_comparison):
    """Test read_partial_charge function."""
    nested_dict_comparison(
        read_stdout(cwd + "critic2_stdout/critic2.out"),
        dict(load_yaml_file(cwd + "critic2_stdout/ref.yaml")),
    )


def test_planes(nested_dict_comparison):
    """Test read_plane function."""
    plane_ref = dict(load_yaml_file(cwd + "critic2_planes/ref.yaml"))
    plane_ref["coordinates"] = [tuple(val) for val in plane_ref["coordinates"]]
    plane = read_plane(cwd + "critic2_planes/rhodef")
    nested_dict_comparison(plane, plane_ref)
