"""Test calculate-functions of the Structure class."""

# Standard library imports
import os

# Third party library imports
import pytest

# Internal library imports:
from aim2dat.strct import Structure
from aim2dat.io.yaml import load_yaml_file

STRUCTURES_PATH = os.path.dirname(__file__) + "/structures/"
COORDINATION_PATH = os.path.dirname(__file__) + "/coordination/"


@pytest.mark.parametrize(
    "structure_label, method",
    [
        ("GaAs_216_conv", "minimum_distance"),
        ("GaAs_216_conv", "n_nearest_neighbours"),
        ("GaAs_216_conv", "econ"),
        ("GaAs_216_conv", "okeeffe"),
        ("Cs2Te_62_prim", "minimum_distance"),
        ("Cs2Te_62_prim", "n_nearest_neighbours"),
        ("Cs2Te_62_prim", "econ"),
        ("Cs2Te_62_prim", "okeeffe"),
        ("Cs2Te_62_prim", "voronoi_no_weights"),
        ("Cs2Te_62_prim", "voronoi_radius"),
        ("Cs2Te_62_prim", "voronoi_area"),
    ],
)
def test_cn_analysis(nested_dict_comparison, structure_label, method):
    """
    Test the different methods to determine the coordination number of atomic sites.
    """
    inputs = dict(load_yaml_file(STRUCTURES_PATH + structure_label + ".yaml"))
    ref = dict(load_yaml_file(COORDINATION_PATH + structure_label + "_" + method + ".yaml"))
    structure = Structure(**inputs)
    outputs = structure.calculate_coordination(**ref["parameters"])
    nested_dict_comparison(outputs, ref["ref"])
