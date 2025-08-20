"""Test external structure analysis functions."""

# Standard library imports
import os

# Third party library imports
import pytest

# Internal library imports
from aim2dat.strct import Structure
from aim2dat.strct.ext_analysis import calc_molecular_fragments, calc_graph, calc_hydrogen_bonds
from aim2dat.io import read_yaml_file

STRUCTURES_PATH = os.path.dirname(__file__) + "/structures/"
FRAG_PATH = os.path.dirname(__file__) + "/fragment_analysis/"


def test_func_args_extraction():
    """Test correct extraction of function arguments done by the decorator."""
    strct = Structure(**read_yaml_file(STRUCTURES_PATH + "GaAs_216_prim.yaml"))
    calc_graph(strct, method="n_nearest_neighbours")
    assert strct._function_args == {
        "coordination": {
            "r_max": 10.0,
            "method": "n_nearest_neighbours",
            "min_dist_delta": 0.1,
            "n_nearest_neighbours": 5,
            "radius_type": "chen_manz",
            "atomic_radius_delta": 0.0,
            "econ_tolerance": 0.5,
            "econ_conv_threshold": 0.001,
            "voronoi_weight_type": "rel_solid_angle",
            "voronoi_weight_threshold": 0.5,
            "okeeffe_weight_threshold": 0.5,
        },
        "graph": {
            "method": "n_nearest_neighbours",
            "get_graphviz_graph": False,
            "graphviz_engine": "circo",
            "graphviz_edge_rank_colors": ["blue", "red", "green", "orange", "darkblue"],
        },
    }
    calc_graph(strct)
    assert strct._function_args == {
        "coordination": {
            "r_max": 10.0,
            "method": "atomic_radius",
            "min_dist_delta": 0.1,
            "n_nearest_neighbours": 5,
            "radius_type": "chen_manz",
            "atomic_radius_delta": 0.0,
            "econ_tolerance": 0.5,
            "econ_conv_threshold": 0.001,
            "voronoi_weight_type": "rel_solid_angle",
            "voronoi_weight_threshold": 0.5,
            "okeeffe_weight_threshold": 0.5,
        },
        "graph": {
            "get_graphviz_graph": False,
            "graphviz_engine": "circo",
            "graphviz_edge_rank_colors": ["blue", "red", "green", "orange", "darkblue"],
        },
    }


@pytest.mark.parametrize(
    "system, file_suffix, backend",
    [
        ("Benzene", ".xyz", "ase"),
        ("ZIF-8", ".cif", "internal"),
    ],
)
def test_calc_molecular_fragments_function(structure_comparison, system, file_suffix, backend):
    """Test calc_molecular_fragments function."""
    kwargs, ref = read_yaml_file(FRAG_PATH + system + ".yaml")
    strct = Structure.from_file(STRUCTURES_PATH + system + file_suffix, backend=backend)
    fragments = calc_molecular_fragments(strct, **kwargs)
    for frag, frag_ref in zip(fragments, ref):
        structure_comparison(frag, Structure(**frag_ref))
        assert frag.site_attributes["parent_indices"] == tuple(
            frag_ref["site_attributes"]["parent_indices"]
        )


def test_calc_graph(nested_dict_comparison):
    """Test creating a graph from a structure."""
    strct = Structure(**read_yaml_file(STRUCTURES_PATH + "GaAs_216_prim.yaml"))
    nx_graph, graphviz_graph = calc_graph(
        strct, get_graphviz_graph=True, method="minimum_distance"
    )
    assert list(nx_graph.edges) == [
        (0, 1, 0),
        (0, 1, 1),
        (0, 1, 2),
        (0, 1, 3),
        (1, 0, 0),
        (1, 0, 1),
        (1, 0, 2),
        (1, 0, 3),
    ]
    assert list(nx_graph.nodes) == [0, 1]
    graphviz_ref = (
        "digraph {\n",
        "\tGa0\n",
        "\tAs1\n",
        "\tGa0 -> As1 [color=blue]\n",
        "\tGa0 -> As1 [color=blue]\n",
        "\tGa0 -> As1 [color=blue]\n",
        "\tGa0 -> As1 [color=blue]\n",
        "\tAs1 -> Ga0 [color=blue]\n",
        "\tAs1 -> Ga0 [color=blue]\n",
        "\tAs1 -> Ga0 [color=blue]\n",
        "\tAs1 -> Ga0 [color=blue]\n",
        "}\n",
    )
    for idx, line in enumerate(graphviz_graph):
        assert line == graphviz_ref[idx]


def test_calc_hydrogen_bonds():
    """Test hydrogen bond analysis."""
    strct = Structure.from_file(STRUCTURES_PATH + "MOF-303_30xH2O.xsf")
    with pytest.raises(ValueError) as error:
        calc_hydrogen_bonds(strct, scheme="test")
    assert (
        str(error.value)
        == "`scheme` 'test' is not supported. Valid options are: ['baker_hubbard']."
    )

    hbonds = calc_hydrogen_bonds(
        strct, host_elements="O", index_constraint=[104, 128, 129, 130, 131, 132, 148, 200, 201]
    )
    assert hbonds == ((128, 201, 200), (200, 132, 131))
