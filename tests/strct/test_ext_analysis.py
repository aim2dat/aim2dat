"""Test external structure analysis functions."""

# Standard library imports
import os

# Third party library imports
import pytest

# Internal library imports
from aim2dat.strct import Structure
from aim2dat.strct.ext_analysis import (
    calc_ffingerprint_order_p,
    calc_prdf,
    calc_warren_cowley_order_p,
    calc_molecular_fragments,
    calc_graph,
    calc_hydrogen_bonds,
    calc_planes,
)
from aim2dat.io import read_yaml_file

STRUCTURES_PATH = os.path.dirname(__file__) + "/structures/"
REF_PATH = os.path.dirname(__file__) + "/ext_analysis/"


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
            "get_statistics": True,
            "indices": None,
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
            "get_statistics": True,
            "indices": None,
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
    "structure,ref_order_p",
    [
        ("GaAs_216_prim", (742.1268, (495.7578, 495.7578))),
        (
            "Cs2Te_62_prim",
            (
                115.6221,
                (
                    155.4298,
                    113.9550,
                    114.2768,
                    153.4972,
                    153.4300,
                    113.8181,
                    113.7066,
                    153.9680,
                    62.4293,
                    62.2036,
                    62.2604,
                    62.6054,
                ),
            ),
        ),
    ],
)
def test_ffingerprint_order_parameters(structure, ref_order_p):
    """Test order F-fingerprint order parameters."""
    strct = Structure.from_file(STRUCTURES_PATH + structure + ".yaml", backend="internal")
    output = calc_ffingerprint_order_p(strct, r_max=20.0, delta_bin=0.005, sigma=10.0)
    assert (
        abs(output["order_p"] - ref_order_p[0]) < 1e-3
    ), f"Total order parameter of structure {structure} is wrong."
    assert len(output["site_order_p"]) == len(
        ref_order_p[1]
    ), f"Wrong number of site order parameters for structure {structure}."
    for idx, (site_order_p0, ref_site_order_p0) in enumerate(
        zip(output["site_order_p"], ref_order_p[1])
    ):
        assert (
            abs(site_order_p0 - ref_site_order_p0) < 1e-3
        ), f"Order parameter of site {idx} of structure {structure} is wrong."


@pytest.mark.parametrize("structure", ["Cs2Te_62_prim", "GaAs_216_conv", "GaAs_216_prim"])
def test_prdf_functions(structure, nested_dict_comparison):
    """Test partial radial distribution calculation."""
    ref = read_yaml_file(REF_PATH + "calc_prdf_" + structure + ".yaml")
    strct = Structure.from_file(STRUCTURES_PATH + structure + ".yaml", backend="internal")
    element_prdf, atomic_prdf = calc_prdf(strct, **ref["parameters"])
    assert len(element_prdf) == len(ref["element_prdf"]), "Wrong number of el-pairs."
    for el_pair, prdf in element_prdf.items():
        assert el_pair in ref["element_prdf"], f"Element pair {el_pair} not in reference output."
        assert all(
            [
                abs(val0 - ref["element_prdf"][el_pair][idx]) < 1.0e-5
                for idx, val0 in enumerate(prdf)
            ]
        ), f"Element prdf is wrong for element pair {el_pair}."
    assert len(atomic_prdf) == len(ref["atomic_prdf"]), "Wrong number of sites."
    for site_idx, site in enumerate(atomic_prdf):
        assert len(site) == len(
            ref["atomic_prdf"][site_idx]
        ), f"Wrong number of element pairs for site {site_idx}."
        for el_pair, prdf in site.items():
            assert (
                el_pair in ref["atomic_prdf"][site_idx]
            ), f"Element pair {el_pair} not in atomic prdf."
            assert all(
                [
                    abs(val0 - ref["atomic_prdf"][site_idx][el_pair][idx]) < 1.0e-5
                    for idx, val0 in enumerate(prdf)
                ]
            ), f"Atomic prdf is wrong for element pair {el_pair} at site {site_idx}."


@pytest.mark.parametrize(
    "system, file_suffix, backend",
    [
        ("Benzene", ".yaml", "internal"),
        ("ZIF-8", ".cif", "internal"),
    ],
)
def test_calc_molecular_fragments_function(structure_comparison, system, file_suffix, backend):
    """Test calc_molecular_fragments function."""
    kwargs, ref = read_yaml_file(REF_PATH + "calc_molecular_fragments_" + system + ".yaml")
    strct = Structure.from_file(STRUCTURES_PATH + system + file_suffix, backend=backend)
    fragments = calc_molecular_fragments(strct, **kwargs)
    for frag, frag_ref in zip(fragments, ref):
        structure_comparison(frag, Structure(**frag_ref))
        assert frag.site_attributes["parent_indices"] == tuple(
            frag_ref["site_attributes"]["parent_indices"]
        )


@pytest.mark.parametrize(
    "structure, r_max, max_shells",
    [
        ("Al_225_conv", 5.0, 3),
        ("NaCl_225_prim", 5.0, 3),
        ("Cs2Te_19_prim", 5.0, 2),
        ("Cs2Te_62_prim", 5.0, 2),
        ("GaAs_216_prim", 5.0, 3),
        ("GaAs_216_conv", 5.0, 3),
    ],
)
def test_warren_cowley_like_order_parameters(nested_dict_comparison, structure, r_max, max_shells):
    """Test calculation of warren cowley order parameters."""
    ref = dict(read_yaml_file(REF_PATH + "calc_warren_cowley_order_p_" + structure + ".yaml"))
    strct = Structure.from_file(STRUCTURES_PATH + structure + ".yaml", backend="internal")
    output = calc_warren_cowley_order_p(strct, r_max=r_max, max_shells=max_shells)
    nested_dict_comparison(output, ref)


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


@pytest.mark.parametrize(
    "structure,file_type", [("Imidazole", "xyz"), ("ZIF-8_complex", "xyz"), ("ZIF-8", "cif")]
)
def test_planes(structure, file_type):
    """
    Test finding planes.
    """
    ref = read_yaml_file(REF_PATH + "calc_planes_" + structure + ".yaml")
    strct = Structure.from_file(STRUCTURES_PATH + structure + "." + file_type)
    planes = calc_planes(strct, **ref["parameters"])
    for plane, ref_plane in zip(planes, ref["reference"]):
        assert len(plane["site_indices"]) == len(
            ref_plane["site_indices"]
        ), "Number of site indices do not match."
        for site_idx in ref_plane["site_indices"]:
            assert site_idx in plane["site_indices"], f"Site index {site_idx} not found."
        for vect, ref_vect in zip(plane["plane"], ref_plane["plane"]):
            for coord, ref_coord in zip(vect, ref_vect):
                assert abs(coord - ref_coord) < 1e-4, "Vectors do not match."
        for proj_pos, ref_proj_pos in zip(plane["proj_positions"], ref_plane["proj_positions"]):
            assert (
                proj_pos["label"] == ref_proj_pos["label"]
            ), "Wrong label for projected position."
            for coord in "xy":
                assert (
                    abs(proj_pos[coord] - ref_proj_pos[coord]) < 1e-4
                ), "Coordinates of projected positions is wrong."


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
