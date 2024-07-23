"""Test determine-functions of the StructureCollection class."""

# Standard library imports
import os

# Third party library imports
import pytest

# Internal library imports
from aim2dat.strct import StructureCollection, StructureOperations
from aim2dat.strct.ext_analysis import (
    determine_molecular_fragments,
    create_graph,
)
from aim2dat.io.yaml import load_yaml_file

STRUCTURES_PATH = os.path.dirname(__file__) + "/structures/"
FRAG_ANALYSIS_PATH = os.path.dirname(__file__) + "/fragment_analysis/"
SPACE_GROUP_PATH = os.path.dirname(__file__) + "/space_group_analysis/"
POINT_GROUP_PATH = os.path.dirname(__file__) + "/point_group_analysis/"


@pytest.mark.parametrize(
    "structure, file_suffix, backend, excl_elements",
    [
        ("Benzene", "xyz", "ase", []),
        ("ZIF-8", "cif", "internal", ["Zn"]),
    ],
)
def test_determine_molecular_fragments(structure, file_suffix, backend, excl_elements):
    """Test determine_molecular_fragments function."""
    kwargs = {
        "exclude_elements": excl_elements,
        "cn_method": "econ",
        "econ_tolerance": 0.001,
        "econ_conv_threshold": 0.0001,
    }
    ref_outputs = load_yaml_file(FRAG_ANALYSIS_PATH + structure + ".yaml")
    strct_c = StructureCollection()
    strct_c.append_from_file(
        "test", STRUCTURES_PATH + structure + "." + file_suffix, backend=backend
    )
    strct_ops = StructureOperations(strct_c)
    fragments = strct_ops.perform_analysis(
        "test", method=determine_molecular_fragments, kwargs=kwargs
    )
    assert len(fragments) == len(ref_outputs), "Number of fragments differ."
    for frag, ref_frag in zip(fragments, ref_outputs):
        assert len(frag["elements"]) == len(ref_frag["elements"]), "Fragment size differs."
        for pos, pos_ref, el, el_ref, at_idx, at_idx_ref in zip(
            frag["positions"],
            ref_frag["positions"],
            frag["elements"],
            ref_frag["elements"],
            frag["site_indices"],
            ref_frag["site_indices"],
        ):
            assert all(
                abs(coord0 - coord1) < 1e-5 for coord0, coord1 in zip(pos, pos_ref)
            ), "Fragment positions differ."
            assert el == el_ref, "Fragment elements differ."
            assert at_idx == at_idx_ref, "Fragment indices differ."


@pytest.mark.parametrize(
    "structure, file_suffix, kwargs",
    [
        ("ZIF-8", "cif", {"symprec": 1e-5}),
        ("GaAs_216_prim", "yaml", {}),
        ("Cs2Te_62_prim", "yaml", {"symprec": 0.01, "return_sym_operations": True}),
    ],
)
def test_determine_space_group(nested_dict_comparison, structure, file_suffix, kwargs):
    """Test determine_space_group function."""
    ref_outputs = dict(load_yaml_file(SPACE_GROUP_PATH + structure + "_ref.yaml"))
    strct_c = StructureCollection()
    if file_suffix == "yaml":
        strct_c.append("test", **load_yaml_file(STRUCTURES_PATH + structure + "." + file_suffix))
    else:
        strct_c.append_from_file(
            "test", STRUCTURES_PATH + structure + "." + file_suffix, backend="internal"
        )
    strct_ops = StructureOperations(strct_c)
    sg_dict = strct_ops.determine_space_group("test", **kwargs)
    nested_dict_comparison(sg_dict, ref_outputs)
    assert strct_c["test"]["attributes"]["space_group"] == sg_dict["space_group"]["number"]


@pytest.mark.parametrize(
    "structure",
    [
        "CHBrClF",
        "HClO",
        "C4H6O6",
        "HCN",
        "C2H2",
        "N2H4",
        "H3PO4",
        "C2H2Cl2",
        "H2O",
        "NH3",
        "C2H4",
        "PCl5",
        "C5H5",
        "Benzene",
        "C7H7",
        "C8H8",
        "C3H4",
        "CH4",
        "SF6",
    ],
)
def test_point_group_determination(nested_dict_comparison, structure):
    """
    Test the method to determine the point group.
    """
    inputs = dict(load_yaml_file(STRUCTURES_PATH + structure + ".yaml"))
    pg = inputs["attributes"].pop("point_group")
    ref = dict(load_yaml_file(POINT_GROUP_PATH + structure + "_ref.yaml"))
    strct_c = StructureCollection()
    strct_c.append(structure, **inputs)
    strct_ops = StructureOperations(strct_c)
    nested_dict_comparison(strct_ops.determine_point_group(structure), ref)
    assert strct_c[structure]["attributes"]["point_group"] == pg


def test_determine_graph(nested_dict_comparison, create_structure_collection_object):
    """Test creating a graph from a structure."""
    strct_c, _ = create_structure_collection_object(["GaAs_216_prim"])
    strct_ops = StructureOperations(strct_c)
    nx_graph, _ = strct_ops.perform_analysis(key=0, method=create_graph)
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
