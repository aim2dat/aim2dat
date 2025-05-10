"""Test determine-functions of the StructureCollection class."""

# Standard library imports
import os

# Third party library imports
import pytest

# Internal library imports
from aim2dat.strct import StructureCollection, StructureOperations
from aim2dat.io import read_yaml_file

STRUCTURES_PATH = os.path.dirname(__file__) + "/structures/"
SPACE_GROUP_PATH = os.path.dirname(__file__) + "/space_group_analysis/"
POINT_GROUP_PATH = os.path.dirname(__file__) + "/point_group_analysis/"


@pytest.mark.parametrize(
    "structure, file_suffix, kwargs",
    [
        ("ZIF-8", "cif", {"symprec": 1e-5}),
        ("GaAs_216_prim", "yaml", {}),
        ("Cs2Te_62_prim", "yaml", {"symprec": 0.01, "return_sym_operations": True}),
    ],
)
def test_calc_space_group(nested_dict_comparison, structure, file_suffix, kwargs):
    """Test calc_space_group function."""
    ref_outputs = dict(read_yaml_file(SPACE_GROUP_PATH + structure + "_ref.yaml"))
    strct_c = StructureCollection()
    if file_suffix == "yaml":
        strct_c.append("test", **read_yaml_file(STRUCTURES_PATH + structure + "." + file_suffix))
    else:
        strct_c.append_from_file(
            "test", STRUCTURES_PATH + structure + "." + file_suffix, backend="internal"
        )
    strct_ops = StructureOperations(strct_c)
    sg_dict = strct_ops["test"].calc_space_group(**kwargs)
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
    inputs = dict(read_yaml_file(STRUCTURES_PATH + structure + ".yaml"))
    pg = inputs["attributes"].pop("point_group")
    ref = dict(read_yaml_file(POINT_GROUP_PATH + structure + "_ref.yaml"))
    strct_c = StructureCollection()
    strct_c.append(structure, **inputs)
    strct_ops = StructureOperations(strct_c)
    nested_dict_comparison(strct_ops[structure].calc_point_group(), ref)
    assert strct_c[structure]["attributes"]["point_group"] == pg
