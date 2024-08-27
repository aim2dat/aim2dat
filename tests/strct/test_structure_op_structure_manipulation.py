"""Test structure manipulation funcitons via the StructureOperations class."""

# Standard library imports
import os

# Third party library imports
import pytest

# Internal library imports
from aim2dat.strct import StructureOperations, StructureCollection
from aim2dat.strct.ext_manipulation import add_structure_coord

from aim2dat.io.yaml import load_yaml_file

STRUCTURES_PATH = os.path.dirname(__file__) + "/structures/"
STRUCTURE_MANIPULATION_PATH = os.path.dirname(__file__) + "/structure_manipulation/"


@pytest.mark.parametrize("structure", ["Benzene"])
def test_delete_atoms(structure_comparison, structure):
    """Test delete atoms method."""
    inputs = dict(load_yaml_file(STRUCTURES_PATH + structure + ".yaml"))
    ref_p = load_yaml_file(STRUCTURE_MANIPULATION_PATH + structure + "_ref.yaml")
    ref_p["structure"]["label"] = structure

    strct_collect = StructureCollection()
    strct_collect.append(structure, **inputs)
    strct_ops = StructureOperations(strct_collect, append_to_coll=False)
    new_strct = strct_ops.delete_atoms(structure, **ref_p["function_args"], change_label=True)
    ref_p["structure"]["label"] += "_del"
    structure_comparison(new_strct, ref_p["structure"])


@pytest.mark.parametrize("structure", ["Cs2Te_62_prim", "GaAs_216_prim", "Cs2Te_19_prim_kinds"])
def test_element_substitution(structure_comparison, structure):
    """Test element substitution method."""
    inputs = dict(load_yaml_file(STRUCTURES_PATH + structure + ".yaml"))
    inputs["label"] = structure
    inputs2 = dict(load_yaml_file(STRUCTURES_PATH + "Al_225_conv.yaml"))
    inputs2["label"] = "Al_test"
    ref_p = load_yaml_file(STRUCTURE_MANIPULATION_PATH + structure + "_ref.yaml")
    strct_collect = StructureCollection()
    strct_collect.append(**inputs)
    strct_collect.append(**inputs2)
    strct_ops = StructureOperations(strct_collect, append_to_coll=True)
    elements = ref_p["function_args"]["elements"]
    elements = elements if isinstance(elements[0], (list, tuple)) else [elements]
    strct_ops.substitute_elements([0], **ref_p["function_args"], change_label=True)
    assert len(strct_ops.structures) == 3
    structure_comparison(strct_ops.structures[0], inputs)
    structure_comparison(strct_ops.structures[1], inputs2)
    structure_comparison(strct_ops.structures[2], ref_p["structure"])


def test_add_structure_coord(structure_comparison):
    """Test add functional group function."""
    inputs = dict(load_yaml_file(STRUCTURES_PATH + "Sc2BDC3.yaml"))
    inputs["kinds"] = ["kind1"] + [None] * (len(inputs["elements"]) - 1)
    ref_p = load_yaml_file(STRUCTURE_MANIPULATION_PATH + "Sc2BDC3_ref.yaml")
    ref_p["label"] = "test"
    strct_collect = StructureCollection()
    strct_collect.append(**inputs, label="test")
    strct_ops = StructureOperations(strct_collect, append_to_coll=True)
    strct_ops.perform_manipulation(
        0,
        method=add_structure_coord,
        kwargs={
            "wrap": True,
            "host_indices": 37,
            "guest_structure": "H",
            "guest_dir": [1.0, 0.0, 0.0],
            "bond_length": 1.0,
            "change_label": False,
        },
    )
    strct_ops.perform_manipulation(
        0,
        method=add_structure_coord,
        kwargs={
            "wrap": True,
            "host_indices": 39,
            "guest_structure": "CH3",
            "guest_dir": [1.0, 0.0, 0.0],
            "bond_length": 1.1,
            "change_label": False,
        },
    )
    strct_ops.perform_manipulation(
        0,
        method=add_structure_coord,
        kwargs={
            "host_indices": 41,
            "guest_structure": "COOH",
            "guest_dir": [1.0, 0.0, 0.0],
            "change_label": False,
        },
    )
    strct_ops.perform_manipulation(
        0,
        method=add_structure_coord,
        kwargs={
            "host_indices": 42,
            "guest_structure": "NH2",
            "guest_dir": [1.0, 0.0, 0.0],
            "change_label": False,
        },
    )
    strct_ops.perform_manipulation(
        0,
        method=add_structure_coord,
        kwargs={
            "host_indices": 62,
            "guest_structure": "NO2",
            "guest_dir": [1.0, 0.0, 0.0],
            "change_label": False,
        },
    )
    strct_ops.perform_manipulation(
        0,
        method=add_structure_coord,
        kwargs={
            "host_indices": 74,
            "guest_structure": "OH",
            "guest_dir": [1.0, 0.0, 0.0],
            "change_label": False,
            "dist_threshold": None,
        },
    )
    strct_ops.structures[0].set_positions(strct_ops.structures[0].positions, wrap=True)
    structure_comparison(strct_ops.structures[0], ref_p)


@pytest.mark.parametrize("new_label", ["GaAs_216_prim", "GaAs_216_prim_scaled-0.7"])
def test_scale_unit_cell(structure_comparison, new_label):
    """Test scale unit cell function."""
    inputs = dict(load_yaml_file(STRUCTURES_PATH + "GaAs_216_prim.yaml"))
    ref = dict(
        load_yaml_file(STRUCTURE_MANIPULATION_PATH + "GaAs_216_prim_scale_unit_cell_ref.yaml")
    )
    ref["structure"]["label"] = new_label
    strct_c = StructureCollection()
    strct_c.append("GaAs_216_prim", **inputs)
    strct_ops = StructureOperations(strct_c, append_to_coll=True)
    strct_ops.scale_unit_cell(
        "GaAs_216_prim", **ref["function_args"], change_label="scaled" in new_label
    )
    structure_comparison(strct_c[new_label], ref["structure"])
