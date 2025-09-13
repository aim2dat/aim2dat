"""Test external structure manipulation functions."""

# Standard library imports
import os
import numpy as np

# Third party library imports
import pytest

# Internal library imports
from aim2dat.strct import Structure, StructureCollection, StructureOperations
from aim2dat.strct.ext_manipulation import (
    add_structure_coord,
    add_structure_random,
    add_structure_position,
    rotate_structure,
    translate_structure,
    DistanceThresholdError,
)
from aim2dat.strct.ext_manipulation.utils import _build_distance_dict, _check_distances
from aim2dat.io import read_yaml_file

STRUCTURES_PATH = os.path.dirname(__file__) + "/structures/"
REF_PATH = os.path.dirname(__file__) + "/ext_manipulation/"


def test_build_distance_dict_fct(nested_dict_comparison):
    """Test _build_distance_dict function."""
    strct_1 = Structure.from_file(STRUCTURES_PATH + "MOF-5_prim.xsf")
    strct_2 = Structure.from_file(STRUCTURES_PATH + "Benzene.yaml", backend="internal")

    dist_dict, min_dist = _build_distance_dict(None, strct_1, strct_2)
    assert dist_dict is None
    assert min_dist == 0.0

    dist_dict, min_dist = _build_distance_dict(0.3, strct_1, strct_2)
    nested_dict_comparison(
        dist_dict,
        {
            ("Zn", "Zn"): [0.3, None],
            ("O", "Zn"): [0.3, None],
            ("H", "Zn"): [0.3, None],
            ("C", "Zn"): [0.3, None],
            ("O", "O"): [0.3, None],
            ("H", "O"): [0.3, None],
            ("C", "O"): [0.3, None],
            ("H", "H"): [0.3, None],
            ("C", "H"): [0.3, None],
            ("C", "C"): [0.3, None],
        },
    )
    assert min_dist == 0.3

    dist_dict, min_dist = _build_distance_dict("covalent+10", strct_1, strct_2)
    nested_dict_comparison(
        dist_dict,
        {
            ("C", "C"): [1.672, None],
            ("C", "O"): [1.562, None],
            ("C", "Zn"): [2.178, None],
            ("C", "H"): [1.177, None],
            ("O", "O"): [1.452, None],
            ("O", "Zn"): [2.068, None],
            ("H", "O"): [1.067, None],
            ("Zn", "Zn"): [2.684, None],
            ("H", "Zn"): [1.683, None],
            ("H", "H"): [0.682, None],
        },
    )
    assert min_dist == 0.682

    dist_dict, min_dist = _build_distance_dict("covalent-20", strct_1, strct_2)
    nested_dict_comparison(
        dist_dict,
        {
            ("O", "O"): [1.056, None],
            ("C", "O"): [1.136, None],
            ("H", "O"): [0.776, None],
            ("O", "Zn"): [1.504, None],
            ("C", "C"): [1.216, None],
            ("C", "H"): [0.856, None],
            ("C", "Zn"): [1.584, None],
            ("H", "H"): [0.496, None],
            ("H", "Zn"): [1.224, None],
            ("Zn", "Zn"): [1.952, None],
        },
    )
    assert min_dist == 0.496

    dist_dict, min_dist = _build_distance_dict([0.5, 1.0], strct_1)
    assert dist_dict == {
        ("O", "O"): [0.5, 1.0],
        ("H", "O"): [0.5, 1.0],
        ("O", "Zn"): [0.5, 1.0],
        ("C", "O"): [0.5, 1.0],
        ("H", "H"): [0.5, 1.0],
        ("H", "Zn"): [0.5, 1.0],
        ("C", "H"): [0.5, 1.0],
        ("Zn", "Zn"): [0.5, 1.0],
        ("C", "Zn"): [0.5, 1.0],
        ("C", "C"): [0.5, 1.0],
    }
    assert min_dist == 0.5

    dist_dict, min_dist = _build_distance_dict({("Zn", "Zn"): 0.9, (0, 5): [0.3, 2.4]}, strct_1)
    assert dist_dict == {("Zn", "Zn"): [0.9, None], (0, 5): [0.3, 2.4]}
    assert min_dist == 0.3

    with pytest.raises(ValueError) as error:
        _build_distance_dict({("Zn", "Zn", "O"): 0.9, (0, 5): [0.3, 2.4]}, strct_1)
    assert (
        str(error.value)
        == "`dist_threshold` needs to have keys with length 2 containing site "
        + "indices or element symbols."
    )

    with pytest.raises(ValueError) as error:
        _build_distance_dict({(0.5, 1): 0.9, (0, 5): [0.3, 2.4]}, strct_1)
    assert (
        str(error.value)
        == "`dist_threshold` needs to have keys of type List[str/int] containing "
        + "site indices or element symbols."
    )
    with pytest.raises(TypeError) as error:
        _build_distance_dict(set(["test", "test2"]), strct_1)
    assert (
        str(error.value)
        == "`dist_threshold` needs to be of type int/float/list/tuple/dict or None."
    )


def test_check_distance_fct():
    """Test _check_distances function"""
    strct_1 = Structure.from_file(STRUCTURES_PATH + "MOF-303_3xH2O_flawed.xsf")
    with pytest.raises(DistanceThresholdError) as error:
        _check_distances(
            strct_1, [*range(131)] + [134, 135, 136], {("N", "O"): [0.8, 2.5]}, None, False
        )
    assert str(error.value) == "Atoms 131 and 50 are too far from each other."


def test_add_structure_coord_index_mismatch():
    """Test add_coord_structure when host/guest index is too large."""
    inputs = dict(read_yaml_file(STRUCTURES_PATH + "NH3.yaml"))
    new_strct = add_structure_coord(
        Structure(**inputs),
        host_indices=[1, 2, 3],
        guest_indices=1,
        guest_structure="H",
    )
    assert len(new_strct) == 4
    assert new_strct.label is None


def test_add_structure_coord(structure_comparison):
    """Test add functional group function."""
    inputs = dict(read_yaml_file(STRUCTURES_PATH + "Sc2BDC3.yaml"))
    inputs["kinds"] = ["kind1"] + [None] * (len(inputs["elements"]) - 1)
    ref_p = read_yaml_file(REF_PATH + "add_structure_coord_Sc2BDC3.yaml")
    strct_collect = StructureCollection()
    strct_collect.append(**inputs, label="test")
    strct_ops = StructureOperations(strct_collect)
    new_strct = strct_ops[0].perform_manipulation(
        method=add_structure_coord,
        kwargs={
            "wrap": True,
            "host_indices": 37,
            "guest_structure": "H",
            "bond_length": 1.0,
            "change_label": False,
            "method": "minimum_distance",
            "min_dist_delta": 0.2,
        },
    )
    new_strct = new_strct.perform_manipulation(
        method=add_structure_coord,
        kwargs={
            "wrap": True,
            "host_indices": 39,
            "guest_structure": "CH3",
            "bond_length": 1.25,
            "change_label": False,
            "method": "minimum_distance",
            "min_dist_delta": 0.2,
        },
    )
    new_strct = new_strct.perform_manipulation(
        method=add_structure_coord,
        kwargs={
            "host_indices": 41,
            "guest_structure": "COOH",
            "bond_length": 1.5,
            "change_label": False,
            "dist_threshold": None,
            "method": "minimum_distance",
            "min_dist_delta": 0.2,
        },
    )
    new_strct = new_strct.perform_manipulation(
        method=add_structure_coord,
        kwargs={
            "host_indices": 42,
            "guest_structure": "NH2",
            "change_label": False,
            "method": "minimum_distance",
            "min_dist_delta": 0.2,
        },
    )
    new_strct = new_strct.perform_manipulation(
        method=add_structure_coord,
        kwargs={
            "host_indices": 62,
            "guest_structure": "NO2",
            "change_label": False,
            "method": "minimum_distance",
            "min_dist_delta": 0.2,
        },
    )
    new_strct = new_strct.perform_manipulation(
        method=add_structure_coord,
        kwargs={
            "host_indices": 74,
            "guest_structure": "OH",
            "change_label": False,
            "dist_threshold": None,
            "method": "minimum_distance",
            "min_dist_delta": 0.2,
        },
    )
    structure_comparison(new_strct, ref_p)


def test_add_structure_coord_planar(structure_comparison):
    """Test add_structure_coord method for planar geometries."""
    strct = Structure.from_file(STRUCTURES_PATH + "MOF-5_prim.xsf")
    new_strct = add_structure_coord(
        strct,
        guest_structure="H2O",
        min_dist_delta=0.2,
        host_indices=54,
        dist_threshold=None,
        bond_length=0.2,
    )
    ref_p = read_yaml_file(REF_PATH + "add_structure_coord_MOF-5_prim_planar.yaml")
    structure_comparison(new_strct, ref_p)


def test_add_structure_coord_rotation(structure_comparison):
    """
    Test add_structure_coord method for rotation of molecule around itself
    and around the host geometries.
    """
    strct = Structure.from_file(STRUCTURES_PATH + "MOF-5_prim.xsf")
    new_strct = add_structure_coord(
        strct,
        host_indices=[10, 24, 30],
        guest_indices=[1, 2],
        guest_structure="H2O",
        rotate_guest=True,
        bond_length=3,
        dist_threshold=2,
    )
    ref_p = read_yaml_file(REF_PATH + "add_structure_coord_MOF-5_prim_rotation.yaml")
    structure_comparison(new_strct, ref_p)
    new_strct = add_structure_coord(
        strct,
        host_indices=0,
        guest_indices=0,
        guest_structure="H2",
        bond_length=3,
        dist_threshold=2.9,
        constrain_steps=30,
    )
    ref_p = read_yaml_file(REF_PATH + "add_structure_coord_MOF-5_prim_constrain.yaml")
    structure_comparison(new_strct, ref_p)


def test_add_structure_coord_molecules(structure_comparison):
    """Test add_structure_coord method."""
    inputs = dict(read_yaml_file(STRUCTURES_PATH + "NH3.yaml"))
    new_strct = add_structure_coord(
        Structure(**inputs),
        host_indices=[1, 2, 3],
        guest_structure="OH",
        min_dist_delta=0.5,
        bond_length=1.0,
        dist_threshold={(1, 1): 0.9},
    )
    ref_p = read_yaml_file(REF_PATH + "add_structure_coord_NH3-OH.yaml")
    structure_comparison(new_strct, ref_p)


def test_add_structure_coord_molecules_2(structure_comparison):
    """Test add_structure_coord method."""
    inputs = dict(read_yaml_file(STRUCTURES_PATH + "NH3.yaml"))
    new_strct = add_structure_coord(
        Structure(**inputs),
        host_indices=[1, 2, 3],
        guest_indices=0,
        guest_structure="H2O",
        min_dist_delta=0.1,
        bond_length=1.5,
        dist_threshold={(2, 1): 1.5},
    )
    ref_p = read_yaml_file(REF_PATH + "add_structure_coord_NH3-H2O.yaml")
    structure_comparison(new_strct, ref_p)


# def test_add_structure_coord_single_atom():
#     strct = Structure.from_file(STRUCTURES_PATH + "Benzene.yaml", backend="internal")
#     print(strct)


def test_add_structure_random_molecule(structure_comparison):
    """Test add_structure_random method for molecules."""
    inputs = dict(read_yaml_file(STRUCTURES_PATH + "NH3.yaml"))
    new_strct = add_structure_random(
        Structure(**inputs),
        guest_structure="H2O",
        random_seed=44,
        dist_threshold=3.4,
    )
    new_strct = add_structure_random(
        new_strct,
        guest_structure="OH",
        random_seed=44,
        dist_threshold=1.0,
    )
    ref_p = read_yaml_file(REF_PATH + "add_structure_random_NH3-H2O-OH.yaml")
    structure_comparison(new_strct, ref_p)


def test_add_structure_random_crystal(structure_comparison):
    """Test add_structure_random method for a crystal."""
    inputs = dict(read_yaml_file(STRUCTURES_PATH + "NaCl_225_conv.yaml"))
    new_strct = add_structure_random(
        Structure(**inputs),
        guest_structure="H",
        random_nrs=[
            0.12256550967387159,
            0.258113074772773,
            0.4057707278211311,
            0.969183948144422,
            0.1623171247855668,
            0.8572936735567716,
            0.16304527107106548,
        ],
        max_tries=1,
        dist_threshold=1.0,
    )
    ref_p = read_yaml_file(REF_PATH + "add_structure_random_NaCl_225_conv-H.yaml")
    structure_comparison(new_strct, ref_p)


def test_add_structure_random_molecules_error():
    """Test add_structure_random method errors."""
    inputs = dict(read_yaml_file(STRUCTURES_PATH + "GaAs_216_prim.yaml"))
    with pytest.raises(DistanceThresholdError) as error:
        add_structure_random(
            Structure(**inputs),
            guest_structure="H2O",
            random_seed=44,
            dist_threshold=10.0,
            max_tries=3,
        )
    assert (
        str(error.value)
        == "Could not add guest structure, host structure seems to be too aggregated."
    )


def test_add_structure_position(structure_comparison):
    """Test add_structure_random method for a crystal."""
    inputs = [
        dict(read_yaml_file(STRUCTURES_PATH + "PBI3.yaml")),
        dict(read_yaml_file(STRUCTURES_PATH + "CN2H5.yaml")),
    ]
    new_strct = add_structure_position(
        Structure(**inputs[0]),
        position=[2.88759377, 3.244215, 3.25149],
        guest_structure=Structure(**inputs[1]),
    )
    ref_p = read_yaml_file(REF_PATH + "add_structure_position_PBI3+CN2H5.yaml")
    structure_comparison(new_strct, ref_p)

    with pytest.raises(DistanceThresholdError) as error:
        new_strct = add_structure_position(
            Structure(**inputs[0]),
            position=[2.88759377, 3.244215, 3.25149],
            guest_structure=Structure(**inputs[1]),
            dist_threshold=10.0,
        )
    assert str(error.value) == "Atoms 3 and 7 are too close to each other."


def test_rotate_structure(structure_comparison):
    """Test rotate_structure method for a crystal."""
    inputs = (
        dict(read_yaml_file(REF_PATH + "add_structure_position_PBI3+CN2H5.yaml")),
        Structure.from_file(STRUCTURES_PATH + "MOF-5_prim.xsf"),
    )
    new_strct = rotate_structure(
        Structure(**inputs[0]),
        angles=[90, 0, 0],
        site_indices=[4, 5, 6, 7, 8, 9, 10, 11],
    )
    ref_p = read_yaml_file(REF_PATH + "rotate_structure_PBI3+CN2H5.yaml")
    structure_comparison(new_strct, ref_p)

    new_strct = rotate_structure(
        Structure(**inputs[1]),
        angles=90,
        site_indices=[44, 56, 76, 81, 57, 45, 77, 80, 104, 101, 105, 100],
        origin=[2.90097537349613, 6.52064699916667, 6.52064699916667],
        vector=[1.00000000e00, 2.44932357e-15, 2.44932357e-15],
    )
    ref_p = read_yaml_file(REF_PATH + "rotate_structure_MOF-5_prim.yaml")
    structure_comparison(new_strct, ref_p)


def test_translate_structure(structure_comparison):
    """Test translate_structure method for a crystal."""
    vector = np.array([1.0, 0.4, 0.6])
    strct = Structure.from_file(STRUCTURES_PATH + "Benzene.yaml", backend="internal")
    strct.label = "Test"
    ref_strct = strct.copy()
    ref_strct.label = "Test_translated-[1.0, 0.4, 0.6]"
    ref_strct.set_positions([np.array(pos) + vector for pos in ref_strct.positions])
    new_strct = translate_structure(strct, vector=vector, change_label=True)
    structure_comparison(new_strct, ref_strct)
