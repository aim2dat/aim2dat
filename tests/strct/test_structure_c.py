"""Test the StructureCollection class."""

# Standard library imports
import os

# Third party library imports
import pytest
import pandas as pd

# Internal library imports
from aim2dat.strct import Structure, StructureCollection, StructureOperations
from aim2dat.io import read_yaml_file
from aim2dat.ext_interfaces.pandas import _turn_dict_into_pandas_df

# from aim2dat.ext_interfaces.ase_atoms import _create_atoms_from_structure

STRUCTURES_PATH = os.path.dirname(__file__) + "/structures/"
EXT_INTERFACES_PATH = os.path.dirname(__file__) + "/ext_interfaces/"


def test_basic_features(create_structure_collection_object, structure_comparison):
    """Test basic features of the StructureCollection class."""
    strct_list_1 = ["Benzene", "SF6", "H2O", "H3PO4"]
    strct_list_2 = ["ZIF-8", "GaAs_216_prim", "Cs2Te_19_prim"]
    strct1_collect, structures_1 = create_structure_collection_object(
        strct_list_1, label_prefix="t1_"
    )
    strct2_collect, structures_2 = create_structure_collection_object(
        strct_list_2, label_prefix="t2_"
    )
    with pytest.raises(TypeError) as error:
        strct_collect = strct1_collect + 0
    assert str(error.value) == "Can only add objects of type StructureCollection."
    strct_collect = strct1_collect + strct2_collect
    strct1_collect += strct2_collect

    assert (
        len(strct1_collect) == 7 and len(strct2_collect) == 3 and len(strct_collect) == 7
    ), "Lengths are wrong."
    assert strct_collect.labels == [
        "t1_Benzene",
        "t1_SF6",
        "t1_H2O",
        "t1_H3PO4",
        "t2_ZIF-8",
        "t2_GaAs_216_prim",
        "t2_Cs2Te_19_prim",
    ], "Wrong labels."

    strct1_collect["t1_H3PO4"] = structures_1[2]
    structures_1[2]["label"] = "t1_H3PO4"
    structure_comparison(strct1_collect["t1_H3PO4"], structures_1[2])
    del structures_1[2]["label"]

    with pytest.raises(TypeError) as error:
        strct1_collect[2.0] = structures_1[2]
    assert str(error.value) == "`key` needs to be of type: str, int, slice, tuple or list."
    with pytest.raises(ValueError) as error:
        strct1_collect[10] = structures_1[2]
    assert str(error.value) == "Index out of range (10 >= 7)."
    with pytest.raises(TypeError) as error:
        structures_1[2]["label"] = 10.0
        strct1_collect[3] = structures_1[2]
    assert str(error.value) == "`label` needs to be of type str."
    with pytest.raises(ValueError) as error:
        structures_1[2]["label"] = "t1_Benzene"
        strct1_collect.append(**structures_1[2])
    assert str(error.value) == "Label 't1_Benzene' already used."
    assert strct1_collect.get_structure(10.0) is None
    del structures_1[2]["label"]
    strct1_collect[3] = structures_1[2]
    structures_1[2]["label"] = "t1_H3PO4"
    structure_comparison(strct1_collect["t1_H3PO4"], structures_1[2])
    del structures_1[2]["label"]

    structures_1[2]["label"] = "t3_0"
    strct1_collect[3] = structures_1[2]
    structures_1[2]["label"] = "t3_0"
    structure_comparison(strct1_collect["t3_0"], structures_1[2])
    del structures_1[2]["label"]

    for label, strct1, strct1_c in zip(
        ["t1_Benzene", "t1_SF6"], structures_1[:2], strct1_collect[:2]
    ):
        strct1["label"] = label
        structure_comparison(strct1_c, strct1)
        # del strct1["label"]
    structure_comparison(strct1_collect[1], structures_1[1])
    structure_comparison(strct1_collect.pop("t1_Benzene"), structures_1[0])

    strct1_collect["t1_4"] = structures_1[0]
    structures_1[0]["label"] = "t1_4"
    structure_comparison(strct1_collect["t1_4"], structures_1[0])

    del strct1_collect["t1_4"]
    assert strct1_collect.labels == [
        "t1_SF6",
        "t1_H2O",
        "t3_0",
        "t2_ZIF-8",
        "t2_GaAs_216_prim",
        "t2_Cs2Te_19_prim",
    ], "Wrong labels."
    for label, strct_c, strct in zip(
        ["t2_ZIF-8", "t2_GaAs_216_prim", "t2_Cs2Te_19_prim"], strct2_collect, structures_2
    ):
        strct["label"] = label
        structure_comparison(strct_c, strct)
    for idx, (label, strct) in enumerate(strct2_collect.items()):
        structure_comparison(strct, structures_2[idx])

    strct_c_red = strct1_collect[1:4]
    assert strct_c_red.labels == strct1_collect.labels[1:4]


def test_duplicate_structure(create_structure_collection_object, structure_comparison):
    """Test duplicate_structure function."""
    strct_c, structures = create_structure_collection_object(["H3PO4"])
    strct_c.duplicate_structure("H3PO4", "new_H3PO4")
    structures[0]["label"] = "H3PO4"
    structure_comparison(strct_c["H3PO4"], structures[0])
    structures[0]["label"] = "new_H3PO4"
    structure_comparison(strct_c["new_H3PO4"], structures[0])


def test_aiida_interface(create_structure_collection_object, structure_comparison):
    """Test AiiDA interface."""
    structure_list = ["Benzene", "SF6", "H2O", "H3PO4", "GaAs_216_prim", "Cs2Te_19_prim"]
    strct_c, structures = create_structure_collection_object(structure_list)
    strct_c.store_in_aiida_db(group_label="test")
    strct_c_2 = StructureCollection()
    strct_c_2.import_from_aiida_db("test")
    for label, structure in zip(structure_list, structures):
        structure["label"] = label
        structure_comparison(strct_c_2[label], structure)


def test_create_pandas_df(nested_dict_comparison):
    """Test pandas data frame creation."""
    coord_kwargs = {
        "r_max": 5.0,
        "method": "minimum_distance",
        "min_dist_delta": 0.1,
    }
    structures = ["GaAs_216_prim", "GaAs_216_conv", "Cs2Te_62_prim", "Benzene"]
    ref = dict(read_yaml_file(EXT_INTERFACES_PATH + "pandas_df_ref.yaml"))

    strct_collect = StructureCollection()
    for strct in structures:
        strct_collect.append(strct, **dict(read_yaml_file(STRUCTURES_PATH + strct + ".yaml")))
    strct_ops = StructureOperations(strct_collect)
    strct_ops[strct_collect.labels].calc_coordination(**coord_kwargs)
    df = strct_collect.create_pandas_df()
    df_dict = {}
    for column in df.columns:
        df_dict[column] = [None if value is pd.NA else value for value in df[column]]
    nested_dict_comparison(df_dict, ref)


def test_h5py_interface(structure_comparison):
    """Test store/import functions for the hdf5 interface."""
    ref_data = read_yaml_file(EXT_INTERFACES_PATH + "hdf5_ref.yaml")
    strct_collect = StructureCollection()
    for label, structure in ref_data.items():
        strct_collect.append(label, **structure)
    strct_collect.store_in_hdf5_file(EXT_INTERFACES_PATH + "test_strct_c.hdf5")

    strct_collect = StructureCollection()
    strct_collect.import_from_hdf5_file(EXT_INTERFACES_PATH + "test_strct_c.hdf5")

    for label, structure in ref_data.items():
        structure["label"] = label
        structure_comparison(strct_collect[label], structure)


@pytest.mark.parametrize("with_labels", [True, False])
def test_import_from_pandas_df(
    create_structure_collection_object, structure_comparison, with_labels
):
    """Test import_from_pandas_df function."""
    structure_list = ["Benzene", "SF6", "H2O", "H3PO4", "GaAs_216_prim", "Cs2Te_19_prim"]
    strct_c, structures = create_structure_collection_object(structure_list)
    pandas_dict = {
        "band gap (eV)": [0.0, 1.0, None, 2.0, 3.0, 2.0],
        "energy": [-1.0, -3.0, -4.0, -10.0, 2.0, 3.0],
    }
    if with_labels:
        pandas_dict["label"] = structure_list
        pandas_dict["node_col"] = [
            node_dict["structure"] for node_dict in strct_c.store_in_aiida_db(group_label="test")
        ]
        labels = structure_list
    else:
        labels = ["pandas_" + str(idx0) for idx0 in range(len(structure_list))]
        node_list = []
        for strct in strct_c._structures:
            strct.label = None
            node = strct.to_aiida_structuredata()
            node.store()
            node_list.append(node.pk)
        pandas_dict["node_col"] = node_list
    df = _turn_dict_into_pandas_df(pandas_dict)
    strct_c_2 = StructureCollection()
    strct_c_2.import_from_pandas_df(df, structure_column="node_col")
    for label, structure in zip(labels, structures):
        structure["label"] = label
        structure_comparison(strct_c_2[label], structure)
    for label, bg, e0 in zip(labels, pandas_dict["band gap (eV)"], pandas_dict["energy"]):
        assert strct_c_2[label]["attributes"]["energy"] == e0
        assert strct_c_2[label]["attributes"]["band gap"]["unit"] == "eV"
        if bg is not pd.NA:
            assert strct_c_2[label]["attributes"]["band gap"]["value"] == bg
        else:
            assert strct_c_2[label]["attributes"]["band gap"]["value"] is pd.NA


def test_print(create_structure_collection_object):
    """Test print function."""
    strct_c = StructureCollection()
    assert strct_c.__str__() == (
        "----------------------------------------------------------------------\n"
        "------------------------ Structure Collection ------------------------\n"
        "----------------------------------------------------------------------\n"
        "\n"
        " - Number of structures: 0\n"
        " - Elements: \n"
        "\n"
        "                              Structures                              \n"
        "  not set.\n"
        "----------------------------------------------------------------------"
    )
    strct_c, _ = create_structure_collection_object(
        ["Benzene", "SF6", "ZIF-8"]
    )  # , "H2O", "H3PO4", "ZIF-8", "GaAs_216_prim", "Cs2Te_19_prim"])
    assert strct_c.__str__() == (
        "----------------------------------------------------------------------\n"
        "------------------------ Structure Collection ------------------------\n"
        "----------------------------------------------------------------------\n"
        "\n"
        " - Number of structures: 3\n"
        " - Elements: C-F-H-N-S-Zn\n"
        "\n"
        "                              Structures                              \n"
        "  - Benzene             C6H6                [False False False]\n"
        "  - SF6                 SF6                 [False False False]\n"
        "  - ZIF-8               Zn6C48N24H60        [True  True  True ]\n"
        "----------------------------------------------------------------------"
    )
    strct_c, _ = create_structure_collection_object(
        [
            "Benzene",
            "SF6",
            "H2O",
            "H3PO4",
            "ZIF-8",
            "GaAs_216_prim",
            "Cs2Te_19_prim",
            "Al_225_conv",
            "CsK2Sb_225",
            "NaCl_225_prim",
            "HClO",
        ]
    )
    assert strct_c.__str__() == (
        "----------------------------------------------------------------------\n"
        "------------------------ Structure Collection ------------------------\n"
        "----------------------------------------------------------------------\n"
        "\n"
        " - Number of structures: 11\n"
        " - Elements: Al-As-C-Cl-Cs-F-Ga-H-K-N-Na-O-P-S-Sb-Te-Zn\n"
        "\n"
        "                              Structures                              \n"
        "  - Benzene             C6H6                [False False False]\n"
        "  - SF6                 SF6                 [False False False]\n"
        "  - H2O                 OH2                 [False False False]\n"
        "  - H3PO4               PO4H3               [False False False]\n"
        "  - ZIF-8               Zn6C48N24H60        [True  True  True ]\n"
        "  ...\n"
        "  - Cs2Te_19_prim       Cs8Te4              [True  True  True ]\n"
        "  - Al_225_conv         Al4                 [True  True  True ]\n"
        "  - CsK2Sb_225          K8Cs4Sb4            [True  True  True ]\n"
        "  - NaCl_225_prim       NaCl                [True  True  True ]\n"
        "  - HClO                ClOH                [False False False]\n"
        "----------------------------------------------------------------------"
    )


@pytest.mark.parametrize(
    "structure",
    ["Benzene", "ZIF-8"],
)
def test_append_from_ase_atoms(structure_comparison, structure):
    """Test append_from_ase_atoms function."""
    ref_structure_dict = read_yaml_file(STRUCTURES_PATH + structure + ".yaml")
    strct_c = StructureCollection()
    strct_c.append_from_ase_atoms(
        ase_atoms=Structure(**ref_structure_dict).to_ase_atoms(), label="test"
    )
    ref_structure_dict["label"] = "test"
    structure_comparison(strct_c["test"], ref_structure_dict)


@pytest.mark.parametrize(
    "structure",
    ["Benzene", "Cs2Te_62_prim_kinds"],
)
def test_append_from_pymatgen_structure(structure_comparison, structure):
    """Test append_from_ase_atoms function."""
    ref_structure_dict = read_yaml_file(STRUCTURES_PATH + structure + ".yaml")
    strct_c = StructureCollection()
    strct_c.append_from_pymatgen_structure(
        pymatgen_structure=Structure(**ref_structure_dict).to_pymatgen_structure(), label="test"
    )
    ref_structure_dict["label"] = "test"
    structure_comparison(strct_c["test"], ref_structure_dict)


@pytest.mark.parametrize(
    "structure, use_node_label, use_uuid",
    [("Benzene", True, True), ("Cs2Te_62_prim_kinds", False, False)],
)
def test_append_from_aiida_structuredata(
    structure_comparison, structure, use_node_label, use_uuid
):
    """Test append_from_aiida_structuredata function."""
    ref_structure_dict = read_yaml_file(STRUCTURES_PATH + structure + ".yaml")
    ref_structure_dict["label"] = structure
    strct = Structure(**ref_structure_dict)
    structure_node = strct.to_aiida_structuredata()
    structure_node.store()
    strct_c = StructureCollection()
    strct_c.append_from_aiida_structuredata(
        label=None if use_node_label else "test", aiida_node=structure_node.pk, use_uuid=use_uuid
    )
    if not use_node_label:
        ref_structure_dict["label"] = "test"
    structure_comparison(strct_c[0], ref_structure_dict)


@pytest.mark.parametrize(
    "structure,file_suffix",
    [("Benzene", ".xyz"), ("GaAs_216_prim", ".xsf")],
)
def test_file_support(structure_comparison, structure, file_suffix):
    """Test the ase atoms interface by loading structures from file."""
    ref_structure_dict = read_yaml_file(STRUCTURES_PATH + structure + ".yaml")
    Structure(**ref_structure_dict).to_file(STRUCTURES_PATH + "test" + file_suffix)
    strct_c = StructureCollection()
    strct_c.append_from_file(file_path=STRUCTURES_PATH + "test" + file_suffix, label="test")
    ref_structure_dict["label"] = "test"
    structure_comparison(strct_c["test"], ref_structure_dict)


def test_store_calc_properties(create_structure_collection_object, nested_dict_comparison):
    """Test storage of calculated data."""
    strct_list = ["GaAs_216_prim", "Cs2Te_19_prim"]
    strct_c, structures = create_structure_collection_object(strct_list, "")
    function_args = {
        "r_max": 10.0,
        "method": "minimum_distance",
        "min_dist_delta": 0.1,
        "n_nearest_neighbours": 5,
        "radius_type": "chen_manz",
        "atomic_radius_delta": 0.0,
        "econ_tolerance": 0.5,
        "econ_conv_threshold": 0.001,
        "voronoi_weight_type": "rel_solid_angle",
        "voronoi_weight_threshold": 0.5,
        "okeeffe_weight_threshold": 0.5,
    }
    strct_ops = StructureOperations(strct_c)
    coord = strct_ops[0].calc_coordination(**function_args)
    assert (
        strct_ops.structures._structures[0]._function_args["coordination"] == function_args
    ), "Function parameters are wrong."
    assert (
        strct_ops.structures._structures[0]["extras"]["coordination"] == coord
    ), "Calculated extra is wrong."
    assert strct_ops[0].calc_coordination(**function_args) == coord, "Recalculation is wrong."

    function_args = {
        "symprec": 0.005,
        "angle_tolerance": -1.0,
        "hall_number": 0,
        "return_sym_operations": True,
        "no_idealize": False,
        "return_primitive_structure": False,
        "return_standardized_structure": False,
    }
    space_group = strct_ops[0].calc_space_group(**function_args)
    assert (
        strct_ops.structures._structures[0]._function_args["space_group"] == function_args
    ), "Function parameters are wrong."
    assert (
        strct_ops.structures._structures[0]["attributes"]["space_group"]
        == space_group["space_group"]["number"]
    ), "Calculated extra is wrong."
    assert (
        strct_ops.structures._structures[0]["extras"]["space_group"] == space_group
    ), "Calculated extra is wrong."
    assert strct_ops[0].calc_space_group(**function_args) == space_group, "Recalculation is wrong."
    function_args["return_sym_operations"] = False
    space_group = strct_ops[0].calc_space_group(**function_args)
    assert (
        strct_ops.structures._structures[0]._function_args["space_group"] == function_args
    ), "Function parameters are wrong."
    assert (
        strct_ops.structures._structures[0]["attributes"]["space_group"]
        == space_group["space_group"]["number"]
    ), "Calculated extra is wrong."
    assert (
        strct_ops.structures._structures[0]["extras"]["space_group"] == space_group
    ), "Calculated extra is wrong."

    structure = strct_c[1]
    with pytest.raises(TypeError) as error:
        structure.store_calculated_properties = 1
    assert str(error.value) == "`store_calculated_properties` needs to be of type bool."
    strct_ops.structures._structures[1].store_calculated_properties = False

    assert not strct_ops.structures._structures[
        1
    ].store_calculated_properties, "Setting `store_calculated_properties` to False not working."
    space_group = strct_ops[1].calc_space_group(**function_args)
    assert (
        strct_ops.structures._structures[1]._function_args == {}
    ), "Function parameters are wrong."
    assert (
        strct_ops.structures._structures[1]["attributes"]["space_group"]
        == space_group["space_group"]["number"]
    ), "Calculated extra is wrong."
    assert strct_ops.structures._structures[1]["extras"] == {}, "Calculated extra is wrong."
