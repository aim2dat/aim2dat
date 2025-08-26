"""Tests for the StructureCollection class."""

# Standard library imports
import os

# Third party library imports
import pytest
import pandas as pd

# Internal library imports
from aim2dat.strct import Structure, StructureCollection, StructureOperations
from aim2dat.io import read_yaml_file

STRUCTURES_PATH = os.path.dirname(__file__) + "/structures/"
IO_PATH = os.path.dirname(__file__) + "/io/"


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
    strct_c, _ = create_structure_collection_object(["Benzene", "SF6", "GaAs_216_conv"])
    assert strct_c.__str__() == (
        "----------------------------------------------------------------------\n"
        "------------------------ Structure Collection ------------------------\n"
        "----------------------------------------------------------------------\n"
        "\n"
        " - Number of structures: 3\n"
        " - Elements: As-C-F-Ga-H-S\n"
        "\n"
        "                              Structures                              \n"
        "  - Benzene             C6H6                [False False False]\n"
        "  - SF6                 SF6                 [False False False]\n"
        "  - GaAs_216_conv       Ga4As4              [True  True  True ]\n"
        "----------------------------------------------------------------------"
    )
    strct_c, _ = create_structure_collection_object(
        [
            "Benzene",
            "SF6",
            "H2O",
            "H3PO4",
            "GaAs_216_prim",
            "Cs2Te_62_prim",
            "HCN",
            "CHBrClF",
            "HClO",
            "GaAs_216_conv",
            "C4H6O6",
        ]
    )
    assert strct_c.__str__() == (
        "----------------------------------------------------------------------\n"
        "------------------------ Structure Collection ------------------------\n"
        "----------------------------------------------------------------------\n"
        "\n"
        " - Number of structures: 11\n"
        " - Elements: As-Br-C-Cl-Cs-F-Ga-H-N-O-P-S-Te\n"
        "\n"
        "                              Structures                              \n"
        "  - Benzene             C6H6                [False False False]\n"
        "  - SF6                 SF6                 [False False False]\n"
        "  - H2O                 OH2                 [False False False]\n"
        "  - H3PO4               PO4H3               [False False False]\n"
        "  - GaAs_216_prim       GaAs                [True  True  True ]\n"
        "  ...\n"
        "  - HCN                 NCH                 [False False False]\n"
        "  - CHBrClF             BrClFCH             [False False False]\n"
        "  - HClO                ClOH                [False False False]\n"
        "  - GaAs_216_conv       Ga4As4              [True  True  True ]\n"
        "  - C4H6O6              O6C4H6              [False False False]\n"
        "----------------------------------------------------------------------"
    )


def test_list_methods():
    """Test listing different method categories."""
    assert StructureCollection.list_import_methods() == [
        "from_file",
        "from_pandas_df",
        "from_aiida_db",
    ]
    assert StructureCollection.list_export_methods() == ["to_file", "to_pandas_df", "to_aiida_db"]


def test_basic_features(create_structure_collection_object, structure_comparison):
    """Test basic features of the StructureCollection class."""
    strct_list_1 = ["Benzene", "SF6", "H2O", "H3PO4"]
    strct_list_2 = ["GaAs_216_conv", "GaAs_216_prim", "Cs2Te_62_prim"]
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
        "t2_GaAs_216_conv",
        "t2_GaAs_216_prim",
        "t2_Cs2Te_62_prim",
    ], "Wrong labels."

    strct1_collect["t1_H3PO4"] = structures_1[2].to_dict()
    structures_1[2].label = "t1_H3PO4"
    structure_comparison(strct1_collect["t1_H3PO4"], structures_1[2])
    structures_1[2].label = None

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
        strct1_collect.append(**structures_1[2].to_dict())
    assert str(error.value) == "Label 't1_Benzene' already used."
    assert strct1_collect.get_structure(10.0) is None
    structures_1[2].label = None
    strct1_collect[3] = structures_1[2]
    structures_1[2]["label"] = "t1_H3PO4"
    structure_comparison(strct1_collect["t1_H3PO4"], structures_1[2])
    structures_1[2].label = None

    structures_1[2]["label"] = "t3_0"
    strct1_collect[3] = structures_1[2]
    structures_1[2]["label"] = "t3_0"
    structure_comparison(strct1_collect["t3_0"], structures_1[2])
    structures_1[2].label = None

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
        "t2_GaAs_216_conv",
        "t2_GaAs_216_prim",
        "t2_Cs2Te_62_prim",
    ], "Wrong labels."
    for label, strct_c, strct in zip(
        ["t2_GaAs_216_conv", "t2_GaAs_216_prim", "t2_Cs2Te_62_prim"], strct2_collect, structures_2
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


# TODO adapt test.
# def test_aiida_interface(create_structure_collection_object, structure_comparison):
#     """Test AiiDA interface."""
#     structure_list = ["Benzene", "SF6", "H2O", "H3PO4", "GaAs_216_prim", "Cs2Te_62_prim"]
#     strct_c, structures = create_structure_collection_object(structure_list)
#     strct_c.store_in_aiida_db(group_label="test")
#     strct_c_2 = StructureCollection()
#     strct_c_2.import_from_aiida_db("test")
#     for label, structure in zip(structure_list, structures):
#         structure["label"] = label
#         structure_comparison(strct_c_2[label], structure)


def test_create_pandas_df(
    create_structure_collection_object, nested_dict_comparison, structure_comparison
):
    """Test pandas data frame creation."""
    coord_kwargs = {
        "r_max": 5.0,
        "method": "minimum_distance",
        "min_dist_delta": 0.1,
    }
    ref = dict(read_yaml_file(IO_PATH + "pandas/df_ref.yaml"))

    strct_c, _ = create_structure_collection_object(
        ["GaAs_216_prim", "GaAs_216_conv", "Cs2Te_62_prim", "Benzene"]
    )
    strct_ops = StructureOperations(strct_c)
    strct_ops[strct_c.labels].calc_coordination(**coord_kwargs)
    strct_c["Benzene"].calc_point_group()
    strct_c["GaAs_216_conv"]["attributes"] = {"band_gap": {"value": 1.0, "unit": "eV"}}
    df = strct_c.create_pandas_df()
    df_dict = {}
    for column in df.columns:
        if column != "structure":
            df_dict[column] = [None if value is pd.NA else value for value in df[column]]
    nested_dict_comparison(df_dict, ref)
    strct_c_new = StructureCollection.from_pandas_df(df)
    for strct1, strct2 in zip(strct_c, strct_c_new):
        structure_comparison(strct1, strct2)


def test_h5py_interface(structure_comparison):
    """Test store/import functions for the hdf5 interface."""
    ref_data = read_yaml_file(IO_PATH + "hdf5/ref.yaml")
    strct_collect = StructureCollection()
    for label, structure in ref_data.items():
        strct_collect.append(label, **structure)
    strct_collect.to_file(IO_PATH + "hdf5/test_strct_c.hdf5")

    strct_collect = StructureCollection.from_file(IO_PATH + "hdf5/test_strct_c.hdf5")

    for label, structure in ref_data.items():
        structure["label"] = label
        structure_comparison(strct_collect[label], structure)


@pytest.mark.parametrize(
    "structure",
    ["Benzene.yaml", "ZIF-8.cif"],
)
def test_append_from_ase_atoms(structure_comparison, structure):
    """Test append_from_ase_atoms function."""
    ref_structure = Structure.from_file(STRUCTURES_PATH + structure, backend="internal")
    strct_c = StructureCollection()
    strct_c.append_from_ase_atoms(ase_atoms=ref_structure.to_ase_atoms(), label="test")
    ref_structure.label = "test"
    structure_comparison(strct_c["test"], ref_structure)


@pytest.mark.parametrize(
    "structure",
    ["Benzene", "Cs2Te_62_prim"],
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


@pytest.mark.parametrize(
    "structure, use_node_label, use_uuid",
    [("Benzene", True, True), ("Cs2Te_62_prim", False, False)],
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
