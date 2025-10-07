"""Test the general interface of the StructuresOperations class."""

# Standard library imports
import os

# Third party library imports
import pandas as pd
import pytest

# Internal library imports
from aim2dat.strct import StructureOperations, StructureCollection, Structure
from aim2dat.io import read_yaml_file

STRUCTURES_PATH = os.path.dirname(__file__) + "/structures/"
REF_PATH = os.path.dirname(__file__) + "/structure_op/"


def test_structure_op_basics():
    """Test basic features of the StructuresOperations class."""
    strct_collect = StructureCollection()
    inputs = dict(read_yaml_file(STRUCTURES_PATH + "Cs2Te_62_prim.yaml"))
    strct_collect.append("Cs2Te_62_prim", **inputs)

    inputs = dict(read_yaml_file(STRUCTURES_PATH + "NaCl_225_prim.yaml"))
    strct_collect.append("NaCl_225_prim", **inputs)

    strct_ops = StructureOperations(structures=strct_collect)
    strct_ops_inp_list = StructureOperations([Structure(label="Cs2Te_62_prim", **inputs)])
    assert isinstance(strct_ops_inp_list.structures, StructureCollection)

    with pytest.raises(TypeError) as error:
        StructureOperations([1, 2, 3])

    assert isinstance(strct_ops["Cs2Te_62_prim"].calc_distance(0, 1), float)
    assert isinstance(strct_ops[["Cs2Te_62_prim", "NaCl_225_prim"]].calc_distance(0, 1), dict)
    strct_ops.output_format = "DataFrame"
    assert isinstance(
        strct_ops[["Cs2Te_62_prim", "NaCl_225_prim"]].calc_distance(0, 1), pd.DataFrame
    )
    with pytest.raises(TypeError) as error:
        strct_ops.output_format = 10
    assert str(error.value) == "`output_format` needs to be of type str."
    with pytest.raises(ValueError) as error:
        strct_ops.output_format = "DataFramE"
    assert (
        str(error.value)
        == "`output_format` 'DataFramE' is not supported."
        + " It has to be one of the following options: ['dict', 'DataFrame']"
    )

    with pytest.raises(ValueError) as error:
        strct_ops.structures["Cs2Te_62_prim"].kinds = None
        strct_ops.compare_structures_via_ffingerprint(0, 1, distinguish_kinds=True)
    assert str(error.value) == "If `distinguish_kinds` is true, all `kinds` must be unequal None."

    with pytest.raises(ValueError) as error:
        strct_ops.compare_sites_via_ffingerprint(0, 1, 20, 20)
    assert str(error.value) == "Site index out of range for structure '0'."


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
        "get_statistics": True,
        "indices": None,
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


def test_structure_op_pipeline(structure_comparison):
    """Test pipeline implementation of the StructureOperations class."""
    ref = read_yaml_file(REF_PATH + "pipeline.yaml")
    strct_c = StructureCollection()
    for file_name, label in ref["input_structures"]:
        strct_c.append_structure(Structure.from_file(STRUCTURES_PATH + file_name), label=label)
    strct_op = StructureOperations(strct_c)
    strct_op.pipeline = ref["pipeline"]
    # TODO test output pipeline, return None if pipeline is not set and errors.
    structures = strct_op.run_pipeline()
    assert len(structures) == len(ref["output_structures"])
    for strct, ref_strct in zip(structures, ref["output_structures"]):
        structure_comparison(strct, ref_strct)
    assert len(strct_op.structures) == len(strct_c)
    for strct0, strct1 in zip(strct_op.structures, strct_c):
        structure_comparison(strct0, strct1)


@pytest.mark.parametrize(
    "structure1,structure2,distinguish_kinds,use_weights,ref_value",
    [
        ("GaAs_216_conv", "GaAs_216_prim", False, True, 0.0),
        ("GaAs_216_conv", "GaAs_216_prim", False, False, 0.0),
        ("Cs2Te_62_prim", "Cs2Te_19_prim", False, True, 0.035570759),
        ("Cs2Te_62_prim", "Cs2Te_19_prim", False, False, 0.037050961),
        ("Cs2Te_62_prim", "Cs2Te_19_prim", True, True, 0.108369),
    ],
)
def test_compare_structures_via_ffingerprint(
    structure1, structure2, distinguish_kinds, use_weights, ref_value
):
    """Test the F-Fingerprint distance between two structures."""
    strct_collect = StructureCollection()
    ffprint_args = {
        "r_max": 15.0,
        "sigma": 10.0,
        "delta_bin": 0.005,
        "distinguish_kinds": distinguish_kinds,
        "use_legacy_smearing": False,
        "use_weights": use_weights,
    }

    # Load structures:
    for structure in [structure1, structure2]:
        inputs = dict(read_yaml_file(STRUCTURES_PATH + structure + ".yaml"))
        strct_collect.append(structure, **inputs)
    strct_ops = StructureOperations(strct_collect)
    assert (
        abs(
            strct_ops.compare_structures_via_ffingerprint(structure1, structure2, **ffprint_args)
            - ref_value
        )
        < 1.0e-4
    )


@pytest.mark.parametrize(
    "structure1,structure2,site_index1,site_index2,distinguish_kinds,ref_value",
    [
        ("GaAs_216_conv", "GaAs_216_prim", 0, 0, False, True),
        ("Cs2Te_62_prim", "Cs2Te_19_prim", 0, 0, True, False),
        ("Cs2Te_62_prim", "Cs2Te_19_prim", 0, 6, True, False),
    ],
)
def test_compare_sites_via_coordination(
    structure1, structure2, site_index1, site_index2, distinguish_kinds, ref_value
):
    """Test compare_sites_via_coordination function."""
    strct_collect = StructureCollection()
    for structure in [structure1, structure2]:
        inputs = dict(read_yaml_file(STRUCTURES_PATH + structure + ".yaml"))
        strct_collect.append(structure, **inputs)

    strct_ops = StructureOperations(strct_collect)
    assert (
        strct_ops.compare_sites_via_coordination(
            structure1,
            structure2,
            site_index1,
            site_index2,
            distinguish_kinds=distinguish_kinds,
            method="minimum_distance",
        )
        == ref_value
    )


@pytest.mark.parametrize(
    "structure1,structure2,atom_idx,distinguish_kinds,ref_value",
    [
        ("GaAs_216_conv", "GaAs_216_prim", 0, False, 0.0),
        ("Cs2Te_62_prim", "Cs2Te_19_prim", 0, True, 0.120721),
    ],
)
def test_compare_sites_via_ffingerprint(
    structure1, structure2, atom_idx, distinguish_kinds, ref_value
):
    """Test the distance between two atomic sites."""
    strct_collect = StructureCollection()

    # Load structures:
    for structure in [structure1, structure2]:
        inputs = dict(read_yaml_file(STRUCTURES_PATH + structure + ".yaml"))
        strct_collect.append(structure, **inputs)

    strct_ops = StructureOperations(strct_collect)
    assert (
        abs(
            strct_ops.compare_sites_via_ffingerprint(
                structure1,
                structure2,
                atom_idx,
                atom_idx,
                use_weights=False,
                distinguish_kinds=distinguish_kinds,
            )
            - ref_value
        )
        < 1.0e-4
    )
    assert (
        abs(
            strct_ops.compare_sites_via_ffingerprint(
                structure1,
                structure2,
                atom_idx,
                atom_idx,
                use_weights=True,
                distinguish_kinds=distinguish_kinds,
            )
            - ref_value
        )
        < 1.0e-4
    )


@pytest.mark.parametrize(
    "method, confined, remove_structures, n_procs, verbose, ref",
    [
        ("ffingerprint", None, True, 1, True, 6),
        ("ffingerprint", None, True, 2, True, 6),
        ("ffingerprint", None, True, 2, False, 6),
        ("ffingerprint", None, False, 1, False, 8),
        ("comp_sym", [None, 7], False, 1, False, 8),
        ("comp_sym", [1, 10], True, 1, False, 6),
        ("direct_comp", None, True, 1, False, 6),
    ],
)
def test_find_duplicates(method, confined, remove_structures, n_procs, verbose, ref):
    """Test find_duplicates_via_... functions."""
    structures = {
        "TiO2_136": "TiO2_136",
        "Cs2Te_62_prim": "Cs2Te_62_prim",
        "Cs2Te_19_prim": "Cs2Te_19_prim",
        "GaAs_216_conv": "GaAs_216_conv",
        "GaAs_216_prim": "GaAs_216_prim",
        "NaCl_225_prim": "NaCl_225_prim",
        "NaCl_225_prim_2": "NaCl_225_prim",
        "CsK2Sb_225": "CsK2Sb_225",
    }
    strct_c = StructureCollection()
    for strct_label, structure in structures.items():
        inputs = dict(read_yaml_file(STRUCTURES_PATH + structure + ".yaml"))
        strct_c.append(label=strct_label, **inputs)

    strct_ops = StructureOperations(strct_c)
    strct_ops.verbose = False
    strct_ops.n_procs = n_procs
    strct_ops.chunksize = 2
    strct_ops.verbose = verbose
    function = getattr(strct_ops, "find_duplicates_via_" + method)
    duplicates = function(confined=confined, remove_structures=remove_structures)
    assert duplicates == [("GaAs_216_prim", "GaAs_216_conv"), ("NaCl_225_prim_2", "NaCl_225_prim")]
    assert len(strct_c) == ref


@pytest.mark.parametrize(
    "structure, file_suffix, method",
    [("ZIF-8", "cif", "ffingerprint"), ("ZIF-8", "cif", "coordination")],
)
def test_find_eq_sites_functions(nested_dict_comparison, structure, file_suffix, method):
    """Test functions to determine equivalent sites."""
    ref_outputs = dict(
        read_yaml_file(REF_PATH + "find_eq_sites_via_" + method + "_" + structure + ".yaml")
    )
    strct_c = StructureCollection()
    if file_suffix == "yaml":
        strct_c.append("test", **read_yaml_file(STRUCTURES_PATH + structure + "." + file_suffix))
    else:
        strct_c.append_from_file(
            "test", STRUCTURES_PATH + structure + "." + file_suffix, backend="internal"
        )

    strct_ops = StructureOperations(strct_c)
    eq_sites = getattr(strct_ops, "find_eq_sites_via_" + method)(
        "test", **ref_outputs["function_args"]
    )
    nested_dict_comparison(eq_sites, ref_outputs["reference"])
