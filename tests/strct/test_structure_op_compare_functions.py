"""Test the compare functions of the StructureCollection class."""

# Standard library imports
import os

# Third party library imports
import pytest

# Internal library imports
from aim2dat.strct import StructureCollection, StructureOperations
from aim2dat.io.yaml import load_yaml_file


STRUCTURES_PATH = os.path.dirname(__file__) + "/structures/"
EQ_SITES_PATH = os.path.dirname(__file__) + "/eq_sites_analysis/"


@pytest.mark.parametrize(
    "structure1,structure2,distinguish_kinds,use_weights,ref_value",
    [
        ("GaAs_216_conv", "GaAs_216_prim", False, True, 0.0),
        ("GaAs_216_conv", "GaAs_216_prim", False, False, 0.0),
        ("Cs2Te_62_prim", "Cs2Te_19_prim", False, True, 0.035570759),
        ("Cs2Te_62_prim", "Cs2Te_19_prim", False, False, 0.037050961),
        ("Cs2Te_62_prim_kinds", "Cs2Te_19_prim_kinds", True, True, 0.108369),
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
        inputs = dict(load_yaml_file(STRUCTURES_PATH + structure + ".yaml"))
        # inputs["structure_label"] = structure
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
        ("Cs2Te_62_prim_kinds", "Cs2Te_19_prim_kinds", 0, 0, True, False),
        ("Cs2Te_62_prim_kinds", "Cs2Te_19_prim_kinds", 0, 6, True, False),
    ],
)
def test_compare_sites_via_coordination(
    structure1, structure2, site_index1, site_index2, distinguish_kinds, ref_value
):
    """Test compare_sites_via_coordination function."""
    strct_collect = StructureCollection()
    for structure in [structure1, structure2]:
        inputs = dict(load_yaml_file(STRUCTURES_PATH + structure + ".yaml"))
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
        ("Cs2Te_62_prim_kinds", "Cs2Te_19_prim_kinds", 0, True, 0.120721),
    ],
)
def test_compare_sites_via_ffingerprint(
    structure1, structure2, atom_idx, distinguish_kinds, ref_value
):
    """Test the distance between two atomic sites."""
    strct_collect = StructureCollection()

    # Load structures:
    for structure in [structure1, structure2]:
        inputs = dict(load_yaml_file(STRUCTURES_PATH + structure + ".yaml"))
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
        inputs = dict(load_yaml_file(STRUCTURES_PATH + structure + ".yaml"))
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
def test_determine_eq_sites(nested_dict_comparison, structure, file_suffix, method):
    """Test functions to determine equivalent sites."""
    ref_outputs = dict(load_yaml_file(EQ_SITES_PATH + structure + "_" + method + ".yaml"))
    strct_c = StructureCollection()
    if file_suffix == "yaml":
        strct_c.append("test", **load_yaml_file(STRUCTURES_PATH + structure + "." + file_suffix))
    else:
        strct_c.append_from_file(
            "test", STRUCTURES_PATH + structure + "." + file_suffix, backend="internal"
        )

    strct_ops = StructureOperations(strct_c)
    eq_sites = getattr(strct_ops, "find_eq_sites_via_" + method)(
        "test", **ref_outputs["function_args"]
    )
    nested_dict_comparison(eq_sites, ref_outputs["reference"])
