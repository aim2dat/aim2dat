"""Test custom sci-kit structure Transformer classes."""

# Standard library imports
import os

# Third party library imports
import pytest
from sklearn.exceptions import NotFittedError

# Internal library imports
from aim2dat.ml import transformers
from aim2dat.io import read_yaml_file


REF_PATH = os.path.dirname(__file__) + "/structure_transformers_ref/"


def compare_features(features, ref_features):
    """Compare features."""
    if isinstance(ref_features, list):
        assert len(features) == len(ref_features), "Wrong number of data points."
        for feat0, feat1 in zip(features, ref_features):
            compare_features(feat0, feat1)
    else:
        assert abs(features - ref_features) < 1.0e-5, "Feature list is false."


def test_precomputed_properties(create_structure_collection_object):
    """Test precomputed properties."""
    strct_c, _ = create_structure_collection_object(["Cs2Te_62_prim", "Cs2Te_194_prim"])
    parameters = {"r_max": [5.0, 6.0]}
    transf = transformers.StructureFFPrintTransformer()
    transf.precompute_parameter_space(parameters, strct_c)
    assert transf.precomputed_properties[0][0] == {
        "r_max": 5.0,
        "delta_bin": 0.005,
        "sigma": 10.0,
        "distinguish_kinds": False,
    }
    assert transf.precomputed_properties[0][1].structures[0]._function_args == {
        "ffingerprint": {
            "r_max": 5.0,
            "delta_bin": 0.005,
            "sigma": 10.0,
            "distinguish_kinds": False,
            "use_legacy_smearing": False,
        }
    }
    assert transf.precomputed_properties[1][0] == {
        "r_max": 6.0,
        "delta_bin": 0.005,
        "sigma": 10.0,
        "distinguish_kinds": False,
    }
    assert transf.precomputed_properties[1][1].structures[0]._function_args == {
        "ffingerprint": {
            "r_max": 6.0,
            "delta_bin": 0.005,
            "sigma": 10.0,
            "distinguish_kinds": False,
            "use_legacy_smearing": False,
        }
    }
    assert id(transf.precomputed_properties[0][1]) != id(transf.precomputed_properties[1][1])
    for label in transf.precomputed_properties[0][1].structures.labels:
        assert id(transf.precomputed_properties[0][1].structures[label]) != id(
            transf.precomputed_properties[1][1].structures[label]
        )
    strct_op = transf.precomputed_properties[0][1]
    transf.clear_precomputed_properties()
    assert transf.precomputed_properties == ()
    transf.add_precomputed_properties({"r_max": 5.0}, strct_op)
    assert transf.precomputed_properties[0][0] == {
        "r_max": 5.0,
        "delta_bin": 0.005,
        "sigma": 10.0,
        "distinguish_kinds": False,
    }
    assert transf.precomputed_properties[0][1].structures[0]._function_args == {
        "ffingerprint": {
            "r_max": 5.0,
            "delta_bin": 0.005,
            "sigma": 10.0,
            "distinguish_kinds": False,
            "use_legacy_smearing": False,
        }
    }

    transf = transformers.StructureCompositionTransformer()
    transf.precompute_parameter_space({"test_p": [0.0, 1.0]}, strct_c)
    assert transf.precomputed_properties == ()


@pytest.mark.parametrize(
    "transformer_type,test_case,fit_structure_list,transf_structure_list",
    [
        ("Composition", "molecules", ["Benzene"], ["Benzene", "SF6", "H2O", "H3PO4"]),
        (
            "Density",
            "Cs2Te",
            ["Cs2Te_62_prim"],
            ["Cs2Te_62_prim", "Cs2Te_194_prim", "Cs2Te_19_prim", "CsK2Sb_225"],
        ),
        (
            "Coordination",
            "Cs2Te",
            ["Cs2Te_62_prim"],
            ["Cs2Te_62_prim", "Cs2Te_194_prim", "Cs2Te_19_prim", "CsK2Sb_225"],
        ),
        (
            "ChemOrder",
            "Cs2Te",
            ["Cs2Te_62_prim"],
            ["Cs2Te_62_prim", "Cs2Te_194_prim", "Cs2Te_19_prim", "CsK2Sb_225"],
        ),
        (
            "FFPrint",
            "Cs2Te",
            ["Cs2Te_62_prim"],
            ["Cs2Te_62_prim", "Cs2Te_194_prim", "Cs2Te_19_prim", "CsK2Sb_225"],
        ),
        (
            "PRDF",
            "Cs2Te",
            ["Cs2Te_62_prim"],
            ["Cs2Te_62_prim", "Cs2Te_194_prim", "Cs2Te_19_prim", "CsK2Sb_225"],
        ),
        # Due to dscribe not being installable on Python 3.13:
        # (
        #     "Matrix",
        #     "Cs2Te",
        #     ["CsK2Sb_225"],
        #     ["Cs2Te_62_prim", "Cs2Te_194_prim", "Cs2Te_19_prim", "CsK2Sb_225"],
        # ),
        # (
        #     "ACSF",
        #     "molecules",
        #     ["HClO"],
        #     ["HClO", "HCN", "H2O"],
        # ),
        # (
        #     "SOAP",
        #     "Cs2Te",
        #     ["Cs2Te_62_prim"],
        #     ["Cs2Te_62_prim", "Cs2Te_194_prim", "Cs2Te_19_prim", "CsK2Sb_225"],
        # ),
        # (
        #     "MBTR",
        #     "Cs2Te",
        #     ["Cs2Te_62_prim"],
        #     ["Cs2Te_62_prim", "Cs2Te_194_prim", "Cs2Te_19_prim", "CsK2Sb_225"],
        # ),
    ],
)
def test_transformer(
    create_structure_collection_object,
    transformer_type,
    test_case,
    fit_structure_list,
    transf_structure_list,
):
    """Test structure transformers."""
    ref = read_yaml_file(REF_PATH + transformer_type + "_" + test_case + ".yaml")

    strct_c_fit, _ = create_structure_collection_object(fit_structure_list)
    strct_c_transf, _ = create_structure_collection_object(transf_structure_list)
    # if ref["add_structure_collection"]:
    #    ref["init_kwargs"]["structure_operations"] = StructureOperations(structures=strct_c_fit)
    for keyw, val in ref["init_kwargs"].items():
        if isinstance(val, list):
            ref["init_kwargs"][keyw] = tuple(val)
    Transformer = getattr(transformers, "Structure" + transformer_type + "Transformer")

    for n_procs in [1, 2]:
        ref["init_kwargs"]["n_procs"] = n_procs
        transf = Transformer(**ref["init_kwargs"])

        if "no_fit" not in ref:
            with pytest.raises(NotFittedError) as error:
                transf.transform([strct_c_fit[label] for label in fit_structure_list])
            assert (
                str(error.value)
                == "This Structure"
                + transformer_type
                + "Transformer instance is not fitted yet."
                + " Call 'fit' with appropriate arguments before using this estimator."
            )

        compare_features(
            transf.fit_transform([strct_c_fit[label] for label in fit_structure_list]),
            ref["features_fit"],
        )
        compare_features(transf.transform(strct_c_transf), ref["features_transf"])

        if "feature_names_out" in ref:
            assert all(
                name == ref_name
                for name, ref_name in zip(transf.get_feature_names_out(), ref["feature_names_out"])
            ), "Feature names are wrong."
