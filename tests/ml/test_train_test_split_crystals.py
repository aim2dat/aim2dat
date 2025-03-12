"""Test train_test_split_crystals function."""

# Standard library imports
import os

# Third party library imports
import pytest

# Internal library imports
from aim2dat.ml.utils import _check_train_test_size, train_test_split_crystals
import aim2dat.utils.chem_formula as utils_cf
from aim2dat.io import read_yaml_file


REF_PATH = os.path.dirname(__file__) + "/train_test_split_crystals_ref/"


def test_check_train_test_size():
    """Test _check_train_test_sizte function."""
    with pytest.raises(ValueError) as error:
        _check_train_test_size(10, None, None)
    assert str(error.value) == "`train_size` or `test_size` need to be set."
    with pytest.raises(ValueError) as error:
        _check_train_test_size(10, 0.8, 0.3)
    assert (
        str(error.value)
        == "`train_size`+`test_size` need to be smaller than the overall dataset size."
    )

    n_train, n_test = _check_train_test_size(11, 0.4, 0.2)
    assert n_train == 4
    assert n_test == 3

    n_train, n_test = _check_train_test_size(11, 4, None)
    assert n_train == 4
    assert n_test == 7

    n_train, n_test = _check_train_test_size(11, None, 2)
    assert n_train == 9
    assert n_test == 2


@pytest.mark.parametrize("system", ["Cs-Te_random_split", "Cs-Te_comp_bins", "Cs-Te_target_bins"])
def test_train_test_split_crystals_binary(create_structure_collection_object, system):
    """Test train_test_split_crystals function."""
    ref = read_yaml_file(REF_PATH + system + ".yaml")
    structures = ref["structures"]
    if isinstance(structures, str):
        structures = REF_PATH + structures
    strct_c, _ = create_structure_collection_object(structures)

    for ret_strct_c in [True, False]:
        X_train, X_test, y_train, y_test = train_test_split_crystals(
            strct_c, **ref["input_kwargs"], return_structure_collections=ret_strct_c
        )

        add_tol = 0
        if "exclude_labels" in ref["input_kwargs"] and len(ref["input_kwargs"]["exclude_labels"]):
            add_tol = 2 * len(ref["input_kwargs"]["exclude_labels"])
            labels = [strct["label"] for strct in X_train + X_test]
            for label in ref["input_kwargs"]["exclude_labels"]:
                assert label not in labels

        assert len(X_train) == ref["len_train"]
        assert len(y_train) == ref["len_train"]
        assert len(X_test) == ref["len_test"]
        assert len(y_test) == ref["len_test"]

        if "comp_fractions" in ref:
            for subset in [X_train, X_test]:
                bin_nrs = {el: [0] * len(fract) for el, fract in ref["comp_fractions"].items()}
                bin_tol = (1 + add_tol) / len(subset)
                for strct in subset:
                    n_atoms = len(strct["elements"])
                    el_dict = utils_cf.transform_list_to_dict(strct["elements"])
                    for el in bin_nrs.keys():
                        conc = el_dict[el] / n_atoms if el in el_dict else 0.0
                        for bin_idx in range(len(bin_nrs[el])):
                            # TODO generalize for different integer numbers as input
                            if (
                                ref["input_kwargs"]["composition_bins"][bin_idx]
                                <= conc
                                < ref["input_kwargs"]["composition_bins"][bin_idx + 1]
                            ):
                                break
                        bin_nrs[el][bin_idx] += 1
                for el, numbers in bin_nrs.items():
                    for bin_idx, number in enumerate(numbers):
                        assert (
                            abs(ref["comp_fractions"][el][bin_idx] - number / len(subset))
                            <= bin_tol
                        )

        if "target_fractions" in ref:
            for subset in [y_train, y_test]:
                bin_nrs = [0] * len(ref["target_fractions"])
                bin_tol = (1 + add_tol) / len(subset)
                for t_value in subset:
                    for bin_idx in range(len(bin_nrs)):
                        if (
                            ref["input_kwargs"]["target_bins"][bin_idx]
                            <= t_value
                            < ref["input_kwargs"]["target_bins"][bin_idx + 1]
                        ):
                            break
                    bin_nrs[bin_idx] += 1
                for bin_idx, number in enumerate(bin_nrs):
                    assert abs(ref["target_fractions"][bin_idx] - number / len(subset)) <= bin_tol
