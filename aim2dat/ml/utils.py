"""Helper functions for machine learning tasks."""

# Standard library imports
from random import shuffle
import math

# Third party library imports
import numpy as np
from sklearn.model_selection import train_test_split

# Internal library imports
from aim2dat.chem_f import transform_list_to_dict
from aim2dat.strct import StructureCollection


def _get_all_elements(X, distinguish_kinds):
    """Create element or kind pairs."""
    el_type = "elements"
    if distinguish_kinds:
        el_type = "kinds"
    all_elements = []
    for strct in X:
        all_elements += strct[el_type]
    return sorted(set(all_elements))


def _retrieve_target_value(structure, target_attribute):
    """Retrieve attribute from structure dictionary."""
    if target_attribute not in structure["attributes"]:
        raise ValueError(
            f"Target 'target_attribute' not available for structure '{structure['label']}'."
        )
    if (
        isinstance(structure["attributes"][target_attribute], dict)
        and "value" in structure["attributes"][target_attribute]
    ):
        return structure["attributes"][target_attribute]["value"]
    else:
        return structure["attributes"][target_attribute]


def _remove_structures(structure_list, exclude_labels):
    """Remove structures from list."""
    if len(exclude_labels) > 0:
        ind2del = []
        for idx, strct in enumerate(structure_list):
            if strct["label"] in exclude_labels:
                ind2del.append(idx)
        ind2del.sort(reverse=True)
        for idx in ind2del:
            del structure_list[idx]


def _check_train_test_size(n_structures, train_size, test_size):
    """Chcek sizes of training and test subset."""

    def _get_n_dataset(dataset_size, n_structures, round_type):
        n_ds = None
        if dataset_size is not None:
            if dataset_size > 1.0:
                n_ds = int(dataset_size)
            else:
                n_ds = getattr(math, round_type)(dataset_size * n_structures)
        return n_ds

    n_train = _get_n_dataset(train_size, n_structures, "floor")
    n_test = _get_n_dataset(test_size, n_structures, "ceil")
    if n_train is None and n_test is None:
        raise ValueError("`train_size` or `test_size` need to be set.")
    if n_test is None:
        n_test = n_structures - n_train
    if n_train is None:
        n_train = n_structures - n_test
    if n_structures < n_train + n_test:
        raise ValueError(
            "`train_size`+`test_size` need to be smaller than the overall dataset size."
        )
    return n_train, n_test


def _build_stratified_subset(subset_size, strct_list, hist_data, used_indices):
    """Create a strafied subset based on the target attribute and/or the composition."""
    hist_subset = {key: np.zeros(len(val[1])) for key, val in hist_data.items()}
    subset = []
    target = []
    for idx, strct in enumerate(strct_list):
        if idx in used_indices:
            continue
        add2subset = True
        bin_indices = {}
        for key, hist in hist_data.items():
            value = hist[0][idx]
            for bin_idx, bin_e in enumerate(hist[2][1:]):
                if value < bin_e:
                    break
            if hist_subset[key][bin_idx] < math.floor(subset_size * hist[1][bin_idx]):
                bin_indices[key] = bin_idx
            else:
                add2subset = False
                break
        if add2subset:
            for key, val in bin_indices.items():
                hist_subset[key][val] += 1
            subset.append(strct)
            target.append(hist_data["target"][0][idx])
            used_indices.append(idx)
        if len(subset) == subset_size:
            break
    return subset, target


def train_test_split_crystals(
    structure_collection,
    target_attribute,
    train_size=None,
    test_size=None,
    target_bins=None,
    composition_bins=None,
    elements=None,
    exclude_labels=[],
    return_structure_collections=False,
):
    """
    Split dataset of crystals into a training and test dataset. The target attribute and/or the
    composition can be strafied based on binning.

    Parameters
    ----------
    structure_collection : aim2dat.strct.StructureCollection
        ``StructureCollection'' containing the crystals.
    target_attribute : str
        Label of the target attribute.
    train_size : float, int or None (optional)
        Training set size.
    test_size : float, int or None (optional)
        Test set size.
    target_bins : int or sequence of scalars or str or None (optional)
        Input for np.histogram function. If set to ``None`` binning is not performed. If
        ``target_bins`` and ``composition_bins`` is set to ``None`` the ``train_test_split``
        function of scikit learn is used.
    composition_bins : int or sequence of scalars or str or None (optional)
        Input for np.histogram function. If set to ``None`` binning is not performed. If
        ``target_bins`` and ``composition_bins`` is set to ``None`` the ``train_test_split``
        function of scikit learn is used.
    elements : list or None
        Elements that are considered for composition binning. If set to ``None`` all elements are
        taken into account.
    exclude_labels : list
        Structure labels that should be excluded from the train and test dataset.
    return_structure_collections : bool
        Whether to return the train and test dataset as ``StructureCollection`` objects.

    Returns
    -------
    subset_train : list or StructureCollection
        Training set returned as list or ``StructureCollection`` object.
    subset_test : list or StructureCollection
        Test set returned as list or ``StructureCollection`` object.
    target_train : list
        List of target values of the training set.
    target_test : list
        List of target values of the test set.
    """
    strct_list = structure_collection.get_all_structures()

    if target_bins is not None or composition_bins is not None:
        shuffle(strct_list)
        if elements is None:
            all_elements = _get_all_elements(strct_list, False)
            all_elements = all_elements[:-1]
        else:
            all_elements = elements
        el_comps = {el: [] for el in all_elements}
        target = []
        for strct in strct_list:
            chem_f = transform_list_to_dict(strct["elements"])
            for el in all_elements:  # , val in chem_f.items():
                if el in chem_f:
                    el_comps[el].append(chem_f[el] / len(strct["elements"]))
                else:
                    el_comps[el].append(0.0)
            target.append(_retrieve_target_value(strct, target_attribute))
        n_strct_list = len(strct_list)
        _remove_structures(strct_list, exclude_labels)

        n_train, n_test = _check_train_test_size(len(strct_list), train_size, test_size)
        hist_data = {}
        if target_bins is None:
            target_bins = 1
        hist, bin_edges = np.histogram(target, bins=target_bins)
        hist = hist / n_strct_list
        hist_data["target"] = (target, hist, bin_edges)
        if composition_bins is None:
            composition_bins = 1
        for el, vals in el_comps.items():
            hist, bin_edges = np.histogram(vals, bins=composition_bins)
            hist = hist / n_strct_list
            hist_data["el"] = (vals, hist, bin_edges)

        used_indices = []
        subset_train, target_train = _build_stratified_subset(
            n_train, strct_list, hist_data, used_indices
        )
        subset_test, target_test = _build_stratified_subset(
            n_test, strct_list, hist_data, used_indices
        )
        for subset, subset_t, n_subset in [
            (subset_train, target_train, n_train),
            (subset_test, target_test, n_test),
        ]:
            idx0 = 0
            while len(subset) < n_subset:
                if idx0 not in used_indices:
                    used_indices.append(idx0)
                    subset.append(strct_list[idx0])
                    subset_t.append(target[idx0])
                idx0 += 1
    else:
        _remove_structures(strct_list, exclude_labels)
        target = [_retrieve_target_value(strct, target_attribute) for strct in strct_list]
        subset_train, subset_test, target_train, target_test = train_test_split(
            strct_list, target, train_size=train_size, test_size=test_size
        )
    if return_structure_collections:
        return (
            StructureCollection(subset_train),
            StructureCollection(subset_test),
            target_train,
            target_test,
        )
    else:
        return subset_train, subset_test, target_train, target_test
