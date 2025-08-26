"""Module to calculate  atomic fingerprints."""

# Standard library imports
import math
import itertools

# Third party library imports
import numpy as np
from scipy.spatial.distance import cdist

# Internal library imports
from aim2dat.strct.manipulation.cell import _create_supercell_positions
from aim2dat.fct.smearing import apply_smearing


def calc_ffingerprint(structure, r_max, delta_bin, sigma, use_legacy_smearing, distinguish_kinds):
    """Calculate f-fingerprint."""
    elements_uc = structure["elements"]
    element_dict = structure._element_dict
    if distinguish_kinds:
        if any(k is None for k in structure.kinds):
            raise ValueError("If `distinguish_kinds` is true, all `kinds` must be unequal None.")
        elements_uc = structure["kinds"]
        element_dict = structure._kind_dict
    bins, el_rdf, atomic_rdf = _calculate_prdf(
        structure, r_max, delta_bin, distinguish_kinds, "ffingerprint"
    )

    if use_legacy_smearing:
        sigma /= delta_bin

    # Subtract 1, smear out RDFs and add fingerprints to structure:
    element_fingerprints = {"bins": bins, "fingerprints": {}}
    for el_pair in itertools.combinations_with_replacement(element_dict.keys(), 2):
        el_pair = tuple(sorted(el_pair))
        fingerprint = apply_smearing(el_rdf[el_pair], sigma=sigma, method="gaussian")
        fingerprint -= 1.0
        element_fingerprints["fingerprints"][el_pair] = fingerprint.tolist()

    atomic_fingerprints = []
    for atom_idx, atom_fps in enumerate(atomic_rdf):
        atomic_fprint = {"bins": bins, "fingerprints": {}}
        element0 = elements_uc[atom_idx]
        for element in element_dict.keys():
            el_tuple = tuple(sorted([element0, element]))
            atom_fps[element] *= len(element_dict[element0])
            fingerprint = apply_smearing(atom_fps[element], sigma=sigma, method="gaussian")
            fingerprint -= 1.0
            atomic_fprint["fingerprints"][el_tuple] = fingerprint.tolist()
        atomic_fingerprints.append(atomic_fprint)
    return (element_fingerprints, atomic_fingerprints)


def _calculate_prdf(structure, r_max, delta_bin, distinguish_kinds, method):
    def _get_weights(el_pair, el_dict, delta_bin, distances, volume, method):
        if method == "prdf":
            b_min = distances - distances % delta_bin
            b_max = distances - distances % delta_bin + delta_bin
            return 3.0 / (4.0 * np.pi * (b_max**3.0 - b_min**3.0) * len(element_dict[el_pair[0]]))
        elif method == "ffingerprint":
            el_product = len(element_dict[el_pair[0]]) * len(element_dict[el_pair[1]])
            return volume / (4.0 * delta_bin * np.pi * distances**2.0 * el_product)

    volume = structure["cell_volume"]
    elements_uc = structure["elements"]
    positions_uc = structure.get_positions(cartesian=True, wrap=True)
    elements_sc, kinds_sc, positions_sc, indices_sc, _, _ = _create_supercell_positions(
        structure, r_max
    )
    element_dict = structure._element_dict
    if distinguish_kinds:
        if any(k is None for k in structure.kinds):
            raise ValueError("If `distinguish_kinds` is true, all `kinds` must be unequal None.")
        elements_uc = structure["kinds"]
        element_dict = structure._kind_dict
        elements_sc = kinds_sc

    # Initialize lists:
    bins = [
        0.5 * delta_bin + bin_idx * delta_bin for bin_idx in range(math.ceil(r_max / delta_bin))
    ]

    # Create mask based on r_max and elements:
    distance_matrix = cdist(positions_uc, positions_sc)
    general_mask = np.arange(len(elements_uc)).reshape(-1, 1) != np.array(indices_sc).reshape(
        1, -1
    )
    general_mask = np.where((distance_matrix < r_max) & general_mask, True, False)

    # Map elements to number
    el_mapping = {el: i for i, el in enumerate(set(elements_uc))}
    mapped_elements_uc = np.array([el_mapping[el] for el in elements_uc]).reshape(-1, 1)
    mapped_elements_sc = np.array([el_mapping[el] for el in elements_sc]).reshape(1, -1)

    # Create array of bins corresponding to each distance value
    bins_idx = np.floor(distance_matrix / delta_bin).astype(int)

    el_rdf = {}
    for el_pair in itertools.product(element_dict.keys(), element_dict.keys()):
        el_tuple_mask = (mapped_elements_uc == el_mapping[el_pair[0]]) & (
            mapped_elements_sc == el_mapping[el_pair[1]]
        )
        mask = np.logical_and(el_tuple_mask, general_mask)
        bins_idx0 = bins_idx[mask]
        distances = distance_matrix[mask]
        weights = _get_weights(el_pair, element_dict, delta_bin, distances, volume, method)
        el_rdf[el_pair] = np.bincount(bins_idx0, weights=weights, minlength=len(bins)).astype(
            float
        )
    atomic_rdf = []
    for site_idx, el1 in enumerate(elements_uc):
        site_rdf = {}
        for el2 in element_dict.keys():
            el_tuple_mask = (mapped_elements_sc == el_mapping[el2]).reshape(-1)
            mask = np.logical_and(general_mask[site_idx], el_tuple_mask)
            distances = distance_matrix[site_idx][mask]
            bins_idx0 = bins_idx[site_idx][mask]
            weights = _get_weights([el1, el2], element_dict, delta_bin, distances, volume, method)
            site_rdf[el2] = np.bincount(bins_idx0, weights=weights, minlength=len(bins)).astype(
                float
            )
        atomic_rdf.append(site_rdf)
    return bins, el_rdf, atomic_rdf


def _ffingerprint_compare_sites(
    structures, site_indices, calc_properties, distinguish_kinds, use_weights
):
    element_dicts = [strct._element_dict for strct in structures]
    if distinguish_kinds:
        element_dicts = [strct._kind_dict for strct in structures]
    return _ffingerprint_compare(
        element_dicts,
        [calc_properties[idx][1][site_indices[idx]] for idx in range(2)],
        use_weights,
    )


def _ffingerprint_compare(element_dicts, fprints, use_weights):
    """
    Calculate the cosine distance of f-fingerprints, if site_index1 and site_index2 are -1,
    elemental fingerprints are calculated otherwise atomic ones are used.
    """
    # Get weights:
    if use_weights:
        weights1 = _calculate_weights(element_dicts[0])
        weights2 = _calculate_weights(element_dicts[1])

        # In case of different compositions the average of both weights is taken:
        av_weights = {
            el_pair1: (weight1 + weights2[el_pair1]) / 2.0
            for el_pair1, weight1 in weights1.items()
        }
    else:
        weights1 = {el_pair1: 1.0 for el_pair1 in fprints[0]["fingerprints"].keys()}
        weights2 = weights1
        av_weights = weights1

    # Calculate the distance of between the two fingerprints and return it:
    return _calculate_cosine_dist(fprints[0], fprints[1], weights1, weights2, av_weights)


def _calculate_weights(element_dict):
    """Calculate weights to determine the similarity of the structure."""
    weights = {}
    tot_weight = 0.0
    for el1_idx, element1 in enumerate(element_dict.keys()):
        for element2 in list(element_dict.keys())[el1_idx:]:
            weight = float(len(element_dict[element1]) * len(element_dict[element2]))
            weights[tuple(sorted([element1, element2]))] = weight
            tot_weight += weight
    for key in weights.keys():
        weights[key] /= tot_weight
    return weights


def _calculate_cosine_dist(vectors1, vectors2, weights1, weights2, av_weights):
    """Calculate the distance of between the two fingerprint functions."""
    distance = 0.0
    norm1 = 0.0
    norm2 = 0.0
    for el_pair in vectors1["fingerprints"].keys():
        y_values1 = np.array(vectors1["fingerprints"][el_pair])
        y_values2 = np.array(vectors2["fingerprints"][el_pair])
        norm1 += (np.linalg.norm(y_values1) ** 2) * weights1[el_pair]
        norm2 += (np.linalg.norm(y_values2) ** 2) * weights2[el_pair]
        distance += np.sum(y_values1 * y_values2) * av_weights[el_pair]
    return 0.5 * (1 - distance / (math.sqrt(norm1) * math.sqrt(norm2)))
