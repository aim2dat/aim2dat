"""Structure comparison methods."""

# Standard library imports
import itertools

# Third party library imports
import numpy as np

# Internal library imports
from aim2dat.strct.structure import Structure
from aim2dat.strct.analysis.rdf import _ffingerprint_compare
from aim2dat.strct.analysis.coordination import _coordination_compare_sites
from aim2dat.chem_f import transform_list_to_dict, compare_formulas


def _compare_structures_ffprint(
    structure1,
    structure2,
    r_max,
    delta_bin,
    sigma,
    use_weights,
    use_legacy_smearing,
    distinguish_kinds,
):
    dict_type = "_kind_dict" if distinguish_kinds else "_element_dict"
    structures = [structure1, structure2]
    element_dicts = [getattr(strct, dict_type) for strct in structures]
    if set(element_dicts[0].keys()) != set(element_dicts[1].keys()):
        return 1.0
    else:
        ffprints = [
            strct.calc_ffingerprint(
                r_max=r_max,
                delta_bin=delta_bin,
                sigma=sigma,
                use_legacy_smearing=use_legacy_smearing,
                distinguish_kinds=distinguish_kinds,
            )
            for strct in structures
        ]
        return _ffingerprint_compare(element_dicts, ffprints, use_weights)


def _compare_structures_direct_comp(
    structure1,
    structure2,
    symprec,
    angle_tolerance,
    hall_number,
    no_idealize,
    return_standardized_structure,
    length_threshold,
    angle_threshold,
    position_threshold,
    distinguish_kinds,
):
    def check_lattice_p(parameters1, parameters2, permutation, eps):
        """Compare lattice parameters."""
        # We have the same length as the cell parameters:
        rel_deviation = [0.0] * len(parameters1)
        for par1_idx, (per_idx, par1) in enumerate(zip(permutation, parameters1)):
            rel_deviation[par1_idx] = abs((par1 - parameters2[per_idx]) / float(par1))
        if all([rel_dev < eps for rel_dev in rel_deviation]):
            return True, rel_deviation
        else:
            return False, rel_deviation

    def check_positions(elements, positions, permutation, eps):
        """Compare positions of the atoms."""
        # List of relative deviations:
        deviations = [None] * len(elements[0])
        # Stores the indices of those positions from positions[1] that have not yet
        # been matched with positions[0]
        free_indices = list(range(len(elements[0])))

        # Loop over elements and positions:
        for idx1, (el1, pos1) in enumerate(zip(elements[0], positions[0])):
            # Loop over unmatched indices from positions[1]
            for idx2 in free_indices:
                if el1 == elements[1][idx2]:
                    # Calculate the distance between the two atoms:
                    deviation = np.array(
                        [
                            [
                                pos - positions[1][idx2][per_idx]
                                for pos, per_idx in zip(pos1, permutation)
                            ],
                            [
                                1.0 - pos - positions[1][idx2][per_idx]
                                for pos, per_idx in zip(pos1, permutation)
                            ],
                        ]
                    )
                    dev_norm = min(np.linalg.norm(deviation[0]), np.linalg.norm(deviation[1]))
                    # If the deviation is below the threshold we add the position to
                    # matched_indices:
                    if dev_norm < eps:
                        deviations[idx1] = dev_norm
                        free_indices.remove(idx2)
                        # If one match is found it goes to the next position
                        break
            # If there are more free indices than positions left in positions[0] the
            # structures cannot match:
            if len(free_indices) > (len(elements[0]) - idx1):
                break

        # Create output:
        if len(free_indices) == 0:
            return True, deviations
        else:
            return False, deviations

    from warnings import warn

    warn(
        "This function is experimental and may produce false results.",
        UserWarning,
        2,
    )

    formulas = []
    cell_lengths = []
    cell_angles = []
    el_lists = []
    positions = []
    for strct in [structure1, structure2]:
        output = strct.calc_space_group(
            symprec=symprec,
            angle_tolerance=angle_tolerance,
            hall_number=hall_number,
            no_idealize=no_idealize,
            return_standardized_structure=return_standardized_structure,
        )
        # TODO Throw error if kinds is None
        # TODO include kinds in spglib interface.
        std_structure = Structure(**output["standardized_structure"])
        formulas.append(std_structure["chem_formula"])
        cell_lengths.append(std_structure["cell_lengths"])
        cell_angles.append(std_structure["cell_angles"])
        if distinguish_kinds:
            el_lists.append(std_structure["kinds"])
        else:
            el_lists.append(std_structure["elements"])
        positions.append(std_structure["scaled_positions"])

    if not compare_formulas(formulas[0], formulas[1], reduce_formulas=False):
        return False

    for per in list(itertools.permutations([0, 1, 2], 3)):
        check_cell_lengths = check_lattice_p(
            cell_lengths[0], cell_lengths[1], per, length_threshold
        )
        # We can skip their permutation if the cell lengths don't match
        if not check_cell_lengths[0]:
            continue
        check_cell_angles = check_lattice_p(cell_angles[0], cell_angles[1], per, angle_threshold)

        # We can skip the permutation if cell angles don't match
        if not check_cell_angles[0]:
            continue

        positions_match, pos_dev = check_positions(el_lists, positions, per, position_threshold)

        if check_cell_lengths[0] and check_cell_angles[0] and positions_match:
            return True
    return False


def _compare_structures_comp_sym(
    structure1, structure2, symprec, angle_tolerance, hall_number, return_standardized_structure
):
    formulas = []
    space_groups = []
    if any(el1 not in structure2.elements for el1 in set(structure1.elements)) or any(
        el2 not in structure1.elements for el2 in set(structure2.elements)
    ):
        return False
    for strct in [structure1, structure2]:
        sg_info = strct.calc_space_group(
            symprec=symprec,
            angle_tolerance=angle_tolerance,
            hall_number=hall_number,
            return_standardized_structure=return_standardized_structure,
        )
        formulas.append(transform_list_to_dict(sg_info["standardized_structure"]["elements"]))
        space_groups.append(sg_info["space_group"]["number"])
    if space_groups[0] != space_groups[1]:
        return False
    if not compare_formulas(formulas[0], formulas[1], reduce_formulas=False):
        return False
    return True


def _compare_structures_coord(
    structure1, structure2, distinguish_kinds, length_threshold, **cn_kwargs
):
    # def _check_connected(site1, site2, matched_sites):
    #     for neigh1 in site1["neighbors"]:
    #         for neigh2 in site2["neighbours"]:

    # TODO check this, allow only for molecules for now.
    same_chem_f = compare_formulas(
        structure1.chem_formula, structure2.chem_formula, reduce_formulas=True
    )
    if not same_chem_f:
        return False

    # Create neighbour lists for each site:
    comp_type = "kind" if distinguish_kinds else "element"
    neigh_lists = []
    coords = []
    for strct in [structure1, structure2]:
        neigh_list = []
        coords.append(structure1.calc_coordination(**cn_kwargs))  # TODO expose cn_kwargs
        for site in coords[-1]["sites"]:
            neigh_dict = {k: 0 for k in getattr(strct, f"_{comp_type}_dict")}
            for neigh in site["neighbours"]:
                neigh_dict[neigh[comp_type]] += 1
            neigh_list.append(neigh_dict)
        neigh_lists.append(neigh_list)
    print(neigh_lists)

    # First, we match each site that has the same element and neighbours:
    matched_sites = {i: [] for i in range(len(structure1))}
    for i, neighs1 in enumerate(neigh_lists[0]):
        for j, neighs2 in enumerate(neigh_lists[1]):
            if neighs1 == neighs2:
                matched_sites[i].append(j)
    print(matched_sites)
    if any(len(val) == 0 for val in matched_sites.values()):
        return False

    # Then we discard sites that don't have the same distances:
    for i in range(len(structure1)):
        new_matched_sites = []
        for j in matched_sites[i]:
            # if not _check_connected(coords[0]["sites"][i], coord[1]["sites"][j], matched_sites):
            #    continue
            if not _coordination_compare_sites(
                [structure1, structure2], [i, j], coords, distinguish_kinds, length_threshold
            ):  # TODO expose this
                continue
            new_matched_sites.append(j)
        matched_sites[i] = set(new_matched_sites)
    print(matched_sites)

    strct_factor = len(structure2) / len(structure1)
    multiplicities = [0] * len(structure1)
    for i, sites1 in matched_sites.items():
        if multiplicities[i] > 0:
            continue

        target_mult = len(matched_sites[i]) / strct_factor
        if target_mult == 0:
            return False
        elif target_mult == 1:
            multiplicities[i] = 1
        else:
            mult_to_add = [i]
            for j, sites2 in matched_sites.items():
                if i == j:
                    continue
                if multiplicities[j] > 0:
                    continue
                if sites1 == sites2:
                    mult_to_add.append(j)
            if len(mult_to_add) != target_mult:
                return False

            for j in mult_to_add:
                multiplicities[j] = target_mult
    print(multiplicities)
    if all(mult > 0 for mult in multiplicities):
        return True
    else:
        return False
