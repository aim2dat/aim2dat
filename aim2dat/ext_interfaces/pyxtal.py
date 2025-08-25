"""Interface to the PyXtaL python package."""

# Standard library imports
import itertools
import random
import math

# Third party library imports
import numpy as np
from pyxtal.tolerance import Tol_matrix
from pyxtal.crystal import random_crystal
from pyxtal.msg import Comp_CompatibilityError


SPACE_GROUP_LIMITS = {
    3: [
        ("triclinic", 0, 3),
        ("monoclinic", 2, 16),
        ("orthorhombic", 15, 75),
        ("tetragonal", 74, 143),
        ("trigonal", 142, 168),
        ("hexagonal", 167, 195),
        ("cubic", 194, 231),
    ]
}
NR_OF_SPACE_GROUPS = {
    0: 32,
    1: 75,
    2: 80,
    3: 230,
}


def _pyxtal_tolerance_matrix(tuples=None, molecular=False, factor=1.0):
    if molecular:
        prototype = "molecular"
    else:
        prototype = "atomic"
    if tuples is None:
        return Tol_matrix(prototype=prototype, factor=factor)
    else:
        return Tol_matrix(*tuples, prototype=prototype, factor=factor)


def _process_element_set(
    element_set,
    formula_series,
    bin_size,
    tol_matrix,
    space_group_list,
    crystal_sys_list,
    molecular,
    dimensions,
    excl_space_groups,
    max_structures,
    max_structures_per_cs,
    max_structures_per_sg,
    volume_factor,
) -> list:
    """
    Process a set of elements.
    """
    structures = []
    bins = np.arange(-0.5 * bin_size, 1.0 + 0.5 * bin_size, bin_size)
    permutations = [
        per0
        for per0 in itertools.product(bins, repeat=len(element_set) - 1)
        if sum(per0) < 1.0 + 0.5 * bin_size
    ]
    strct_bin = int(round(max_structures / ((len(bins) - 1) ** (len(element_set) - 1))))
    strct_p_cs = int(round(max_structures_per_cs / ((len(bins) - 1) ** (len(element_set) - 1))))
    strct_p_sg = int(round(max_structures_per_sg / ((len(bins) - 1) ** (len(element_set) - 1))))
    max_structures = [strct_bin] * len(bins)
    max_structures_per_cs = [strct_p_cs] * len(bins)
    max_structures_per_sg = [strct_p_sg] * len(bins)
    for max_strct_list in [max_structures, max_structures_per_cs, max_structures_per_sg]:
        max_strct_list[0] = math.floor(0.5 * max_strct_list[0])
        max_strct_list[-1] = math.ceil(0.5 * max_strct_list[-1])

    for max_strct, max_p_cs, max_p_sg, permutation in zip(
        max_structures, max_structures_per_cs, max_structures_per_sg, permutations
    ):
        space_group_list = [0] * NR_OF_SPACE_GROUPS[dimensions]
        crystal_sys_list = [0] * len(SPACE_GROUP_LIMITS[dimensions])

        print("Conc. interval: " + ", ".join(str(round(per0, 3)) for per0 in permutation) + ".")
        if sum(permutation) < 1.0:
            conc_list = []
            formula_subset = []
            for conc_idx, (conc_tuple, formula) in enumerate(formula_series):
                conc_dist = [round(con0 - per0, 12) for con0, per0 in zip(conc_tuple, permutation)]
                # print(conc_tuple, conc_dist)
                if all([conc0 > 0.0 for conc0 in conc_dist]) and sum(conc_dist) <= bin_size:
                    conc_list.append(conc_idx)
                    formula_subset.append(formula)
            # print(formula_subset)
            # print()

            for conc_idx in sorted(conc_list, reverse=True):
                del formula_series[conc_idx]
            structures += _create_crystals(
                formula_subset,
                tol_matrix,
                space_group_list,
                crystal_sys_list,
                molecular,
                dimensions,
                excl_space_groups,
                max_strct,
                max_p_cs,
                max_p_sg,
                volume_factor,
            )
        if all([per0 > 0.5 * bin_size for per0 in permutation]) or sum(permutation) > 1.0:
            conc_list = []
            formula_subset = []
            for conc_idx, (conc_tuple, formula) in enumerate(formula_series):
                conc_dist = [per0 - conc0 for conc0, per0 in zip(conc_tuple, permutation)]
                if all([conc0 >= 0.0 for conc0 in conc_dist]) and sum(conc_dist) < bin_size:
                    conc_list.append(conc_idx)
                    formula_subset.append(formula)
            # print(formula_subset)

            for conc_idx in sorted(conc_list, reverse=True):
                del formula_series[conc_idx]
            structures += _create_crystals(
                formula_subset,
                tol_matrix,
                space_group_list,
                crystal_sys_list,
                molecular,
                dimensions,
                excl_space_groups,
                max_strct,
                max_p_cs,
                max_p_sg,
                volume_factor,
            )
        print("  Total number of structures: ", len(structures))
    return structures


def _create_crystals(
    formulas,
    tol_matrix,
    space_group_list,
    crystal_sys_list,
    molecular,
    dimensions,
    excl_space_groups,
    max_structures,
    max_structures_per_cs,
    max_structures_per_sg,
    volume_factor,
) -> list:
    """
    Create the crystal structures.
    """
    structures = []
    if len(formulas) > 0:
        counter_tries = 0
        while len(structures) < max_structures and counter_tries < 10000:
            crystal_sys, space_group = _choose_crystal_system_and_space_group(
                dimensions,
                crystal_sys_list,
                space_group_list,
                max_structures_per_cs,
                max_structures_per_sg,
                excl_space_groups,
            )
            if crystal_sys is None:
                break
            formula_idx = int(random.random() * len(formulas))

            structure_pyxtal = None
            try:
                structure_pyxtal = random_crystal(
                    dim=3,
                    group=space_group,
                    species=list(formulas[formula_idx].keys()),
                    numIons=[int(val) for val in formulas[formula_idx].values()],
                    factor=volume_factor,
                    sites=None,
                    lattice=None,
                    conventional=True,
                    tm=tol_matrix,
                )
            except Comp_CompatibilityError:
                structure_pyxtal = None

            if structure_pyxtal is not None and structure_pyxtal.valid:
                space_group_list[space_group - 1] += 1
                crystal_sys_list[crystal_sys - 1] += 1
                structures.append(_parse_structure(structure_pyxtal))
            else:
                counter_tries += 1
    return structures


def _choose_crystal_system_and_space_group(
    dimensions,
    crystal_sys_list,
    space_group_list,
    max_structures_per_cs,
    max_structures_per_sg,
    excl_space_groups,
):
    # Check possible space groups:
    possible_space_groups = []
    possible_crystal_systems = []
    for sg_idx in range(NR_OF_SPACE_GROUPS[dimensions]):
        if (
            space_group_list[sg_idx] < max_structures_per_sg
            and sg_idx + 1 not in excl_space_groups
        ):
            crystal_sys = 1
            for sg_l_idx, sg_limit in enumerate(SPACE_GROUP_LIMITS[dimensions]):
                if sg_limit[1] < sg_idx + 1 < sg_limit[2]:
                    crystal_sys = sg_l_idx
            if crystal_sys_list[crystal_sys] < max_structures_per_cs:
                possible_space_groups.append(sg_idx + 1)
                possible_crystal_systems.append(crystal_sys + 1)
    if len(possible_space_groups) == 0 or len(possible_crystal_systems) == 0:
        return None, None

    sg_idx = int(random.random() * len(possible_space_groups))
    return possible_crystal_systems[sg_idx], possible_space_groups[sg_idx]


def _parse_structure(structure_pyxtal) -> dict:
    """Parse PyXtaL structure to input dict."""
    structure = {
        "cell": structure_pyxtal.lattice.matrix.tolist(),
        "elements": [],
        "positions": [],
        "is_cartesian": False,
        "pbc": [direction == 1 for direction in structure_pyxtal.PBC],
        "attributes": {"space_group": structure_pyxtal.group.number},
    }
    for site in structure_pyxtal.atom_sites:
        structure["elements"] += [site.specie] * site.multiplicity
        for coord in site.coords:
            # Project coords onto the unit cell
            for direction in range(3):
                while coord[direction] > 1.0:
                    coord[direction] = coord[direction] - 1.0
                while coord[direction] < 0.0:
                    coord[direction] = coord[direction] + 1.0
            structure["positions"].append(coord.tolist())
    return structure
