"""Internal manipulation methods dealing with the cell of the structure."""

# Standard library imports
from __future__ import annotations
import math
import itertools
from typing import List, TYPE_CHECKING, Union
import warnings

# Third party library imports
import numpy as np

# Internal library imports
from aim2dat.strct.manipulation.utils import _add_label_suffix
from aim2dat.utils.maths import calc_plane_equation

if TYPE_CHECKING:
    from aim2dat.strct.structure import Structure


def scale_unit_cell(
    structure,
    scaling_factors: Union[float, List[float]],
    pressure: float,
    bulk_modulus: float,
    random_factors: float,
    random_seed: int,
    change_label: bool,
) -> Union["Structure", dict]:
    """Scale the unit cell of a structure."""

    def get_scaling_matrix(scaling_factors):
        """Construct a 3x3 scaling matrix."""
        if isinstance(scaling_factors, (float, int)):
            return np.eye(3) * scaling_factors

        scaling_factors = np.array(scaling_factors)
        if not (
            np.issubdtype(scaling_factors.dtype, np.floating)
            or np.issubdtype(scaling_factors.dtype, np.integer)
        ):
            raise TypeError(
                "`scaling_factors` must be of type float/int or a list of float/int values."
            )
        elif scaling_factors.size == 9:
            return scaling_factors.reshape((3, 3))
        elif scaling_factors.size == 3:
            return np.eye(3) * scaling_factors
        raise ValueError(
            "`scaling_factors` must be a single value, a list of 3 values, or a 3x3 nested list."
        )

    if structure.cell is None:
        return None

    if pressure is not None:
        if bulk_modulus is None:
            raise ValueError("`bulk_modulus` must be provided when applying `pressure`.")
        scaling_factors = 1 - pressure / bulk_modulus
    if random_factors is not None:
        rng = np.random.default_rng(seed=random_seed)
        scaling_factors = np.array(
            [[0.0 if i < j else rng.random() - 0.5 for j in range(3)] for i in range(3)]
        ) * 2.0 * random_factors + np.eye(3)

    if scaling_factors is None:
        raise ValueError(
            "Provide either `scaling_factors`, `pressure` (with `bulk_modulus`) or "
            + "`random_factors`."
        )

    scaling_matrix = get_scaling_matrix(scaling_factors)

    scaled_cell = [
        [sum(row[k] * scaling_matrix[k][j] for k in range(3)) for j in range(3)]
        for row in structure["cell"]
    ]

    new_structure = structure.to_dict(cartesian=False)
    new_structure["cell"] = scaled_cell

    return _add_label_suffix(new_structure, f"_scaled-{scaling_factors}", change_label)


def create_supercell(
    structure: "Structure",
    size: Union[tuple, list, int],
    wrap: bool,
    change_label: bool,
) -> Union["Structure", dict]:
    """Create supercell."""
    if structure.cell is None:
        return None

    if not isinstance(size, (tuple, list, np.ndarray)):
        size = [size] * 3
    if len(size) != 3:
        raise ValueError("`size` must have a length of 3.")
    for i, (pbc, s) in enumerate(zip(structure.pbc, size)):
        try:
            s = int(s)
        except ValueError:
            raise TypeError("All entries of `size` must be integer numbers.")
        if s < 1:
            raise ValueError("All entries of `size` must be greater or equal to 1.")
        if not pbc and s > 1:
            warnings.warn(
                f"Direction {i} is non-periodic but `size{[i]}` is larger than 1. "
                + "This direction will be ignored.",
            )

    elements_sc, kinds_sc, positions_sc, indices_sc, mapping, rep_cells = (
        _create_supercell_positions(structure, r_max=None, size=size, wrap=wrap)
    )

    strct_dict = structure.to_dict(cartesian=True)
    strct_dict["cell"] = [
        [v * s if pbc else v for v in vect]
        for s, vect, pbc in zip(size, structure.cell, structure.pbc)
    ]
    strct_dict["elements"] = elements_sc
    strct_dict["kinds"] = kinds_sc
    strct_dict["positions"] = positions_sc
    strct_dict["site_attributes"] = {}
    site_attributes = structure.site_attributes
    for site_idx in mapping:
        for attr_key, attr_val in site_attributes.items():
            strct_dict["site_attributes"].setdefault(attr_key, []).append(
                site_attributes[attr_key][site_idx]
            )
    return _add_label_suffix(strct_dict, f"_supercell-{size}", change_label)


def _create_supercell_positions(
    structure: "Structure",
    r_max: float,
    size: Union[tuple, list] = None,
    indices: list = None,
    wrap: bool = True,
) -> tuple:
    if any(pbc0 for pbc0 in structure["pbc"]):
        translation_list = (
            _create_global_translation_list(structure, r_max, size)
            if indices is None
            else _create_index_translation_list(structure, indices, r_max)
        )
        rep_cells = np.repeat(translation_list, len(structure), axis=0)
        num_combinations = len(translation_list)
        positions_sc = (
            np.tile(
                structure.get_positions(cartesian=False, wrap=wrap),
                (len(translation_list), 1),
            )
            + rep_cells
        )
        positions_sc = positions_sc.dot(structure.cell)
        elements_sc = list(structure.elements) * num_combinations
        kinds_sc = (
            [None] * (len(structure) * num_combinations)
            if structure.kinds is None
            else list(structure.kinds) * num_combinations
        )
        mapping = list(range(len(structure))) * num_combinations
        indices_sc = [
            idx if trans_comb == (0, 0, 0) else -1
            for trans_comb in translation_list
            for idx in range(len(structure))
        ]

    else:
        elements_sc = structure["elements"]
        kinds_sc = structure["kinds"]
        positions_sc = structure["positions"]
        indices_sc = list(range(len(elements_sc)))
        mapping = indices_sc
        rep_cells = [np.array([0, 0, 0]) for el in structure["elements"]]
    return elements_sc, kinds_sc, positions_sc, indices_sc, mapping, rep_cells


def _create_global_translation_list(
    structure: "Structure", r_max: float, size: Union[tuple, list] = None
):
    translation_list = []
    for direction, pbc in enumerate(structure.pbc):
        if pbc:
            if r_max is None:
                translation_list.append(list(range(0, size[direction])))
            else:
                max_nr_trans = math.ceil(r_max / structure.cell_lengths[direction]) + 2
                translation_list.append(list(range(-max_nr_trans, max_nr_trans)))
        else:
            translation_list.append([0])
    return list(itertools.product(*translation_list))


def _create_index_translation_list(
    structure: "Structure", indices: Union[int, list], r_max: float
):
    translation_list = [[0], [0], [0]]

    # We pre-calculate the plane limiting the unit cell in each periodic direction and the
    # projected cell length:
    planes = {}
    for dir0, pbc in enumerate(structure.pbc):
        if not pbc:
            continue

        plane_points = [(0.0, 0.0, 0.0)] + [structure.cell[i] for i in range(3) if i != dir0]
        plane_p = calc_plane_equation(*plane_points)
        norm = np.linalg.norm(plane_p[:3])
        proj_cell_length = abs(np.dot(structure.cell[dir0], plane_p[:3]) / norm)
        planes[dir0] = (plane_p, norm, proj_cell_length)

    for index in indices:
        pos = structure.get_position(index, cartesian=True, wrap=True)
        for dir0, pbc in enumerate(structure.pbc):
            if not pbc:
                continue

            # Now, We calculate the left/right distance of the position to the limiting unitcell
            # planes:
            plane_points = [(0.0, 0.0, 0.0)] + [structure.cell[i] for i in range(3) if i != dir0]
            plane_p, norm, proj_cell_length = planes[dir0]
            dotp = np.dot(pos, plane_p[:3])
            distances = [
                abs(dotp + p3) / norm
                for p3 in [plane_p[3], -1.0 * sum(plane_p[:3] * np.array(structure.cell[dir0]))]
            ]

            # We take multiples of the projected cell length to estimate the number of periodic
            # replica in that direction.
            for i, dist in enumerate(distances):
                multiples = 0
                if dist < 1e-3:
                    dist += distances[i - 1]
                    multiples += 1
                elif dist > r_max:
                    continue

                multiples += math.ceil((r_max - dist) / proj_cell_length) + 1
                for mult in range(1, multiples):
                    mult = mult * -1 if i == 0 else mult
                    if mult not in translation_list[dir0]:
                        translation_list[dir0].append(mult)
    return list(itertools.product(*translation_list))
