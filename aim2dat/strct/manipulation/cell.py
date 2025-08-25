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
    structure: "Structure", r_max: float, size: Union[tuple, list] = None, wrap: bool = True
) -> tuple:
    if any(pbc0 for pbc0 in structure["pbc"]):
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
        translational_combinations = list(itertools.product(*translation_list))
        rep_cells = np.repeat(translational_combinations, len(structure), axis=0)
        num_combinations = len(translational_combinations)
        positions_sc = (
            np.tile(
                structure.get_positions(cartesian=False, wrap=wrap),
                (len(translational_combinations), 1),
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
            for trans_comb in translational_combinations
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
