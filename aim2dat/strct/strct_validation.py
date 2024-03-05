"""Validate structure input parameters."""

# Standard library imports
# import copy
import re
from typing import List, Tuple

# Third party library imports
import numpy as np

# Internal library imports
from aim2dat.utils.element_properties import get_element_symbol


def _structure_validate_elements(elements):
    if not isinstance(elements, (list, tuple, np.ndarray, str)):
        raise TypeError("`elements` must be a list or str.")
    if isinstance(elements, str):
        elements = re.findall("[A-Z][^A-Z]*", elements)
    if len(elements) == 0:
        raise ValueError("`elements` must have a length greater than 0.")
    return tuple([get_element_symbol(el) for el in elements])


def _structure_validate_cell(cell: List[List[float]]):
    if isinstance(cell, (tuple, list, np.ndarray)):
        cell = np.array(cell).reshape((3, 3))
        return tuple([tuple(v) for v in cell.tolist()]), tuple(
            [tuple(v) for v in np.linalg.inv(cell).tolist()]
        )
    else:
        raise TypeError("`cell` must be a list or numpy array for periodic boundaries.")


def _structure_validate_positions(positions, is_cartesian, cell, inv_cell, pbc):
    if not is_cartesian and cell is None:
        raise ValueError("`cell` must be set if `is_cartesian` is False.")
    positions_cart = []
    positions_scaled = []
    for position in positions:
        if len(position) != 3:
            raise ValueError("Length of one position must be 3.")

        # Old version
        # if is_cartesian and cell is not None:
        #     pos_scaled = np.transpose(inv_cell).dot(np.array(position))
        # else:
        #     pos_scaled = np.array(position)
        #
        # # Back-transform coordinates to the unit-cell:
        # if cell is not None:
        #     for direction in range(3):
        #         if pbc[direction]:
        #             while pos_scaled[direction] < 0.0:
        #                 pos_scaled[direction] += 1.0
        #             while pos_scaled[direction] >= 1.0:
        #                 pos_scaled[direction] -= 1.0
        #     positions_scaled.append(tuple(float(p) for p in pos_scaled))
        #     positions_cart.append(
        #         tuple(float(p) for p in np.transpose(cell).dot(np.array(pos_scaled)))
        #     )
        # else:
        #     positions_cart.append(tuple(float(p) for p in position))
        ###

        # New version
        if cell is None:
            positions_cart.append(tuple(float(p) for p in position))
        else:
            if is_cartesian:
                positions_cart.append(tuple(float(p) for p in position))
                positions_scaled.append(
                    tuple(float(p) for p in np.transpose(inv_cell).dot(np.array(position)))
                )
            else:
                positions_scaled.append(tuple(float(p) for p in position))
                positions_cart.append(
                    tuple(float(p) for p in np.transpose(cell).dot(np.array(position)))
                )
        ###
    if len(positions_scaled) == 0:
        return tuple(positions_cart), None
    else:
        return tuple(positions_cart), tuple(positions_scaled)


def _structure_validate_el_pos(
    elements: List[str],
    positions: List[List[float]],
    pbc: List[bool],
    cell: List[List[float]],
    inv_cell: List[List[float]],
    is_cartesian: bool,
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    if len(elements) != len(positions):
        raise ValueError("`elements` and `positions` must have the same length.")
    elements = _structure_validate_elements(elements)
    positions_cart, positions_scaled = _structure_validate_positions(
        positions, is_cartesian, cell, inv_cell, pbc
    )
    if cell is None:
        positions_scaled = None
    return elements, positions_cart, positions_scaled
