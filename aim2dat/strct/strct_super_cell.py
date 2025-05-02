"""Module to create a supercell of an existing structure."""

# Standard library imports
from __future__ import annotations
import math
import itertools
from typing import Tuple, List, TYPE_CHECKING, Union

# Third party library imports
import numpy as np
from scipy.spatial import Voronoi


# Internal library imports
if TYPE_CHECKING:
    from aim2dat.strct.structure import Structure


def calc_voronoi_tessellation(structure: Structure, r_max: float) -> Tuple[None, List[List[dict]]]:
    """Calculate voronoi tessellation."""
    voronoi_list = []
    (
        elements_sc,
        kinds_sc,
        positions_sc,
        indices_sc,
        mapping,
        rep_cells,
    ) = _create_supercell_positions(structure, r_max)
    vor_scipy = Voronoi(positions_sc)
    for idx_sc, idx_uc in enumerate(indices_sc):
        if idx_uc < 0:
            continue
        neighbours = []
        position = np.array(structure.positions[idx_uc])
        for ridge_p, ridge_v in zip(vor_scipy.ridge_points, vor_scipy.ridge_vertices):
            if idx_sc in ridge_p:
                n_idx = ridge_p[1] if ridge_p[0] == idx_sc else ridge_p[0]
                shift = position - positions_sc[idx_sc]
                neighbours.append(
                    {
                        "index": mapping[n_idx],
                        "position": positions_sc[n_idx] + shift,
                        "element": elements_sc[n_idx],
                        "kind": kinds_sc[n_idx],
                        "replica": rep_cells[n_idx],
                        "vertices": [vor_scipy.vertices[idx] + shift for idx in ridge_v],
                    }
                )
        voronoi_list.append(
            sorted(
                neighbours,
                key=lambda neighbor: (neighbor["index"], *neighbor["position"].tolist()),
            )
        )
    return voronoi_list


def _create_supercell_positions(
    structure: Structure, r_max: float, size: Union[tuple, list] = None, wrap: bool = True
):
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
