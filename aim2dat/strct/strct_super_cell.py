"""Module to create a supercell of an existing structure."""

# Standard library imports
from __future__ import annotations
import math
import itertools
from typing import Tuple, List, TYPE_CHECKING

# Third party library imports
import numpy as np
from scipy.spatial import Voronoi


# Internal library imports
if TYPE_CHECKING:
    from aim2dat.strct.structure import Structure


def calculate_voronoi_tessellation(
    structure: Structure, r_max: float
) -> Tuple[None, List[List[dict]]]:
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
        voronoi_list.append(neighbours)
    return None, voronoi_list


def _create_supercell_positions(
    structure: Structure, r_max: float
) -> Tuple[List[str], List[str], List[List[float]], List[int], List[np.ndarray[:, :]]]:
    """Create supercell to calculate the distances to the periodic image atoms."""
    if any(pbc0 for pbc0 in structure["pbc"]):
        elements_uc = structure["elements"]
        kinds_uc = [None] * len(elements_uc) if structure["kinds"] is None else structure["kinds"]
        positions_scaled_uc = structure.get_positions(cartesian=False, wrap=True)
        translation_list = []

        for direction in range(3):
            if structure["pbc"][direction]:
                max_nr_trans = math.ceil(r_max / structure["cell_lengths"][direction]) + 2
                translation_list.append(list(range(-max_nr_trans, max_nr_trans)))
            else:
                translation_list.append([0])
        translational_combinations = list(itertools.product(*translation_list))
        num_combinations = len(translational_combinations)
        elements_sc = []
        kinds_sc = []
        positions_sc = []
        indices_sc = []
        mapping = []
        rep_cells = []

        translational_combinations = np.array(translational_combinations).T

        for idx0, (element, kind, position) in enumerate(
            zip(elements_uc, kinds_uc, positions_scaled_uc)
        ):
            positions_sc.extend(
                (
                    np.transpose(structure["cell"]).dot(
                        np.array(position).reshape(3, 1) + translational_combinations
                    )
                ).T
            )
            elements_sc.extend([element] * num_combinations)
            kinds_sc.extend([kind] * num_combinations)
            mapping.extend([idx0] * num_combinations)
            rep_cells.extend(translational_combinations.T)
            index_sc = -1 * np.ones(num_combinations, dtype=int)
            index_sc = np.where(np.all(translational_combinations == 0, axis=0), idx0, index_sc)
            indices_sc.extend(index_sc.tolist())
    else:
        elements_sc = structure["elements"]
        kinds_sc = structure["kinds"]
        positions_sc = structure["positions"]
        indices_sc = list(range(len(elements_sc)))
        mapping = indices_sc
        rep_cells = [np.array([0, 0, 0]) for el in structure["elements"]]
    return elements_sc, kinds_sc, positions_sc, indices_sc, mapping, rep_cells
