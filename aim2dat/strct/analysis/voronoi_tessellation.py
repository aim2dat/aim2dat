"""Voronoi tessellation."""

# Standard library imports
from typing import TYPE_CHECKING, Tuple, List

# Third party library imports
import numpy as np
from scipy.spatial import Voronoi

# Internal library imports
from aim2dat.strct.manipulation.cell import _create_supercell_positions

if TYPE_CHECKING:
    from aim2dat.strct.structure import Structure


def calc_voronoi_tessellation(
    structure: "Structure", r_max: float
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
        voronoi_list.append(
            sorted(
                neighbours,
                key=lambda neighbor: (neighbor["index"], *neighbor["position"].tolist()),
            )
        )
    return voronoi_list
