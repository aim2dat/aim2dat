"""
Miscellaneous functions for the StructureCollection class.
"""

# Standard library imports:
import itertools

# Third party library imports:
import numpy as np

# Internal library imports:
from aim2dat.utils.maths import calc_angle
from aim2dat.strct.strct_super_cell import _create_supercell_positions


def calculate_distance(
    structure, site_index1, site_index2, backfold_positions, use_supercell, r_max
):
    """Calculate distance."""
    if use_supercell:
        distance, _ = _calc_atomic_distance_sc(structure, site_index1, site_index2, r_max)
    else:
        distance, _ = _calc_atomic_distance(
            structure, site_index1, site_index2, backfold_positions
        )
    return None, distance


def calculate_angle(structure, site_index1, site_index2, site_index3, backfold_positions):
    """Calculate angle between three atomic positions."""
    _check_site_indices(structure, (site_index1, site_index2, site_index3))

    _, pos2 = _calc_atomic_distance(structure, site_index1, site_index2, backfold_positions)
    _, pos3 = _calc_atomic_distance(structure, site_index1, site_index3, backfold_positions)
    pos1 = np.array(structure["positions"][site_index1])
    return None, calc_angle(pos2 - pos1, pos3 - pos1) * 180.0 / np.pi


def calculate_dihedral_angle(
    structure, site_index1, site_index2, site_index3, site_index4, backfold_positions
):
    """Calculate dihedral angle between four atomic positions."""
    _check_site_indices(structure, (site_index1, site_index2, site_index3, site_index4))

    _, pos2 = _calc_atomic_distance(structure, site_index1, site_index2, backfold_positions)
    _, pos3 = _calc_atomic_distance(structure, site_index1, site_index3, backfold_positions)
    _, pos4 = _calc_atomic_distance(structure, site_index1, site_index4, backfold_positions)
    pos1 = np.array(structure["positions"][site_index1])
    n_vector1 = np.cross(pos2 - pos1, pos3 - pos2)
    n_vector2 = np.cross(pos3 - pos2, pos4 - pos3)
    return None, calc_angle(n_vector1, n_vector2) * 180.0 / np.pi


def _check_site_indices(structure, site_indices):
    site_indices = np.concatenate(site_indices, axis=None).flatten()
    if site_indices.dtype not in ["int32", "int64"]:
        raise TypeError("`site_index` needs to be of type int.")
    if len(structure["elements"]) <= site_indices.max():
        raise ValueError("`site_index` needs to be smaller than the number of sites.")


def _calc_reciprocal_cell(cell):
    """
    Calculate the reciprocal cell from the cell in 'real' space.

    Parameters
    ----------
    cell : list or np.array
        Nested 3x3 list of the cell vectors.

    Returns
    -------
    reciprocal_cell : list
        Nested 3x3 list of the cell vectors.
    """
    if isinstance(cell, (list, np.ndarray)):
        cell = np.array(cell).reshape((3, 3))
    else:
        raise TypeError("'cell' must be a list or numpy array.")
    cell_volume = abs(np.dot(np.cross(cell[0], cell[1]), cell[2]))
    reciprocal_cell = np.zeros((3, 3))
    for dir_idx in range(3):
        # We use negative indices here
        reciprocal_cell[dir_idx] = (
            2.0 * np.pi / cell_volume * np.cross(cell[dir_idx - 2], cell[dir_idx - 1])
        )
    return reciprocal_cell.tolist()


def _calc_atomic_distance(structure, site_indices1, site_indices2, backfold_positions):
    """Calculate distance between two atoms."""
    _check_site_indices(structure, (site_indices1, site_indices2))

    if isinstance(site_indices1, int):
        site_indices1 = [site_indices1]
    if isinstance(site_indices2, int):
        site_indices2 = [site_indices2]

    pos1 = np.array(structure["positions"])[site_indices1]
    pos2 = np.array(structure["positions"])[site_indices2]
    dist = np.linalg.norm(pos1 - pos2, axis=1)

    if structure["cell"] is not None and backfold_positions:
        fold_combs = np.array(list(itertools.product([0, -1, 1], repeat=3)))
        pos2_scaled = fold_combs + np.array(structure["scaled_positions"])[site_indices2, None, :]
        pos2_cart = (
            (np.array(structure["cell"]).T)
            .dot(pos2_scaled.reshape(-1, 3).T)
            .T.reshape(pos2_scaled.shape)
        )
        dist = np.linalg.norm(pos1[:, None, :] - pos2_cart, axis=2)
        pos2 = pos2_cart[np.arange(dist.shape[0]), dist.argmin(axis=1), :]
        dist = dist.min(axis=1)

    if len(site_indices1) == 1 and len(site_indices2) == 1:
        dist = dist[0]
        pos2 = pos2[0]

    return dist, pos2


def _calc_atomic_distance_sc(structure, site_indices1, site_indices2, r_max):
    """
    Calculate distance between two atoms, considering the
    replicates in a supercell.
    """
    _check_site_indices(structure, (site_indices1, site_indices2))

    if isinstance(site_indices1, int) and isinstance(site_indices2, int):
        site_indices1 = [site_indices1]
        site_indices2 = [site_indices2]
    elif isinstance(site_indices2, int):
        site_indices2 = [site_indices2] * len(site_indices1)
    elif isinstance(site_indices1, int):
        site_indices1 = [site_indices1] * len(site_indices2)

    dist_out = []
    pos_out = []

    for site_index1, site_index2 in zip(site_indices1, site_indices2):
        _, _, positions_sc, _, mapping, _ = _create_supercell_positions(structure, r_max)
        pos1 = np.array(structure["positions"][site_index1])
        mask = np.where(np.array(mapping) == site_index2, True, False)
        pos2 = np.array(positions_sc)[mask]
        dist = []
        for pos in pos2:
            dist.append(np.linalg.norm(np.array(pos1) - pos))
        zipped = list(zip(dist, pos2.tolist()))
        zipped.sort(key=lambda point: point[0])
        dist, pos = zip(*zipped)
        dist_out.append(dist)
        pos_out.append(pos)

    if len(site_indices1) == 1 and len(site_indices2) == 1:
        dist_out = dist_out[0]
        pos_out = pos_out[0]

    return dist_out, pos_out
