"""
Miscellaneous functions for the StructureCollection class.
"""

# Standard library imports:
import itertools

# Third party library imports:
import numpy as np

# Internal library imports:
import aim2dat.utils.maths as a2d_maths
from aim2dat.strct.manipulation.cell import _create_supercell_positions


def calc_distance(
    structure,
    site_index1,
    site_index2,
    backfold_positions,
    use_supercell=False,
    r_max=0.0,
    return_pos=False,
):
    """Calculate distance."""
    if use_supercell:
        distance, pos = _calc_atomic_distance_sc(structure, site_index1, site_index2, r_max)
    else:
        distance, pos = _calc_atomic_distance(
            structure, site_index1, site_index2, backfold_positions
        )

    return (distance, pos) if return_pos else distance


def calc_angle(structure, site_index1, site_index2, site_index3, backfold_positions):
    """Calculate angle between three atomic positions."""
    comb_indices, is_int = _check_site_indices(structure, [site_index1, site_index2, site_index3])
    site_index1, site_index2, site_index3 = zip(*comb_indices)
    *_, positions = _calc_atomic_distance(
        structure, list(site_index1), list(site_index2) + list(site_index3), backfold_positions
    )
    output = []
    for idx1, idx2, idx3 in comb_indices:
        pos1 = np.array(structure["positions"][idx1])
        output.append(
            a2d_maths.calc_angle(positions[(idx1, idx2)] - pos1, positions[(idx1, idx3)] - pos1)
            * 180.0
            / np.pi
        )
    return _parse_calc_output(comb_indices, output, is_int)


def calc_dihedral_angle(
    structure, site_index1, site_index2, site_index3, site_index4, backfold_positions
):
    """Calculate dihedral angle between four atomic positions."""
    comb_indices, is_int = _check_site_indices(
        structure, [site_index1, site_index2, site_index3, site_index4]
    )
    site_index1, site_index2, site_index3, site_index4 = zip(*comb_indices)
    *_, positions = _calc_atomic_distance(
        structure,
        list(site_index1),
        list(site_index2) + list(site_index3) + list(site_index4),
        backfold_positions,
    )
    output = []
    for idx1, idx2, idx3, idx4 in comb_indices:
        pos1 = np.array(structure["positions"][idx1])
        n_vector1 = np.cross(
            positions[(idx1, idx2)] - pos1, positions[(idx1, idx3)] - positions[(idx1, idx2)]
        )
        n_vector2 = np.cross(
            positions[(idx1, idx3)] - positions[(idx1, idx2)],
            positions[(idx1, idx4)] - positions[(idx1, idx3)],
        )
        output.append(a2d_maths.calc_angle(n_vector1, n_vector2) * 180.0 / np.pi)
    return _parse_calc_output(comb_indices, output, is_int)


def _check_site_indices(structure, site_indices):
    def recursive_combinations(site_indices, curr_indices, final_indices):
        last = len(site_indices) == 1
        for i in range(len(site_indices[0])):
            if site_indices[0][i] not in curr_indices:
                new_indices = curr_indices.copy() + [site_indices[0][i]]
                if last:
                    if list(reversed(new_indices)) not in final_indices:
                        final_indices.append(new_indices)
                else:
                    recursive_combinations(site_indices[1:], new_indices, final_indices)

    is_int = True
    none_indices = []
    for idx, indices in enumerate(site_indices):
        if indices is None:
            site_indices[idx] = []
            none_indices.append(idx)
        elif isinstance(indices, (tuple, list, np.ndarray)):
            is_int = False
        elif isinstance(indices, int):
            site_indices[idx] = [indices]
        else:
            raise TypeError("`site_index` must be of type int, list, tuple, np.ndarray or None.")
    for idx in none_indices:
        site_indices[idx] = list(set([i for idx in site_indices for i in idx]))
    final_indices = []
    recursive_combinations(site_indices, [], final_indices)
    if any(i >= len(structure) for idx in final_indices for i in idx):
        raise ValueError("`site_index` needs to be smaller than the number of sites.")
    if any(not isinstance(i, int) for idx in final_indices for i in idx):
        raise TypeError("`site_index` must be of type int, list, tuple, np.ndarray or None.")
    return final_indices, is_int


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


def _parse_calc_output(comb_indices, output, is_int, positions=None):
    if is_int:
        if positions is None:
            return output[0]
        return output[0], positions[0]
    else:
        output = {tuple(idx): outp for idx, outp in zip(comb_indices, output)}
        if positions is None:
            return output
        positions = {tuple(idx): pos for idx, pos in zip(comb_indices, positions)}
        return output, positions


def _calc_atomic_distance(structure, site_indices1, site_indices2, backfold_positions):
    """Calculate distance between two atoms."""
    comb_indices, is_int = _check_site_indices(structure, [site_indices1, site_indices2])
    site_indices1, site_indices2 = zip(*comb_indices)
    pos1 = np.array(structure.get_positions(cartesian=True, wrap=backfold_positions))[
        list(site_indices1)
    ]
    pos2 = np.array(structure.get_positions(cartesian=True, wrap=backfold_positions))[
        list(site_indices2)
    ]
    dist = np.linalg.norm(pos1 - pos2, axis=1)

    if structure["cell"] is not None and backfold_positions:
        fold_combs = np.array(
            list(itertools.product(*[[0, -1, 1] if pbc else [0] for pbc in structure.pbc]))
        )
        pos2_scaled = (
            np.array(fold_combs)
            + np.array(structure.get_positions(cartesian=False, wrap=True))[site_indices2, None, :]
        )
        pos2_cart = (
            (np.array(structure["cell"]).T)
            .dot(pos2_scaled.reshape(-1, 3).T)
            .T.reshape(pos2_scaled.shape)
        )
        dist = np.linalg.norm(pos1[:, None, :] - pos2_cart, axis=2)
        pos2 = pos2_cart[np.arange(dist.shape[0]), dist.argmin(axis=1), :]
        dist = dist.min(axis=1)

    return _parse_calc_output(comb_indices, dist, is_int, pos2)


def _calc_atomic_distance_sc(structure, site_indices1, site_indices2, r_max):
    """
    Calculate distance between two atoms, considering the
    replicates in a supercell.
    """
    comb_indices, is_int = _check_site_indices(structure, [site_indices1, site_indices2])
    if isinstance(site_indices1, int):
        site_indices1 = [site_indices1]
    _, _, positions_sc, _, mapping, _ = _create_supercell_positions(
        structure, r_max, indices=site_indices1
    )

    dist_out = []
    pos_out = []

    for site_index1, site_index2 in comb_indices:
        pos1 = np.array(structure["positions"][site_index1])
        mask = np.where(np.array(mapping) == site_index2, True, False)
        pos2 = np.array(positions_sc)[mask]
        dist = []
        final_pos = []
        for pos in pos2:
            dist0 = np.linalg.norm(np.array(pos1) - pos)
            if dist0 <= r_max:
                dist.append(dist0)
                final_pos.append(pos.tolist())
        if len(dist) > 0:
            zipped = list(zip(dist, final_pos))
            zipped.sort(key=lambda point: point[0])
            dist, final_pos = zip(*zipped)
        else:
            dist = None
            final_pos = None
        dist_out.append(dist)
        pos_out.append(final_pos)
    return _parse_calc_output(comb_indices, dist_out, is_int, pos_out)
