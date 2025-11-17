"""Find atoms that span a plane."""

# Standard library imports
from typing import List
import itertools

# Third party library imports
import numpy as np
import scipy
from scipy.spatial.distance import cdist

# Internal library imports
from aim2dat.strct.structure import Structure
from aim2dat.strct.ext_analysis.decorator import external_analysis_method
from aim2dat.utils.maths import calc_plane_equation
from aim2dat.strct.manipulation.cell import _create_supercell_positions


@external_analysis_method(attr_mapping=None)
def calc_planes(
    structure: Structure,
    r_max: float = 15.0,
    fragment: list = None,
    threshold: float = 0.05,
    margin: float = 1.0,
    vector_lengths: List[float] = None,
    min_nr_atoms: int = 5,
    use_scaled_coordinates: bool = False,
) -> list:
    """
    Find planar arangements of atoms in the structure.

    Parameters
    ----------
    structure : aim2dat.strct.Structure
        Structure object.
    r_max : float (optional)
        Cut-off value for the maximum distance between two atoms in angstrom.
    fragment : list or None (optional)
        Whether to restrict the search to a fragment of the structure.
    threshold : float (optional)
        Numerical threshold to consider an atom to be part of the plane.
    margin : float (optional)
        Margin between the plane vectors and outermost atoms.
    vector_lengths : list (optional)
        Absolute lengths of the plane vectors (overwrites ``margin``).
    use_scaled_coordinates : bool (optional)
        Whether to use scaled coordinates for the calculation.

    Returns
    -------
    list
        List of planes.
    """
    if isinstance(margin, (list, tuple)):
        if len(margin) != 4:
            raise ValueError("`margin` needs to have a length of 4.")
    elif isinstance(margin, (int, float)):
        margin = [margin] * 4
    else:
        raise TypeError("`margin` needs to be of type float, int, tuple or list.")
    elements_sc, _, positions_sc, indices_sc, mapping, _ = _create_supercell_positions(
        structure, r_max
    )
    dist_mask = [
        any(val <= r_max for val in row)
        for row in cdist(np.array(structure.positions), positions_sc).T
    ]
    positions = positions_sc
    elements = elements_sc
    # if use_scaled_coordinates:
    #    positions = structure["positions_scaled_np"]
    if fragment is None:
        fragment = list(range(len(structure["elements"])))
    frg_indices = [
        idx0 for idx0 in range(len(elements_sc)) if dist_mask[idx0] and mapping[idx0] in fragment
    ]

    # Make a list of all pairs of atom-pairs:
    plane_groups = []
    for pt1, pt2, pt3 in list(itertools.combinations(frg_indices, 3)):
        plane_p = calc_plane_equation(positions[pt1], positions[pt2], positions[pt3])
        plane_group = [pt1, pt2, pt3]
        for atom_idx in frg_indices:
            pos = positions[atom_idx]
            if (
                abs(sum([plane_p[idx] * pos[idx] for idx in range(3)]) + plane_p[3]) < threshold
                and atom_idx not in plane_group
            ):
                plane_group.append(atom_idx)
        if len(plane_group) >= min_nr_atoms:
            plane_groups.append((plane_group, [mapping[idx0] for idx0 in plane_group]))

    # Sort out subsets:
    final_plane_groups = []
    final_plane_groups_mapping = []
    indices_to_delete = []
    for plane_group, pg_mapping in plane_groups:
        is_subset = False
        for pg2_idx, pg_mapping2 in enumerate(final_plane_groups_mapping):
            if all(idx0 in pg_mapping2 for idx0 in pg_mapping):
                is_subset = True
            elif (
                all(idx0 in pg_mapping for idx0 in pg_mapping2)
                and pg2_idx not in indices_to_delete
            ):
                indices_to_delete.append(pg2_idx)
        if not is_subset:
            final_plane_groups.append(plane_group)
            final_plane_groups_mapping.append(pg_mapping)
    for idx in sorted(indices_to_delete, reverse=True):
        del final_plane_groups[idx]
    final_plane_groups.sort(key=lambda value: len(value), reverse=True)

    # Define the origin and the two plane vectors:
    planes = []
    for plane_g in final_plane_groups:
        plane = _construct_orthogonal_plane(
            plane_g, positions, elements, mapping, margin, vector_lengths
        )
        if use_scaled_coordinates and structure["cell"] is not None:
            plane["plane"] = [
                np.transpose(structure._inverse_cell).dot(np.array(pl0)).tolist()
                for pl0 in plane["plane"]
            ]
        planes.append(plane)
    return planes


def _construct_orthogonal_plane(plane_group, positions, elements, mapping, margin, vector_lengths):
    # Calculate plane based on least squares:
    positions = np.array([positions[idx0] for idx0 in plane_group])
    A = np.c_[positions[:, 0], positions[:, 1], np.ones(positions.shape[0])]
    C, res, _, _ = scipy.linalg.lstsq(A, positions[:, 2])

    # Calculate plane origin and plane vectors:
    distances = cdist(positions, positions)
    max_indices = np.unravel_index(np.argmax(distances, axis=None), distances.shape)
    origin_point = positions[max_indices[0]].copy()
    origin_point[2] = origin_point[0] * C[0] + origin_point[1] * C[1] + C[2]
    pos1 = positions[max_indices[1]].copy()
    pos1[2] = pos1[0] * C[0] + pos1[1] * C[1] + C[2]
    v2 = pos1 - origin_point
    dist2 = distances[max_indices[0], max_indices[1]]
    v2 /= np.linalg.norm(v2)
    v1 = _get_second_plane_vector(v2, C)

    # Include margin and add origine to plane vectors:
    dist1 = 0.0
    shift = 0.0
    for pos_idx, pos in enumerate(positions):
        if pos_idx in max_indices:
            continue
        distance = np.dot(pos - origin_point, v1)
        if distance < shift:
            shift = distance
        if distance > dist1:
            dist1 = distance
    origin_point += (shift - margin[0]) * v1
    origin_point -= margin[1] * v2
    if vector_lengths is not None:
        v1 *= vector_lengths[0]
        v2 *= vector_lengths[1]
    else:
        v1 *= dist1 - shift + margin[0] + margin[2]
        v2 *= dist2 + margin[1] + margin[3]
    point1 = origin_point + v1
    point2 = origin_point + v2

    # Create output-lists:
    site_indices = []
    elements_plane = []
    positions_plane = []
    for at_idx, pos in zip(plane_group, positions):
        site_indices.append(mapping[at_idx])
        elements_plane.append(elements[at_idx])
        positions_plane.append(pos - origin_point)

    # Calculate projected positions:
    proj_matrix = np.array([v1 / np.linalg.norm(v1), v2 / np.linalg.norm(v2)])
    proj_positions = []
    for el, pos in zip(elements_plane, positions_plane):
        proj_pos = np.dot(proj_matrix, pos).tolist()
        proj_positions.append({"label": el, "x": proj_pos[0], "y": proj_pos[1]})

    return {
        "plane": [origin_point.tolist(), point1.tolist(), point2.tolist()],
        "site_indices": site_indices,
        "elements": elements_plane,
        "proj_positions": proj_positions,
    }


def _get_second_plane_vector(v1, C):
    if abs(v1[1]) > 1e-3:
        v2x = -1.0 * v1[1]
    elif abs(v1[0]) > 1e-3:
        v2x = v1[0]
    else:
        raise ValueError("Cannot find orthogonal plane vectors.")
    v2y = -1.0 * v2x * (v1[0] + C[0] * v1[2])
    v2y /= v1[1] + C[1] * v1[2]
    v2z = -1.0 * (v1[0] * v2x + v1[1] * v2y) / v1[2]
    v2 = np.array([v2x, v2y, v2z])
    v2 /= np.linalg.norm(v2)
    return v2
