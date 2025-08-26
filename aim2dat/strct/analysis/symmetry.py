"""Analyse the symmetry of moelecules."""

# Standard library imports
import itertools

# Third party library imports
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation
import numpy as np

# Internal library imports
from aim2dat.elements import get_atomic_mass
from aim2dat.utils.maths import calc_angle
from aim2dat.ext_interfaces import _return_ext_interface_modules


def determine_space_group(*args, **kwargs):
    """Determine space group."""
    return _return_ext_interface_modules("spglib")._space_group_analysis(*args, **kwargs)


def determine_point_group(structure, threshold_distance, threshold_angle, threshold_inertia):
    """Determine point group."""
    symmetry_elements = []
    is_spherical = False
    point_group = r"$C_{1}$"
    elements = np.array(structure["elements"])
    positions = _shift_positions_to_com(elements, np.array(structure["positions"]))

    # Check inversion center:
    inv_matrix = np.zeros((3, 3))
    np.fill_diagonal(inv_matrix, -1.0)
    has_inversion = _check_sym_operation(inv_matrix, elements, positions, threshold_distance)
    if has_inversion:
        symmetry_elements.append({"label": "inv_center", "type": "inversion"})
        point_group = r"$C_i$"

    # Detect symetrically equivalent atoms:
    sea_list = _detect_sea(elements, positions, threshold_distance)

    # Calculate inertia eigenstates:
    i_evalues, i_evectors, inertia_type = _calculate_inertia_ev(
        elements, positions, threshold_inertia
    )

    # Check linear orientation:
    # For linear molecules the only options are with and without inversion center
    if inertia_type == "linear":
        symmetry_elements.append(
            {"label": "linear", "type": "linear", "axis": i_evectors[0].tolist()}
        )
        if has_inversion:
            point_group = r"$D_{\infty h}$"
        else:
            point_group = r"$C_{\infty v}$"
    # elif len(sea_list) == len(elements) and "polygonal" in inertia_type and len(elements)
    # > 2:
    # Cs planar
    #    point_group = r"$C_s$"
    else:
        for sea_idx, sea_set in enumerate(sea_list):
            if len(sea_set) == 2:
                # Check for C2:
                new_sym_els = _find_c2_axis(
                    sea_list, sea_idx, elements, positions, threshold_distance, threshold_angle
                )
                _filter_duplicate_sym_elements(symmetry_elements, new_sym_els, threshold_angle)
            elif len(sea_set) > 2:
                # Check for higher Cn
                el_sea = np.array([elements[idx] for idx in sea_set])
                pos_sea = np.array([positions[idx] for idx in sea_set])
                i_evalues_sea, i_evectors_sea, inertia_type = _calculate_inertia_ev(
                    el_sea, pos_sea, threshold_inertia
                )
                if inertia_type == "regular polygonal":
                    # Take eigenvector c only?
                    new_sym_els = _find_cn_axis(
                        i_evectors_sea, len(sea_set), elements, positions, threshold_distance
                    )
                elif inertia_type == "irregular polygonal":
                    # Take eigenvector c only?
                    new_sym_els = _find_cn_axis(
                        i_evectors_sea, len(sea_set) - 1, elements, positions, threshold_distance
                    )
                elif inertia_type == "prolate polyhedral":
                    if len(sea_set) % 2 != 0:
                        raise ValueError("Found prolate polyhedron in odd SEA set.")
                    # Take eigenvector a only?
                    new_sym_els = _find_cn_axis(
                        i_evectors_sea,
                        int(len(sea_set) / 2),
                        elements,
                        positions,
                        threshold_distance,
                    )
                elif inertia_type == "oblate polyhedral":
                    if len(sea_set) % 2 != 0:
                        raise ValueError("Found prolate polyhedron in odd SEA set.")
                    # Take eigenvector c only?
                    new_sym_els = _find_cn_axis(
                        i_evectors_sea,
                        int(len(sea_set) / 2),
                        elements,
                        positions,
                        threshold_distance,
                    )
                elif inertia_type == "asymmetric polyhedral":
                    # Take eigenvector c only?
                    new_sym_els = _find_cn_axis(
                        i_evectors_sea, 2, elements, positions, threshold_distance
                    )
                elif inertia_type == "spherical rotor":
                    new_sym_els = _find_c2_axis(
                        sea_list, sea_idx, elements, positions, threshold_distance, threshold_angle
                    )
                    for nfold in range(3, 6):
                        new_sym_els += _find_cn_axis_spherical(
                            sea_list,
                            nfold,
                            elements,
                            positions,
                            threshold_distance,
                            threshold_angle,
                        )
                else:
                    print("Could not find symmetry.")
                    return None, {}
                _filter_duplicate_sym_elements(symmetry_elements, new_sym_els, threshold_angle)

        # Check for perpendicular C2 axis:
        nfold_max = 0
        main_rot_axis = None
        for sym_el in symmetry_elements:
            if "rotation" == sym_el["type"] and sym_el["n-fold"] > nfold_max:
                nfold_max = sym_el["n-fold"]
                main_rot_axis = sym_el["axis"]
        if not is_spherical and nfold_max > 1:
            for set_index in range(len(sea_list)):
                if len(sea_list[set_index]) < 2:
                    continue
                new_sym_els = _find_c2_axis(
                    sea_list,
                    set_index,
                    elements,
                    positions,
                    threshold_distance,
                    threshold_angle,
                    normal_vector=main_rot_axis,
                )
                _filter_duplicate_sym_elements(symmetry_elements, new_sym_els, threshold_angle)

        # Check for reflection planes:
        reflection_planes = _find_reflection_planes(
            sea_list, elements, positions, threshold_distance
        )
        # Add reflection plane in case molecule is planar:
        if (
            abs(i_evalues[0] + i_evalues[1] - i_evalues[2]) < threshold_inertia
            and inertia_type != "linear"
        ):
            reflection_planes.append(
                {"type": "reflection", "label": r"$\sigma$", "n-vector": i_evectors[:, 2].tolist()}
            )
        _filter_duplicate_sym_elements(symmetry_elements, reflection_planes, threshold_angle)

        # Derive point group from symmetry elements:
        point_group = _derive_point_group(symmetry_elements, threshold_angle)
    return {"point_group": point_group, "symmetry_elements": symmetry_elements}


def _detect_sea(elements, positions, threshold_distance):
    """Find symmetrically equivalent atoms."""
    # Calculate distance matrix
    dist_matrix = cdist(positions, positions)
    el_tuples = [[(el1, el2) for el2 in elements] for el1 in elements]
    # Find symmetrically equivalent atoms (sea)
    sea_list = []
    sorted_tuples = []
    for at_idx, (distances, el_tuple) in enumerate(zip(dist_matrix, el_tuples)):
        zipped = list(zip(distances, el_tuple))
        zipped.sort(key=lambda point: point[0])
        distances_s, el_tuple_s = zip(*zipped)
        distances_s = np.array(distances_s)
        # Check if equivalent atom exists:
        is_equal = False
        for prev_idx, prev_tuple in enumerate(sorted_tuples):
            # Check if all distances are below the threshold and tuples are the same:
            if (
                all([abs(val) < threshold_distance for val in prev_tuple[0] - distances_s])
                and prev_tuple[1] == el_tuple_s
            ):
                is_equal = True
                sea_list[prev_idx].append(at_idx)
                break
        if not is_equal:
            sea_list.append([at_idx])
            sorted_tuples.append([distances_s, el_tuple_s])
    return sea_list


def _shift_positions_to_com(elements, positions):
    """Shift atom positions such that the center of mass is at (0.0, 0.0, 0.0)."""
    masses = np.array([get_atomic_mass(el) for el in elements])
    center_of_mass = sum(masses[:, np.newaxis] * positions) / np.sum(masses)
    return positions - center_of_mass


def _calculate_inertia_ev(elements, positions, threshold_inertia):
    """Calculate eigenvalues and eigenvectors of inertia."""
    masses = np.array([get_atomic_mass(el) for el in elements])
    positions = _shift_positions_to_com(elements, positions)
    pos_t = positions.T
    inertia_matrix = np.zeros((3, 3))
    permutations = list(itertools.combinations_with_replacement([0, 1, 2], 2))
    for perm in permutations:
        if perm[0] == perm[1]:
            inertia_matrix[perm[0]][perm[0]] = np.sum(
                masses * (pos_t[perm[0] - 2] ** 2 + pos_t[perm[0] - 1] ** 2)
            )
        else:
            inertia_cont = np.sum(masses * pos_t[perm[0]] * pos_t[perm[1]])
            inertia_matrix[perm[0]][perm[1]] -= inertia_cont
            inertia_matrix[perm[1]][perm[0]] -= inertia_cont
    evalues, evectors = np.linalg.eigh(inertia_matrix)

    inertia_type = ""
    diff01 = abs(evalues[0] - evalues[1])
    diff02 = abs(evalues[0] - evalues[2])
    diff12 = abs(evalues[1] - evalues[2])
    if abs(evalues[0]) < threshold_inertia and diff12 < threshold_inertia:
        inertia_type = "linear"
    elif abs(evalues[0] + evalues[1] - evalues[2]) < threshold_inertia:
        if diff01 < threshold_inertia:
            inertia_type = "regular polygonal"
        else:
            inertia_type = "irregular polygonal"
    elif diff12 < threshold_inertia and abs(diff01) >= threshold_inertia:
        inertia_type = "prolate polyhedral"
    elif diff01 < threshold_inertia and abs(diff02) >= threshold_inertia:
        inertia_type = "oblate polyhedral"
    elif diff01 >= threshold_inertia and diff12 >= threshold_inertia:
        inertia_type = "asymmetric polyhedral"
    elif diff01 < threshold_inertia and diff12 < threshold_inertia:
        inertia_type = "spherical rotor"
    return evalues, evectors, inertia_type


def _check_sym_operation(matrix, elements, positions, threshold_distance):
    """Check whether a symmetry operations results in the same molecule."""
    has_symmetry = True
    elements_new = list(elements)
    positions_new = (matrix.dot(positions.T)).T
    for el, pos in zip(elements, positions):
        pos_found = False
        for pos_idx, (el_new, pos_new) in enumerate(zip(elements_new, positions_new)):
            if np.linalg.norm(pos - pos_new) < threshold_distance and el_new == el:
                pos_found = True
                positions_new = np.delete(positions_new, pos_idx, axis=0)
                elements_new.pop(pos_idx)
                break
        if not pos_found:
            has_symmetry = False
            break
    return has_symmetry


def _find_cn_axis(rot_axis, max_nfold, elements, positions, threshold_distance):
    """Check rotational symmetry for an axis-vector."""
    sym_elements = []
    for nfold in range(2, max_nfold + 1):
        for dir0 in range(3):
            rotation = Rotation.from_rotvec((2.0 * np.pi / nfold) * rot_axis[:, dir0])
            if _check_sym_operation(rotation.as_matrix(), elements, positions, threshold_distance):
                sym_elements.append(
                    {
                        "axis": rot_axis[:, dir0].tolist(),
                        "n-fold": nfold,
                        "label": f"$C_{nfold}$",
                        "type": "rotation",
                    }
                )
    return sym_elements


def _find_c2_axis(
    sea_list,
    set_index,
    elements,
    positions,
    threshold_distance,
    threshold_angle,
    normal_vector=None,
):
    """Check for 2-fold rotational axis."""

    def normal_vector_check(normal_vector, rot_axis):
        check_vector = False
        angle = calc_angle(normal_vector, rot_axis)
        cond1 = abs(180.0 * angle / np.pi - 90.0) < threshold_angle
        cond2 = abs(180.0 * angle / np.pi - 270.0) < threshold_angle
        if cond1 or cond2:
            check_vector = True
        return check_vector

    sea_set = sea_list[set_index]
    sym_elements = []
    check_vector = True

    # Check if middle point gives axis:
    for atom_index1 in sea_set[1:]:
        for atom_index2 in sea_set[:atom_index1]:
            mid_point = (positions[atom_index1] + positions[atom_index2]) / 2.0
            distance = np.linalg.norm(mid_point)
            if abs(distance) >= threshold_distance:
                if normal_vector is not None:
                    check_vector = normal_vector_check(normal_vector, mid_point)
                if check_vector:
                    rot_axis = mid_point / distance
                    rotation = Rotation.from_rotvec(np.pi * rot_axis)
                    if _check_sym_operation(
                        rotation.as_matrix(), elements, positions, threshold_distance
                    ):
                        # TO-DO Check for dihedral groups with higher Cn rotations.
                        sym_elements.append(
                            {
                                "axis": rot_axis.tolist(),
                                "n-fold": 2,
                                "label": r"$C_2$",
                                "type": "rotation",
                            }
                        )

    # Check if sym axis passes through atom:
    for atom_index in sea_set:
        distance = np.linalg.norm(positions[atom_index])
        # TO-DO check if a different threshold parameter might be needed:
        if distance >= threshold_distance:
            rot_axis = positions[atom_index] / distance
            if normal_vector is not None:
                check_vector = normal_vector_check(normal_vector, rot_axis)
            if check_vector:
                rotation = Rotation.from_rotvec(np.pi * rot_axis)
                if _check_sym_operation(
                    rotation.as_matrix(), elements, positions, threshold_distance
                ):
                    # TO-DO Check for dihedral groups with higher Cn rotations.
                    sym_elements.append(
                        {
                            "axis": rot_axis.tolist(),
                            "n-fold": 2,
                            "label": r"$C_2$",
                            "type": "rotation",
                        }
                    )

    # Check for other diatomic SEA sets:
    if len(sym_elements) < 2:
        vector1 = positions[sea_set[0]] - positions[sea_set[1]]
        for sea_set_2 in sea_list[:set_index]:
            if len(sea_set_2) == 2:
                vector2 = positions[sea_set_2[0]] - positions[sea_set_2[1]]
                rot_axis = np.cross(vector1, vector2)
                rot_norm = np.linalg.norm(rot_axis)
                if rot_norm >= threshold_distance:
                    rot_axis /= rot_norm
                    if normal_vector is not None:
                        check_vector = normal_vector_check(normal_vector, rot_axis)
                    if check_vector:
                        rotation = Rotation.from_rotvec(np.pi * rot_axis)
                        if _check_sym_operation(
                            rotation.as_matrix(), elements, positions, threshold_distance
                        ):
                            # TO-DO Check for dihedral groups with higher Cn rotations.
                            sym_elements.append(
                                {
                                    "axis": rot_axis.tolist(),
                                    "n-fold": 2,
                                    "label": r"$C_2$",
                                    "type": "rotation",
                                }
                            )
    return sym_elements


def _find_cn_axis_spherical(
    sea_list, nfold, elements, positions, threshold_distance, threshold_angle
):
    """Find rotational axis in spherical molecules."""
    # TO-DO limit search to the maximum of possible rotation axis
    sym_elements = []
    for sea_set in sea_list:
        if len(sea_set) >= nfold:
            atom_tuples = list(itertools.combinations(sea_set, nfold))
            for atom_tuple in atom_tuples:
                vectors = [
                    positions[atom_tuple[idx - 1]] - positions[atom_tuple[idx]]
                    for idx in range(nfold)
                ]

                # Check if the vectors are on one plane:
                n_vectors = [np.cross(vectors[idx - 1], vectors[idx]) for idx in range(nfold - 1)]
                is_planar = True
                for n_vec in n_vectors[1:]:
                    angle = calc_angle(n_vectors[0], n_vec)
                    # Conditions for angle to be not the same:
                    cond1 = abs(180.0 * angle / np.pi) >= threshold_angle
                    cond2 = abs(180.0 * angle / np.pi - 180.0) >= threshold_angle
                    cond3 = abs(180.0 * angle / np.pi - 360.0) >= threshold_angle
                    if cond1 and cond2 and cond3:
                        is_planar = False
                        break

                if is_planar:
                    # Check if all distances are the same:
                    norms = [np.linalg.norm(vector) for vector in vectors]
                    if all([abs(norm - norms[0]) < threshold_distance for norm in norms[1:]]):
                        is_regular = True
                        angles = [
                            calc_angle(vectors[idx - 1], vectors[idx]) for idx in range(nfold - 1)
                        ]
                        for angle in angles:
                            cond1 = abs(180.0 * angle / np.pi - 360.0 / nfold) >= threshold_angle
                            cond2 = (
                                abs(abs(180.0 * angle / np.pi - 360.0 / nfold) - 180.0)
                                >= threshold_angle
                            )
                            cond3 = (
                                abs(abs(180.0 * angle / np.pi - 360.0 / nfold) - 360.0)
                                >= threshold_angle
                            )
                            if cond1 and cond2 and cond3:
                                is_regular = False
                                break
                        if is_regular:
                            rot_axis = np.cross(vectors[0], vectors[1])
                            rot_axis /= np.linalg.norm(rot_axis)
                            rotation = Rotation.from_rotvec(2.0 * np.pi / nfold * rot_axis)
                            if _check_sym_operation(
                                rotation.as_matrix(), elements, positions, threshold_distance
                            ):
                                # TO-DO Check for dihedral groups with higher Cn rotations.
                                sym_elements.append(
                                    {
                                        "axis": rot_axis.tolist(),
                                        "n-fold": nfold,
                                        "label": f"$C_{nfold}$",
                                        "type": "rotation",
                                    }
                                )
    return sym_elements


def _find_reflection_planes(sea_list, elements, positions, threshold_distance):
    """Find relfection planes."""
    # Based on paper :
    # TO-DO restrict maximum of symmetry operations
    sym_elements = []
    for sea_set in sea_list:
        if len(sea_set) > 1:
            for sea_idx1 in sea_set[:-1]:
                for sea_idx2 in sea_set[1:]:
                    if elements[sea_idx1] != elements[sea_idx2]:
                        continue
                    n_vector = positions[sea_idx1] - positions[sea_idx2]
                    norm = np.linalg.norm(n_vector)
                    if norm < threshold_distance:
                        continue
                    n_vector /= norm
                    sigma_matrix = np.zeros((3, 3))
                    for dir0 in range(3):
                        sigma_matrix[dir0, dir0] = 1.0 - 2.0 * n_vector[dir0] ** 2.0
                        sigma_matrix[dir0, dir0 - 2] = -2.0 * n_vector[dir0] * n_vector[dir0 - 2]
                        sigma_matrix[dir0 - 2, dir0] = -2.0 * n_vector[dir0] * n_vector[dir0 - 2]
                    if _check_sym_operation(sigma_matrix, elements, positions, threshold_distance):
                        sym_elements.append(
                            {
                                "type": "reflection",
                                "label": r"$\sigma$",
                                "n-vector": n_vector.tolist(),
                            }
                        )
    return sym_elements


def _filter_duplicate_sym_elements(symmetry_elements, new_sym_els, threshold_angle):
    """Check if a symmetry element is already previously detected."""
    for new_el in new_sym_els:
        not_duplicate = True
        for old_el in symmetry_elements:
            # Check rotations:
            if old_el["type"] == "rotation" and new_el["type"] == "rotation":
                if new_el["n-fold"] != old_el["n-fold"]:
                    continue
                angle = calc_angle(old_el["axis"], new_el["axis"])
                cond1 = abs(180.0 * angle / np.pi) < threshold_angle
                cond2 = abs(180.0 * angle / np.pi - 180.0) < threshold_angle
                cond3 = abs(180.0 * angle / np.pi - 360.0) < threshold_angle
                if cond1 or cond2 or cond3:
                    not_duplicate = False
                    break
            # Check reflection planes:
            if old_el["type"] == "reflection" and new_el["type"] == "reflection":
                angle = calc_angle(old_el["n-vector"], new_el["n-vector"])
                cond1 = abs(180.0 * angle / np.pi) < threshold_angle
                cond2 = abs(180.0 * angle / np.pi - 180.0) < threshold_angle
                cond3 = abs(180.0 * angle / np.pi - 360.0) < threshold_angle
                if cond1 or cond2 or cond3:
                    not_duplicate = False
                    break
        if not_duplicate:
            symmetry_elements.append(new_el)


def _derive_point_group(symmetry_elements, threshold_angle):
    """Decision tree to detect the point group."""
    point_group = r"$C_{1}$"
    # Number of n-fold rotational symmetries with n>2:
    nfold_axis = [sym_el["n-fold"] for sym_el in symmetry_elements if sym_el["type"] == "rotation"]
    nfold_max = 0
    nr_nfold_max = 0
    if len(nfold_axis) > 0:
        nfold_max = max(nfold_axis)
        nr_nfold_max = sum([1 for el in nfold_axis if el == nfold_max])
    # Inversion:
    has_inversion = any(["inversion" == sym_el["type"] for sym_el in symmetry_elements])

    # Decision tree from "Group Theory and Chemistry" by David M. Bishop:
    if nfold_max > 2 and nr_nfold_max > 1:
        if has_inversion:
            if any(
                [
                    ("rotation" == sym_el["type"] and sym_el["n-fold"] == 5)
                    for sym_el in symmetry_elements
                ]
            ):
                point_group = r"$I_h$"
            else:
                point_group = r"$O_h$"
        else:
            point_group = r"$T_d$"
    else:
        if any(["rotation" == sym_el["type"] for sym_el in symmetry_elements]):
            # Find rotation with highest n:
            nfold_max = 0
            main_rotation = None
            for sym_el in symmetry_elements:
                if "rotation" == sym_el["type"] and sym_el["n-fold"] > nfold_max:
                    nfold_max = sym_el["n-fold"]
                    main_rotation = sym_el
            # Check the number of 2-fold rotations perpendicular to the rotation with higest n:
            has_c2_normal, c2_normal_axis = _check_c2_axis_normal_to_cn(
                main_rotation, symmetry_elements, threshold_angle
            )
            if has_c2_normal:
                if _check_h_reflection_planes(main_rotation, symmetry_elements, threshold_angle):
                    point_group = r"$D_{" + str(nfold_max) + r"h}$"
                else:
                    if _check_d_reflection_planes(
                        main_rotation, symmetry_elements, c2_normal_axis, threshold_angle
                    ):
                        point_group = r"$D_{" + str(nfold_max) + r"d}$"
                    else:
                        point_group = r"$D_{" + str(nfold_max) + r"}$"
            else:
                if _check_h_reflection_planes(main_rotation, symmetry_elements, threshold_angle):
                    point_group = r"$C_{" + str(nfold_max) + r"h}$"
                else:
                    if _check_v_reflection_planes(
                        main_rotation, symmetry_elements, threshold_angle
                    ):
                        point_group = r"$C_{" + str(nfold_max) + r"v}$"
                    else:
                        point_group = r"$C_{" + str(nfold_max) + r"}$"
        else:
            if any(["reflection" == sym_el["type"] for sym_el in symmetry_elements]):
                point_group = r"$C_s$"
            else:
                if has_inversion:
                    point_group = r"$C_i$"
    return point_group


def _check_c2_axis_normal_to_cn(main_rotation, symmetry_elements, threshold_angle):
    """Find 2-fold rotational axis normal to a higher rotational axis."""
    c2_perpendicular = 0
    c2_normal_axis = []
    for sym_el in symmetry_elements:
        if "rotation" == sym_el["type"] and sym_el["n-fold"] == 2:
            angle = calc_angle(main_rotation["axis"], sym_el["axis"])
            cond1 = abs(180.0 * angle / np.pi - 90.0) < threshold_angle
            cond2 = abs(180.0 * angle / np.pi - 270.0) < threshold_angle
            if cond1 or cond2:
                c2_normal_axis.append(sym_el)
                c2_perpendicular += 1
    return c2_perpendicular == main_rotation["n-fold"], c2_normal_axis


def _check_d_reflection_planes(main_rotation, symmetry_elements, c2_normal_axis, threshold_angle):
    """Check for d-reflection planes."""
    nr_d_planes = 0
    for sym_el in symmetry_elements:
        if sym_el["type"] == "reflection":
            angle = calc_angle(main_rotation["axis"], sym_el["n-vector"])
            cond1 = abs(180.0 * angle / np.pi - 90.0) < threshold_angle
            cond2 = abs(180.0 * angle / np.pi - 270.0) < threshold_angle
            if cond1 or cond2:
                # Check if reflection plane is bisecting two rotation axis:
                angle_c2 = [
                    calc_angle(sym_el_c2["axis"], sym_el["n-vector"])
                    for sym_el_c2 in c2_normal_axis
                ]
                angle_pairs = list(itertools.combinations(range(len(angle_c2)), 2))
                is_bisection = False
                for a_pair in angle_pairs:
                    cond1 = (
                        abs(180.0 * (angle_c2[a_pair[0]] - angle_c2[a_pair[0]]) / np.pi)
                        < threshold_angle
                    )
                    cond2 = (
                        abs(
                            abs(180.0 * (angle_c2[a_pair[0]] - angle_c2[a_pair[0]]) / np.pi)
                            - 180.0
                        )
                        < threshold_angle
                    )
                    cond3 = (
                        abs(
                            abs(180.0 * (angle_c2[a_pair[0]] - angle_c2[a_pair[0]]) / np.pi)
                            - 360.0
                        )
                        < threshold_angle
                    )
                    if cond1 or cond2 or cond3:
                        is_bisection = True
                        break
                if is_bisection:
                    nr_d_planes += 1
    return nr_d_planes == main_rotation["n-fold"]


def _check_h_reflection_planes(main_rotation, symmetry_elements, threshold_angle):
    """Check for horizontal relection planes."""
    has_h_plane = False
    for sym_el in symmetry_elements:
        if sym_el["type"] == "reflection":
            angle = calc_angle(main_rotation["axis"], sym_el["n-vector"])
            cond1 = abs(180.0 * angle / np.pi) < threshold_angle
            cond2 = abs(180.0 * angle / np.pi - 180.0) < threshold_angle
            cond3 = abs(180.0 * angle / np.pi - 360.0) < threshold_angle
            if cond1 or cond2 or cond3:
                has_h_plane = True
                break
    return has_h_plane


def _check_v_reflection_planes(main_rotation, symmetry_elements, threshold_angle):
    """Check for vertical relfection planes."""
    nr_v_planes = 0
    for sym_el in symmetry_elements:
        if sym_el["type"] == "reflection":
            angle = calc_angle(main_rotation["axis"], sym_el["n-vector"])
            cond1 = abs(180.0 * angle / np.pi - 90.0) < threshold_angle
            cond2 = abs(180.0 * angle / np.pi - 270.0) < threshold_angle
            if cond1 or cond2:
                nr_v_planes += 1
    return nr_v_planes == main_rotation["n-fold"]
