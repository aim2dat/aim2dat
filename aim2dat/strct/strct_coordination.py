"""Module to calculate the coordination number of atomic sites."""

# Standard library imports
import math
from statistics import mean, stdev

# Third party libraries
from scipy.spatial.distance import cdist
import numpy as np

# Internal library imports
from aim2dat.strct.strct_super_cell import _create_supercell_positions
from aim2dat.utils.element_properties import get_atomic_radius
from aim2dat.utils.maths import calc_solid_angle, calc_polygon_area

_supported_methods = ["minimum_distance", "n_nearest_neighbours", "econ", "voronoi"]


def _coord_group_method_args(
    min_dist_delta,
    n_nearest_neighbours,
    econ_tolerance,
    econ_conv_threshold,
    voronoi_weight_type,
    voronoi_weight_threshold,
):
    return {
        "minimum_distance": [min_dist_delta],
        "n_nearest_neighbours": [n_nearest_neighbours],
        "econ": [econ_tolerance, econ_conv_threshold],
        "voronoi": [voronoi_weight_type, voronoi_weight_threshold],
    }


def calculate_coordination(
    structure,
    r_max: float,
    method: str,
    min_dist_delta: float,
    n_nearest_neighbours: int,
    econ_tolerance: float,
    econ_conv_threshold: float,
    voronoi_weight_type: str,
    voronoi_weight_threshold: float,
    okeeffe_weight_threshold: float,
):
    """
    Calculate the coordination of all sites.
    """

    def calc_stdev(values):
        if len(values) == 1:
            return 0.0
        else:
            return stdev(values)

    if method == "okeeffe":
        from warnings import warn

        warn(
            "Method 'okeeffe' with `okeeffe_weight_threshold` is depreciated, "
            + "please use 'voronoi' with `voronoi_weight_type`: 'rel_solid_angle' "
            + "and `voronoi_weight_threshold` instead.",
            DeprecationWarning,
            2,
        )
        method = "voronoi"
        voronoi_weight_type = "rel_solid_angle"
        voronoi_weight_threshold = okeeffe_weight_threshold

    if method not in _supported_methods:
        raise ValueError(
            f"Method '{method}' is not supported. Supported methods are: '"
            + "', '".join(_supported_methods)
            + "'."
        )

    method_args = _coord_group_method_args(
        min_dist_delta,
        n_nearest_neighbours,
        econ_tolerance,
        econ_conv_threshold,
        voronoi_weight_type,
        voronoi_weight_threshold,
    )

    method_function = globals()["_coord_calculate_" + method]
    coord_numbers = []
    if method == "voronoi":
        voronoi_list = structure.calculate_voronoi_tessellation(r_max=r_max)
        method_args = [structure, voronoi_list] + method_args[method]
    else:
        elements_sc, kinds_sc, positions_sc, indices_sc, mapping, _ = _create_supercell_positions(
            structure, r_max
        )
        dist_matrix = cdist(structure.get_positions(cartesian=True, wrap=True), positions_sc)
        method_args = [
            structure,
            r_max,
            elements_sc,
            positions_sc,
            indices_sc,
            mapping,
            dist_matrix,
        ] + method_args[method]

    for el_idx in range(len(structure["elements"])):
        coord_numbers.append(method_function(el_idx, *method_args))
    numbers = {}
    distances = {}
    weights = {}
    for el1, coord0 in zip(structure["elements"], coord_numbers):
        for el2 in structure._element_dict.keys():
            if el2 in coord0:
                if (el1, el2) in numbers:
                    numbers[(el1, el2)].append(coord0[el2])
                else:
                    numbers[(el1, el2)] = [coord0[el2]]
        for neighbour in coord0["neighbours"]:
            if (el1, neighbour["element"]) in distances:
                distances[(el1, neighbour["element"])].append(neighbour["distance"])
                weights[(el1, neighbour["element"])].append(neighbour.get("weight", 0.0))
            else:
                distances[(el1, neighbour["element"])] = [neighbour["distance"]]
                weights[(el1, neighbour["element"])] = [neighbour.get("weight", 0.0)]
    coordination = {
        "sites": coord_numbers,
        "nrs_avg": {el_pair: mean(values) for el_pair, values in numbers.items()},
        "nrs_stdev": {el_pair: calc_stdev(values) for el_pair, values in numbers.items()},
        "nrs_max": {el_pair: max(values) for el_pair, values in numbers.items()},
        "nrs_min": {el_pair: min(values) for el_pair, values in numbers.items()},
        "distance_avg": {el_pair: mean(values) for el_pair, values in distances.items()},
        "distance_stdev": {el_pair: calc_stdev(values) for el_pair, values in distances.items()},
        "distance_max": {el_pair: max(values) for el_pair, values in distances.items()},
        "distance_min": {el_pair: min(values) for el_pair, values in distances.items()},
    }
    if method == "voronoi" and voronoi_weight_type is not None or method == "econ":
        coordination["weight_avg"] = {el_pair: mean(values) for el_pair, values in weights.items()}
        coordination["weight_stdev"] = {
            el_pair: calc_stdev(values) for el_pair, values in weights.items()
        }
        coordination["weight_max"] = {el_pair: max(values) for el_pair, values in weights.items()}
        coordination["weight_min"] = {el_pair: min(values) for el_pair, values in weights.items()}
    return None, coordination


def _coordination_compare_sites(
    structures, site_indices, calc_properties, distinguish_kinds, threshold
):
    # TODO: consider angles, too.
    site_infos = [calc_properties[idx]["sites"][site_indices[idx]] for idx in range(2)]
    if len(site_infos[0]["neighbours"]) != len(site_infos[1]["neighbours"]):
        return False

    comp_type = "element"
    if distinguish_kinds:
        comp_type = "kind"
    equal_list = []
    for neigh1 in site_infos[0]["neighbours"]:
        for neigh2_idx, neigh2 in enumerate(site_infos[1]["neighbours"]):
            if (
                neigh2_idx not in equal_list
                and neigh1[comp_type] == neigh2[comp_type]
                and abs(neigh1["distance"] - neigh2["distance"]) < threshold
            ):
                equal_list.append(neigh2_idx)
                break
    if len(equal_list) != len(site_infos[0]["neighbours"]):
        return False
    else:
        return True


def _coord_calculate_minimum_distance(
    el_idx,
    structure,
    r_max,
    elements_sc,
    positions_sc,
    indices_sc,
    mapping,
    dist_matrix,
    distance_delta,
):
    """
    Calculate coordination numbers using the minimum distance method.
    """
    el_idx_sc = indices_sc.index(el_idx)
    min_dist = np.amin(np.delete(dist_matrix[el_idx], el_idx_sc))
    position = np.array(structure.positions[el_idx])
    site_details = []
    coord_dict = {el0: 0 for el0 in structure._element_dict.keys()}
    coord_dict["element"] = structure["elements"][el_idx]
    coord_dict["kind"] = None if structure["kinds"] is None else structure["kinds"][el_idx]
    coord_dict["position"] = list(structure.positions[el_idx])
    coord_dict["total_cn"] = 0
    coord_dict["min_dist"] = float(min_dist)
    coord_dict["max_dist"] = 0.0
    coord_dict["avg_dist"] = 0.0
    distances = []
    for idx in range(len(elements_sc)):
        distance = dist_matrix[el_idx][idx]
        if idx == el_idx_sc or distance > r_max:
            continue
        if distance >= min_dist and distance <= min_dist * (1.0 + distance_delta):
            neigh_pos = position + (
                np.array(positions_sc[idx]) - np.array(positions_sc[el_idx_sc])
            )
            coord_dict[elements_sc[idx]] += 1
            coord_dict["total_cn"] += 1

            site_details.append(
                {
                    "element": elements_sc[idx],
                    "kind": (
                        None if structure["kinds"] is None else structure["kinds"][mapping[idx]]
                    ),
                    "site_index": mapping[idx],
                    "distance": float(distance),
                    "position": [float(pos) for pos in neigh_pos],
                }
            )
            distances.append(float(distance))
            if distance > coord_dict["max_dist"]:
                coord_dict["max_dist"] = float(distance)
    if len(distances) > 0:
        coord_dict["avg_dist"] = sum(distances) / len(distances)
    coord_dict["neighbours"] = site_details
    return coord_dict


def _coord_calculate_n_nearest_neighbours(
    el_idx,
    structure,
    r_max,
    elements_sc,
    positions_sc,
    indices_sc,
    mapping,
    dist_matrix,
    n_neighbours,
):
    """
    Calculate the coordination numbers by taking the n nearest atoms.
    """
    el_idx_sc = indices_sc.index(el_idx)
    position = np.array(structure.positions[el_idx])
    site_details = []
    coord_dict = {el0: 0 for el0 in structure._element_dict.keys()}
    coord_dict["element"] = structure["elements"][el_idx]
    coord_dict["kind"] = None if structure["kinds"] is None else structure["kinds"][el_idx]
    coord_dict["position"] = list(structure.positions[el_idx])
    coord_dict["total_cn"] = 0
    coord_dict["min_dist"] = float(np.amin(np.delete(dist_matrix[el_idx], el_idx_sc)))
    coord_dict["max_dist"] = 0.0
    coord_dict["avg_dist"] = 0.0
    distances = []
    zipped = list(zip(dist_matrix[el_idx].tolist(), range(len(elements_sc))))
    zipped.sort(key=lambda point: point[0])
    _, sc_indices = zip(*zipped)

    for idx in sc_indices[1 : n_neighbours + 1]:
        distance = dist_matrix[el_idx][idx]
        neigh_pos = position + (np.array(positions_sc[idx]) - np.array(positions_sc[el_idx_sc]))
        coord_dict[elements_sc[idx]] += 1
        coord_dict["total_cn"] += 1
        site_details.append(
            {
                "element": elements_sc[idx],
                "kind": None if structure["kinds"] is None else structure["kinds"][mapping[idx]],
                "site_index": mapping[idx],
                "distance": float(distance),
                "position": [float(pos) for pos in neigh_pos],
            }
        )
        distances.append(float(distance))
        if distance > coord_dict["max_dist"]:
            coord_dict["max_dist"] = float(distance)
    if len(distances) > 0:
        coord_dict["avg_dist"] = sum(distances) / len(distances)
    coord_dict["neighbours"] = site_details
    return coord_dict


def _coord_calculate_econ(
    el_idx,
    structure,
    r_max,
    elements_sc,
    positions_sc,
    indices_sc,
    mapping,
    dist_matrix,
    tolerance,
    conv_threshold,
):
    """
    Calculate coordination numbers using the econ method.
    """

    def calc_weighted_average_bond_length(distances, dist_avg0):
        exp_dist = [math.exp(1.0 - (dist0 / dist_avg0) ** 6.0) for dist0 in distances]
        numerator = sum([dist0 * exp_dist0 for dist0, exp_dist0 in zip(distances, exp_dist)])
        dist_avg = numerator / sum(exp_dist)
        return dist_avg

    el_idx_sc = indices_sc.index(el_idx)
    position = np.array(structure.positions[el_idx])
    distances_wo_site = np.delete(dist_matrix[el_idx], el_idx_sc)
    min_dist = np.amin(distances_wo_site)
    site_details = []
    coord_dict = {el0: 0 for el0 in structure._element_dict.keys()}
    coord_dict["element"] = structure["elements"][el_idx]
    coord_dict["kind"] = None if structure["kinds"] is None else structure["kinds"][el_idx]
    coord_dict["position"] = list(structure.positions[el_idx])
    coord_dict["total_cn"] = 0
    coord_dict["min_dist"] = float(min_dist)
    coord_dict["max_dist"] = 0.0
    coord_dict["avg_dist"] = 0.0

    distances = []
    dist_avg0 = coord_dict["min_dist"]
    dist_avg = calc_weighted_average_bond_length(distances_wo_site, dist_avg0)
    while abs(dist_avg - dist_avg0) >= conv_threshold:
        dist_avg0 = dist_avg
        dist_avg = calc_weighted_average_bond_length(distances_wo_site, dist_avg0)

    for idx in range(len(elements_sc)):
        if idx == el_idx_sc or dist_matrix[el_idx][idx] > r_max:
            continue
        weight = math.exp(1.0 - (dist_matrix[el_idx][idx] / dist_avg0) ** 6.0)
        if weight >= tolerance:
            coord_dict[elements_sc[idx]] += 1
            coord_dict["total_cn"] += 1
            neigh_pos = position + (
                np.array(positions_sc[idx]) - np.array(positions_sc[el_idx_sc])
            )
            site_details.append(
                {
                    "element": elements_sc[idx],
                    "kind": (
                        None if structure["kinds"] is None else structure["kinds"][mapping[idx]]
                    ),
                    "site_index": mapping[idx],
                    "distance": float(dist_matrix[el_idx][idx]),
                    "position": [float(pos) for pos in neigh_pos],
                    "weight": weight,
                }
            )
            distances.append(float(dist_matrix[el_idx][idx]))
            if dist_matrix[el_idx][idx] > coord_dict["max_dist"]:
                coord_dict["max_dist"] = float(dist_matrix[el_idx][idx])
    if len(distances) > 0:
        coord_dict["avg_dist"] = sum(distances) / len(distances)
    coord_dict["neighbours"] = site_details
    return coord_dict


def _coord_calculate_voronoi(
    el_idx,
    structure,
    voronoi_list,
    weight_type,
    weight_threshold,
):
    """
    Calculate coordination numbers using o'Keeffe method.
    """
    neighbours = voronoi_list[el_idx]
    max_weight = 0.0
    for neigh in neighbours:
        neigh["distance"] = float(
            np.linalg.norm(neigh["position"] - structure["positions"][el_idx])
        )
        if weight_type is None:
            pass
        elif "covalent_atomic_radius" in weight_type:
            neigh["weight"] = (
                get_atomic_radius(structure["elements"][el_idx], radius_type="covalent")
                + get_atomic_radius(structure["elements"][neigh["index"]], radius_type="covalent")
                - neigh["distance"]
            )
        elif "vdw_atomic_radius" in weight_type:
            neigh["weight"] = (
                get_atomic_radius(structure["elements"][el_idx], radius_type="vdw")
                + get_atomic_radius(structure["elements"][neigh["index"]], radius_type="vdw")
                - neigh["distance"]
            )
        elif "solid_angle" in weight_type:
            neigh["weight"] = calc_solid_angle(structure["positions"][el_idx], neigh["vertices"])
        elif "area" in weight_type:
            neigh["weight"] = calc_polygon_area(neigh["vertices"])
        else:
            raise ValueError(f"`weight_type` '{weight_type}' is not supported.")
        if neigh.get("weight", -1.0) > max_weight:
            max_weight = neigh["weight"]

    coord_dict = {el0: 0 for el0 in structure._element_dict.keys()}
    coord_dict = {el0: 0 for el0 in structure._element_dict.keys()}
    coord_dict["element"] = structure["elements"][el_idx]
    coord_dict["kind"] = None if structure["kinds"] is None else structure["kinds"][el_idx]
    coord_dict["position"] = list(structure.positions[el_idx])
    coord_dict["total_cn"] = 0
    coord_dict["min_dist"] = neighbours[0]["distance"]
    coord_dict["max_dist"] = 0.0
    coord_dict["avg_dist"] = 0.0

    site_details = []
    distances = []
    for neigh in voronoi_list[el_idx]:
        add_site = True
        if weight_type is not None:
            if "rel" in weight_type:
                neigh["weight"] /= max_weight
            if neigh["weight"] <= weight_threshold:
                add_site = False
        if add_site:
            site = {
                "element": structure["elements"][neigh["index"]],
                "kind": None if structure["kinds"] is None else structure["kinds"][neigh["index"]],
                "site_index": neigh["index"],
                "distance": neigh["distance"],
                "position": [float(pos) for pos in neigh["position"]],
            }
            if weight_type is not None:
                site["weight"] = float(neigh["weight"])
            site_details.append(site)
            distances.append(neigh["distance"])
            if coord_dict["max_dist"] < neigh["distance"]:
                coord_dict["max_dist"] = neigh["distance"]
            if coord_dict["min_dist"] > neigh["distance"]:
                coord_dict["min_dist"] = neigh["distance"]
            coord_dict[site["element"]] += 1
            coord_dict["total_cn"] += 1
    if len(distances) > 0:
        coord_dict["avg_dist"] = sum(distances) / len(distances)
    coord_dict["neighbours"] = site_details
    return coord_dict
