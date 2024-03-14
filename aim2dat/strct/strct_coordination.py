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

    method_function = globals()["_coord_calculate_" + method]
    sites = []
    for site_idx in range(len(structure["elements"])):
        sites.append(method_function(site_idx, *method_args))
    stat_keys = ["distance"]
    is_optional = [False]
    if method == "voronoi" and voronoi_weight_type is not None or method == "econ":
        stat_keys.append("weight")
        is_optional.append(False)
    coordination = _calculate_statistical_quantities(structure, sites, stat_keys, is_optional)
    coordination["sites"] = sites
    return None, coordination


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


def _create_site_dict(structure, site_idx, neighbours, weights=None):
    """
    Create dictionary storing information for a specific site.
    """
    site_dict = {el0: 0 for el0 in structure._element_dict.keys()}
    site_dict["element"] = structure.elements[site_idx]
    site_dict["kind"] = None if structure.kinds is None else structure.kinds[site_idx]
    site_dict["position"] = list(structure.positions[site_idx])
    site_dict["neighbours"] = []

    distances = []
    for idx, (neigh_idx, neigh_pos, distance) in enumerate(neighbours):
        neigh_dict = {
            "element": structure.elements[neigh_idx],
            "kind": None if structure.kinds is None else structure.kinds[neigh_idx],
            "site_index": neigh_idx,
            "distance": float(distance),
            "position": [float(pos) for pos in neigh_pos],
        }
        if weights:
            neigh_dict["weight"] = float(weights[idx])
        site_dict["neighbours"].append(neigh_dict)
        site_dict[neigh_dict["element"]] += 1
        distances.append(float(distance))

    site_dict["total_cn"] = len(distances)
    site_dict["min_dist"] = min(distances, default=0.0)
    site_dict["max_dist"] = max(distances, default=0.0)
    site_dict["avg_dist"] = sum(distances) / len(distances) if len(distances) > 0 else 0.0
    return site_dict


def _calculate_statistical_quantities(structure, sites, stat_keys, is_optional):
    """
    Calculate statistical quantities of all sites such as average/minimum/maximum etc.
    """

    def calc_stdev(values):
        if len(values) == 1:
            return 0.0
        return stdev(values)

    temp_lists = {k0: {} for k0 in stat_keys + ["nrs"]}
    for el1, coord0 in zip(structure["elements"], sites):
        for el2 in structure._element_dict.keys():
            if el2 in coord0:
                temp_lists["nrs"].setdefault((el1, el2), []).append(coord0[el2])
        for neighbour in coord0["neighbours"]:
            for k0, optional in zip(stat_keys, is_optional):
                value = neighbour.get(k0, 0.0) if optional else neighbour[k0]
                temp_lists[k0].setdefault((el1, neighbour["element"]), []).append(value)

    stat_functions = [("_avg", mean), ("_stdev", calc_stdev), ("_max", max), ("_min", min)]
    stat_dict = {}
    for global_k in temp_lists.keys():
        for label, func in stat_functions:
            stat_dict[global_k + label] = {
                el_pair: func(values) for el_pair, values in temp_lists[global_k].items()
            }
    return stat_dict


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
    site_idx,
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
    el_idx_sc = indices_sc.index(site_idx)
    min_dist = np.amin(np.delete(dist_matrix[site_idx], el_idx_sc))
    position = np.array(structure.positions[site_idx])
    neighbours = []
    for idx in range(len(elements_sc)):
        distance = dist_matrix[site_idx][idx]
        if idx == el_idx_sc or distance > r_max:
            continue
        if distance >= min_dist and distance <= min_dist * (1.0 + distance_delta):
            neigh_pos = position + (
                np.array(positions_sc[idx]) - np.array(positions_sc[el_idx_sc])
            )
            neighbours.append((mapping[idx], neigh_pos, distance))
    return _create_site_dict(structure, site_idx, neighbours)


def _coord_calculate_n_nearest_neighbours(
    site_idx,
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
    el_idx_sc = indices_sc.index(site_idx)
    position = np.array(structure.positions[site_idx])
    neighbours = []
    zipped = list(zip(dist_matrix[site_idx].tolist(), range(len(elements_sc))))
    zipped.sort(key=lambda point: point[0])
    _, sc_indices = zip(*zipped)

    for idx in sc_indices[1 : n_neighbours + 1]:
        neigh_pos = position + (np.array(positions_sc[idx]) - np.array(positions_sc[el_idx_sc]))
        neighbours.append((mapping[idx], neigh_pos, dist_matrix[site_idx][idx]))
    return _create_site_dict(structure, site_idx, neighbours)


def _coord_calculate_econ(
    site_idx,
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

    el_idx_sc = indices_sc.index(site_idx)
    position = np.array(structure.positions[site_idx])
    distances_wo_site = np.delete(dist_matrix[site_idx], el_idx_sc)
    min_dist = np.amin(distances_wo_site)
    dist_avg0 = min_dist
    dist_avg = calc_weighted_average_bond_length(distances_wo_site, dist_avg0)
    while abs(dist_avg - dist_avg0) >= conv_threshold:
        dist_avg0 = dist_avg
        dist_avg = calc_weighted_average_bond_length(distances_wo_site, dist_avg0)

    neighbours = []
    weights = []
    for idx in range(len(elements_sc)):
        if idx == el_idx_sc or dist_matrix[site_idx][idx] > r_max:
            continue
        weight = math.exp(1.0 - (dist_matrix[site_idx][idx] / dist_avg0) ** 6.0)
        if weight >= tolerance:
            neigh_pos = position + (
                np.array(positions_sc[idx]) - np.array(positions_sc[el_idx_sc])
            )
            neighbours.append((mapping[idx], neigh_pos, dist_matrix[site_idx][idx]))
            weights.append(weight)
    return _create_site_dict(structure, site_idx, neighbours, weights)


def _coord_calculate_voronoi(
    site_idx,
    structure,
    voronoi_list,
    weight_type,
    weight_threshold,
):
    """
    Calculate coordination numbers using o'Keeffe method.
    """
    neighbours = voronoi_list[site_idx]
    max_weight = 0.0
    for neigh in neighbours:
        neigh["distance"] = float(
            np.linalg.norm(neigh["position"] - structure["positions"][site_idx])
        )
        if weight_type is None:
            pass
        elif "covalent_atomic_radius" in weight_type:
            neigh["weight"] = (
                get_atomic_radius(structure["elements"][site_idx], radius_type="covalent")
                + get_atomic_radius(structure["elements"][neigh["index"]], radius_type="covalent")
                - neigh["distance"]
            )
        elif "vdw_atomic_radius" in weight_type:
            neigh["weight"] = (
                get_atomic_radius(structure["elements"][site_idx], radius_type="vdw")
                + get_atomic_radius(structure["elements"][neigh["index"]], radius_type="vdw")
                - neigh["distance"]
            )
        elif "solid_angle" in weight_type:
            neigh["weight"] = calc_solid_angle(structure["positions"][site_idx], neigh["vertices"])
        elif "area" in weight_type:
            neigh["weight"] = calc_polygon_area(neigh["vertices"])
        else:
            raise ValueError(f"`weight_type` '{weight_type}' is not supported.")
        if neigh.get("weight", -1.0) > max_weight:
            max_weight = neigh["weight"]

    neighbours = []
    weights = []
    for neigh in voronoi_list[site_idx]:
        add_site = True
        if weight_type is not None:
            if "rel" in weight_type:
                neigh["weight"] /= max_weight
            if neigh["weight"] <= weight_threshold:
                add_site = False
        if add_site:
            neighbours.append((neigh["index"], neigh["position"], neigh["distance"]))
            if weight_type is not None:
                weights.append(neigh["weight"])
    return _create_site_dict(structure, site_idx, neighbours, weights)
