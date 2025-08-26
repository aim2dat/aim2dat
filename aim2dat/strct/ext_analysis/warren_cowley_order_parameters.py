"""Functions to calculate the Warren-Cowley like order parameters."""

# Standard library imports
from statistics import mean


# Internal library imports
from aim2dat.strct.ext_analysis.decorator import external_analysis_method
from aim2dat.strct.structure import Structure
from aim2dat.utils.maths import calc_polygon_area


@external_analysis_method(attr_mapping=None)
def calc_warren_cowley_order_p(structure: Structure, r_max: float = 20.0, max_shells: int = 3):
    """
    Calculate Warren-Cowley like order parameters as defined in :doi:`10.1103/PhysRevB.96.024104`.

    Parameters
    -----------
    structure : aim2dat.strct.Structure
        Structure object.
    r_max : float (optional)
        Cut-off value for the maximum distance between two atoms in angstrom.
    max_shells : int
        Number of neighbour shells that are evaluated.

    Returns
    --------
    dict
        Dictionary containing the order parameters.
    """
    # TODO calculate in voronoi tessellation? and implement different weights
    voronoi_list = structure.calc_voronoi_tessellation(r_max)
    for vlist_site in voronoi_list:
        for vinfo in vlist_site:
            vinfo["weight"] = calc_polygon_area(vinfo["vertices"])
    kinds = structure["elements"]
    kind_dict = structure._element_dict
    at_fracts = {kind: len(val) / len(kinds) for kind, val in kind_dict.items()}
    order_p_list = []

    if len(kind_dict) == 1:
        return {
            "order_p": {kinds[0]: [0.0 for _ in range(max_shells)]},
            "order_p_sites": [[0.0 for _ in range(max_shells)] for _ in kinds],
        }

    for site_idx, kind in enumerate(kinds):
        total_weights = [0.0 for idx in range(max_shells)]
        order_p = [0.0 for idx in range(max_shells)]
        _calculate_order_p_recursive(
            (site_idx, (0, 0, 0)),
            0,
            1.0,
            [],
            order_p,
            total_weights,
            kinds,
            max_shells,
            voronoi_list,
        )
        order_p_list.append(
            [float(1.0 - val / at_fracts[kind] / t_w) for val, t_w in zip(order_p, total_weights)]
        )
    order_p_dict = {}
    for kind, kind_sites in kind_dict.items():
        order_p_dict[kind] = [
            mean(order_p_list[idx][shell] for idx in kind_sites) for shell in range(max_shells)
        ]
    return {"order_p": order_p_dict, "order_p_sites": order_p_list}


def _calculate_order_p_recursive(
    current_rep, shell, cumul_w, rep_list, order_p, total_weights, kinds, max_shells, voronoi_list
):
    if shell < max_shells:
        orig_kind = kinds[current_rep[0]] if len(rep_list) == 0 else kinds[rep_list[0][0]]
        neighbours = voronoi_list[current_rep[0]]
        rep_list.append(current_rep)
        for neigh_info in neighbours:
            neigh_rep = (
                neigh_info["index"],
                tuple(val0 + val1 for val0, val1 in zip(current_rep[1], neigh_info["replica"])),
            )
            if any(neigh_rep == rep0 for rep0 in rep_list):
                continue
            weight = neigh_info["weight"] * cumul_w
            total_weights[shell] += weight
            if kinds[neigh_rep[0]] == orig_kind:
                order_p[shell] += weight
            _calculate_order_p_recursive(
                neigh_rep,
                shell + 1,
                weight,
                rep_list.copy(),
                order_p,
                total_weights,
                kinds,
                max_shells,
                voronoi_list,
            )
