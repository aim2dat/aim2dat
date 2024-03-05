"""Module to find bonded molecular fragments in an extended structure."""

# Standard library imports
from typing import List

# Third party library imports
import numpy as np

# Internal library imports
from aim2dat.strct.strct import Structure
from aim2dat.strct.ext_analysis.decorator import external_analysis_method


@external_analysis_method
def determine_molecular_fragments(
    structure: Structure,
    exclude_elements: List[str] = [],
    end_point_elements: List[str] = [],
    r_max: float = 20.0,
    cn_method: str = "minimum_distance",
    min_dist_delta: float = 0.1,
    n_nearest_neighbours: int = 5,
    econ_tolerance: float = 0.5,
    econ_conv_threshold: float = 0.001,
    voronoi_weight_type: float = "rel_solid_angle",
    voronoi_weight_threshold: float = 0.5,
    okeeffe_weight_threshold: float = 0.5,
):
    """
    Find molecular fragments in a larger molecule/cluster of periodic crystal.

    Parameters
    ----------
    structure : aim2dat.strct.Structure
        Structure object.
    exclude_elements : list
        List of elements that are excluded from the search.
    end_point_elements : list
        List of elements that serve as an end point for a fragment.
    r_max : float (optional)
        Cut-off value for the maximum distance between two atoms in angstrom.
    cn_method : str (optional)
        Method used to calculate the coordination environment.
    min_dist_delta : float (optional)
        Tolerance parameter that defines the relative distance from the nearest neighbour atom
        for the ``'minimum_distance'`` method.
    n_nearest_neighbours : int (optional)
        Number of neighbours that are considered coordinated for the ``'n_neighbours'``
        method.
    econ_tolerance : float (optional)
        Tolerance parameter for the econ method.
    econ_conv_threshold : float (optional)
        Convergence threshold for the econ method.
    voronoi_weight_type : str (optional)
        Weight type of the Voronoi facets. Supported options are ``'covalent_atomic_radius'``,
        ``'area'`` and ``'solid_angle'``. The prefix ``'rel_'`` specifies that the relative
        weights with respect to the maximum value of the polyhedron are calculated.
    voronoi_weight_threshold : float (optional)
        Weight threshold to consider a neighbouring atom coordinated.

    Returns
    -------
    list
        List of fragments.
    """
    used_site_indices = []
    molecular_fragments = []
    coord = structure.calculate_coordination(
        r_max=r_max,
        method=cn_method,
        min_dist_delta=min_dist_delta,
        n_nearest_neighbours=n_nearest_neighbours,
        econ_tolerance=econ_tolerance,
        econ_conv_threshold=econ_conv_threshold,
        voronoi_weight_type=voronoi_weight_type,
        voronoi_weight_threshold=voronoi_weight_threshold,
        okeeffe_weight_threshold=okeeffe_weight_threshold,
    )

    for site_idx, site in enumerate(coord["sites"]):
        if site["element"] in end_point_elements:
            continue
        mol_fragment = {"elements": [], "site_indices": [], "positions": []}
        _recursive_graph_builder(
            site_idx,
            mol_fragment,
            exclude_elements,
            end_point_elements,
            used_site_indices,
            coord["sites"],
            np.zeros(3),
        )
        if mol_fragment["elements"] != []:
            molecular_fragments.append(mol_fragment)
    return None, molecular_fragments


def _recursive_graph_builder(
    site_idx,
    mol_fragment,
    exclude_elements,
    end_point_elements,
    used_site_indices,
    sites,
    position,
):
    """
    Find nearest neighbour of the first atom and adds it to the fragment. Take the neighbour
    of the neighbour and adds it to the fragment until all neighbours are found.
    """
    element = sites[site_idx]["element"]
    # Conditions to be added to the fragment:
    if (element not in exclude_elements) and (site_idx not in used_site_indices):
        # Steps:
        # 1) Update mol_fragment dictionary.
        mol_fragment["elements"].append(element)
        mol_fragment["site_indices"].append(site_idx)
        mol_fragment["positions"].append(position.tolist())
        shift = position - np.array(sites[site_idx]["position"])
        if element not in end_point_elements:
            # 2) Add site to _used_site_indices.
            used_site_indices.append(site_idx)
            # 3) Iterate through neighbours and start graph builder with new index.
            for neighbour in sites[site_idx]["neighbours"]:
                _recursive_graph_builder(
                    neighbour["site_index"],
                    mol_fragment,
                    exclude_elements,
                    end_point_elements,
                    used_site_indices,
                    sites,
                    np.array(neighbour["position"]) + shift,
                )
