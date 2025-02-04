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
    max_fragment_size: int = 100,
    exclude_elements: List[str] = None,
    exclude_sites: List[int] = None,
    end_point_elements: List[str] = None,
    r_max: float = 10.0,
    cn_method: str = "minimum_distance",
    min_dist_delta: float = 0.1,
    n_nearest_neighbours: int = 5,
    radius_type: str = "chen_manz",
    atomic_radius_delta: float = 0.0,
    econ_tolerance: float = 0.5,
    econ_conv_threshold: float = 0.001,
    voronoi_weight_type: float = "rel_solid_angle",
    voronoi_weight_threshold: float = 0.5,
    okeeffe_weight_threshold: float = 0.5,
) -> List[Structure]:
    """
    Find molecular fragments in a larger molecule/cluster of periodic crystal.

    Parameters
    ----------
    structure : aim2dat.strct.Structure
        Structure object.
    max_fragment_size : int
        Maximum size of the fragments to avoid recursion overflow.
    exclude_elements : list
        List of elements that are excluded from the search.
    exclude_sites : list
        List of site indices that are excluded from the search.
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
    radius_type : str (optional)
        Type of the atomic radius used for the ``'atomic_radius'`` method (``'covalent'`` is
        used as fallback in the radius for an element is not defined).
    atomic_radius_delta : float (optional)
        Tolerance relative to the sum of the atomic radii for the ``'atomic_radius'`` method.
        If set to ``0.0`` the maximum threshold is defined by the sum of the atomic radii,
        positive (negative) values increase (decrease) the threshold.
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
    if exclude_elements is None:
        exclude_elements = []
    if exclude_sites is None:
        exclude_sites = []
    if end_point_elements is None:
        end_point_elements = []
    used_site_indices = []
    molecular_fragments = []
    coord = structure.calculate_coordination(
        r_max=r_max,
        method=cn_method,
        min_dist_delta=min_dist_delta,
        n_nearest_neighbours=n_nearest_neighbours,
        radius_type=radius_type,
        atomic_radius_delta=atomic_radius_delta,
        econ_tolerance=econ_tolerance,
        econ_conv_threshold=econ_conv_threshold,
        voronoi_weight_type=voronoi_weight_type,
        voronoi_weight_threshold=voronoi_weight_threshold,
        okeeffe_weight_threshold=okeeffe_weight_threshold,
    )

    allowed_sites = []
    for site_idx, el in enumerate(structure.elements):
        if site_idx not in exclude_sites and el not in exclude_elements:
            allowed_sites.append(site_idx)

    for site_idx in allowed_sites:
        if (
            coord["sites"][site_idx]["element"] in end_point_elements
            or site_idx in used_site_indices
        ):
            continue
        mol_fragment = {
            "elements": [],
            "site_attributes": {"parent_indices": []},
            "positions": [],
            "pbc": False,
        }
        _recursive_graph_builder(
            site_idx,
            mol_fragment,
            max_fragment_size,
            allowed_sites,
            end_point_elements,
            coord["sites"],
            np.array(structure.positions[site_idx]),
        )
        if mol_fragment["elements"] != []:
            molecular_fragments.append(Structure(**mol_fragment))
            used_site_indices += mol_fragment["site_attributes"]["parent_indices"]
    return None, molecular_fragments


def _recursive_graph_builder(
    site_idx,
    mol_fragment,
    max_fragment_size,
    allowed_sites,
    end_point_elements,
    sites,
    position,
):
    """
    Find nearest neighbour of the first atom and adds it to the fragment. Take the neighbour
    of the neighbour and adds it to the fragment until all neighbours are found.
    """
    # Conditions to be added to the fragment:
    if len(mol_fragment["elements"]) >= max_fragment_size:
        return None

    if site_idx in allowed_sites:
        # Steps:
        # 1) Check if site with same position is already in mol_fragment dictionary.
        for site_idx2, position2 in zip(
            mol_fragment["site_attributes"]["parent_indices"], mol_fragment["positions"]
        ):
            if site_idx == site_idx2 and np.linalg.norm(position - position2) < 1e-3:
                return None

        # 2) Update mol_fragment dictionary.
        element = sites[site_idx]["element"]
        mol_fragment["elements"].append(element)
        mol_fragment["site_attributes"]["parent_indices"].append(site_idx)
        mol_fragment["positions"].append(position.tolist())
        shift = position - np.array(sites[site_idx]["position"])
        if element not in end_point_elements:
            # 3) Iterate through neighbours and start graph builder with new index.
            for neighbour in sites[site_idx]["neighbours"]:
                _recursive_graph_builder(
                    neighbour["site_index"],
                    mol_fragment,
                    max_fragment_size,
                    allowed_sites,
                    end_point_elements,
                    sites,
                    np.array(neighbour["position"]) + shift,
                )
