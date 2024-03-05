"""Methods to create graphs from molecules and crystals."""

# Standard library imports
from typing import List

# Internal library imports
from aim2dat.ext_interfaces import _return_ext_interface_modules
from aim2dat.strct.strct import Structure
from aim2dat.strct.ext_analysis.decorator import external_analysis_method


@external_analysis_method
def create_graph(
    structure: Structure,
    graphviz_engine: str = "circo",
    graphviz_edge_rank_colors: List[str] = ["blue", "red", "green", "orange", "darkblue"],
    r_max: float = 20.0,
    cn_method: str = "minimum_distance",
    min_dist_delta: float = 0.1,
    n_nearest_neighbours: int = 5,
    econ_tolerance: float = 0.5,
    econ_conv_threshold: float = 0.001,
    voronoi_weight_type: float = "rel_solid_angle",
    voronoi_weight_threshold: float = 0.5,
):
    """
    Create graph based on the coordination.

    Parameters
    ----------
    structure : aim2dat.strct.Structure
        Structure object.
    graphviz_engine : str
        Graphviz engine used to create the graph. The default value is ``'circo'``.
    graphviz_edge_rank_colors : list
        List of colors of the different edge ranks.
    r_max : float
        Cut-off value for the maximum distance between two atoms in angstrom. The default
        value is set to ``20.0``.
    cn_method : str
        Method used to calculate the coordination environment. The default value is
        ``'minimum_distance'``.
    min_dist_delta : float
        Tolerance parameter that defines the relative distance from the nearest neighbour atom
        for the ``'minimum_distance'`` method. The default value is ``0.1``.
    n_nearest_neighbours : int
        Number of neighbours that are considered coordinated for the ``'n_neighbours'``
        method. The default value is ``5``.
    econ_tolerance : float
        Tolerance parameter for the econ method. The default value is ``0.5``.
    econ_conv_threshold : float
        Convergence threshold for the econ method. The default value is ``0.001``.
    voronoi_weight_type : str (optional)
        Weight type of the Voronoi facets. Supported options are ``'covalent_atomic_radius'``,
        ``'area'`` and ``'solid_angle'``. The prefix ``'rel_'`` specifies that the relative
        weights with respect to the maximum value of the polyhedron are calculated.
    voronoi_weight_threshold : float (optional)
        Weight threshold to consider a neighbouring atom coordinated.

    Returns
    -------
    nx_graph : nx.MultiDiGraph
        networkx graph of the structure.
    graphviz_graph : graphviz.Digraph
        graphviz graph of the structure.
    """
    return _return_ext_interface_modules("graphs")._graph_create(
        structure,
        graphviz_engine=graphviz_engine,
        graphviz_edge_rank_colors=graphviz_edge_rank_colors,
        r_max=r_max,
        cn_method=cn_method,
        min_dist_delta=min_dist_delta,
        n_nearest_neighbours=n_nearest_neighbours,
        econ_tolerance=econ_tolerance,
        econ_conv_threshold=econ_conv_threshold,
        voronoi_weight_type=voronoi_weight_type,
        voronoi_weight_threshold=voronoi_weight_threshold,
    )
