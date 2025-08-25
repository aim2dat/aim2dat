"""Methods to create graphs from molecules and crystals."""

# Standard library imports
from typing import List

# Third party library imports
import networkx as nx

# Internal library imports
from aim2dat.ext_interfaces import _return_ext_interface_modules
from aim2dat.strct.structure import Structure
from aim2dat.strct.ext_analysis.decorator import external_analysis_method


@external_analysis_method(attr_mapping=None)
def calc_graph(
    structure: Structure,
    get_graphviz_graph: bool = False,
    graphviz_engine: str = "circo",
    graphviz_edge_rank_colors: List[str] = ["blue", "red", "green", "orange", "darkblue"],
    **cn_kwargs,
):
    """
    Create graph based on the coordination.

    Parameters
    ----------
    structure : aim2dat.strct.Structure
        Structure object.
    get_graphviz_graph : bool
        Whether to also output a graphviz.Digraph object.
    graphviz_engine : str
        Graphviz engine used to create the graph. The default value is ``'circo'``.
    graphviz_edge_rank_colors : list
        List of colors of the different edge ranks.
    cn_kwargs :
        Optional keyword arguments passed on to the ``calculate_coordination`` function.

    Returns
    -------
    nx_graph : nx.MultiDiGraph
        networkx graph of the structure.
    graphviz_graph : graphviz.Digraph
        graphviz graph of the structure (if ``get_graphviz_graph`` is set to ``True``).
    """
    coord = structure.calc_coordination(**cn_kwargs)
    nx_graph = nx.MultiDiGraph()
    for site_idx, site in enumerate(coord["sites"]):
        nx_graph.add_node(site_idx, element=site["element"])
    for site_idx, site in enumerate(coord["sites"]):
        if len(site["neighbours"]) == 0:
            continue
        distances = [neigh["distance"] for neigh in site["neighbours"]]
        zipped = list(zip(distances, range(len(site["neighbours"]))))
        zipped.sort(key=lambda point: point[0])
        _, neigh_indices = zip(*zipped)
        last_dist = 0.0
        last_dist_idx = 0
        for dist_idx, neigh_idx in enumerate(neigh_indices):
            if abs(last_dist - distances[neigh_idx]) < 1e-5:
                dist_idx = last_dist_idx
            nx_graph.add_edge(site_idx, site["neighbours"][neigh_idx]["site_index"], rank=dist_idx)
            last_dist_idx = dist_idx
            last_dist = distances[neigh_idx]

    if get_graphviz_graph:
        backend_module = _return_ext_interface_modules("graphviz")
        return nx_graph, backend_module._networkx2graphviz(
            nx_graph, graphviz_engine, graphviz_edge_rank_colors
        )
    else:
        return nx_graph
