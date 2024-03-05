"""Interface to networkx and graphviz to create graphs based on the coordination."""

# Third party library imports
import networkx as nx
from graphviz import Digraph


def _graph_create(
    structure,
    graphviz_engine,
    graphviz_edge_rank_colors,
    r_max,
    cn_method,
    min_dist_delta,
    n_nearest_neighbours,
    econ_tolerance,
    econ_conv_threshold,
    voronoi_weight_type,
    voronoi_weight_threshold,
):
    """Create nx and graphviz graph."""
    coord = structure.calculate_coordination(
        r_max=r_max,
        method=cn_method,
        min_dist_delta=min_dist_delta,
        n_nearest_neighbours=n_nearest_neighbours,
        econ_tolerance=econ_tolerance,
        econ_conv_threshold=econ_conv_threshold,
        voronoi_weight_type=voronoi_weight_type,
        voronoi_weight_threshold=voronoi_weight_threshold,
    )
    coord_sites = coord["sites"]
    nx_graph = nx.MultiDiGraph()
    for site_idx, site in enumerate(coord_sites):
        nx_graph.add_node(site_idx, element=site["element"])
    for site_idx, site in enumerate(coord_sites):
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

    graphiz_graph = Digraph(engine=graphviz_engine)
    for node, attr in nx_graph.nodes(data=True):
        graphiz_graph.node(attr["element"] + str(node))
    for node_1, node_2, edge_idx in nx_graph.edges:
        el1 = nx_graph.nodes[node_1]["element"]
        el2 = nx_graph.nodes[node_2]["element"]
        rank = nx_graph[node_1][node_2][edge_idx]["rank"]
        rank_color = graphviz_edge_rank_colors[-1]
        if rank < len(graphviz_edge_rank_colors):
            rank_color = graphviz_edge_rank_colors[rank]
        graphiz_graph.edge(el1 + str(node_1), el2 + str(node_2), color=rank_color)
    return None, (nx_graph, graphiz_graph)
