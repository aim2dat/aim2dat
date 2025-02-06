"""Graphviz interface of the library."""

# Third party library imports
from graphviz import Digraph


def _networkx2graphviz(nx_graph, graphviz_engine, graphviz_edge_rank_colors):
    """Transform nx graph to graphviz graph."""
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
    return graphiz_graph
