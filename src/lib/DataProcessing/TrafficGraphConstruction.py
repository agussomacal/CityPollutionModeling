"""Miscelaneous methods to handle Google traffic data

    We extensively use geopandas. A nice overview can be found in this FOSDEM talk by Joris Van den Bossche:
        https://www.youtube.com/watch?v=uqnA06fqhqk
"""

from collections import defaultdict
from typing import Dict, Tuple

import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd

from src.lib.DataProcessing.TrafficProcessing import TRAFFIC_VALUES
from PerplexityLab.miscellaneous import if_exist_load_else_do
from src.lib.FeatureExtractors.GraphFeatureExtractors import compute_adjacency


def load_transform_graph(filename):
    graph = ox.load_graphml(filename)
    for e in graph.edges:
        graph.edges[e]["lanes"] = float(graph.edges[e]["lanes"])
    return graph


@if_exist_load_else_do(file_format="gml", loader=load_transform_graph, saver=ox.save_graphml,
                       description=lambda graph: print(f"nodes: {graph.number_of_nodes()}\n"
                                                       f"edges: {graph.number_of_edges()}"))
def osm_graph(south, north, west, east):
    # list(map(lambda x: x[-1], nx.Graph(graph).to_undirected().edges)).count(1) # 61 multi edges
    # return nx.Graph(graph).to_undirected()
    graph = ox.graph_from_bbox(north=north, south=south, east=east, west=west, network_type="drive")

    road_type = {k: v if isinstance(v, str) else v[0] for k, v in nx.get_edge_attributes(graph, "highway").items()}
    road_lanes = {k: sum(map(int, l)) if isinstance(l, list) else int(l) for k, l in
                  nx.get_edge_attributes(graph, "lanes").items()}
    typical_lane = {way: np.mean([l for k, l in road_lanes.items() if k in road_type and road_type[k] == way]) for way
                    in set(road_type.values())}

    for e in graph.edges:
        if e in road_lanes:
            graph.edges[e]["lanes"] = float(road_lanes[e])
        elif e in road_type:
            graph.edges[e]["lanes"] = float(typical_lane[road_type[e]])
        else:
            graph.edges[e]["lanes"] = 1.0
    # nx.set_edge_attributes(graph, list(map(float, nx.get_edge_attributes(graph, "lanes").values())), "lanes")

    return graph


@if_exist_load_else_do(file_format="joblib",
                       description=lambda edges_pixels: print(f"edges with traffic: {len(edges_pixels)}"))
def project_pixels2edges(graph, traffic_pixels_coords):
    # Projet points into graph edges
    index_edges = ox.nearest_edges(graph, traffic_pixels_coords.loc["long", :],
                                   traffic_pixels_coords.loc["lat", :])
    edges_pixels = {(u, v): [] for u, v, _ in index_edges}
    for (u, v, _), pixel_coord in zip(index_edges, traffic_pixels_coords.columns):
        edges_pixels[(u, v)].append(pixel_coord)
    return edges_pixels


@if_exist_load_else_do(file_format="joblib")
def project_traffic_to_edges(traffic_by_pixel: pd.DataFrame, edges_pixels: Dict[Tuple, Tuple]):
    """
    traffic_by_pixel: columns: tuple of pixel coords; index: times; values: [0, 1, 2, 3, 4] for [no traffic info, green, yellow...]
    @:return traffic_by_edge: Dict[DataFrame] each key an edge, each value a dataframe with the amount of pixels with a certain
    color [columns] for each time [rows]
    """
    traffic_by_edge = {e: pd.DataFrame(0, index=traffic_by_pixel.index, columns=TRAFFIC_VALUES.keys()) for e in
                       edges_pixels.keys()}
    for e, pixels in edges_pixels.items():
        for pixel in pixels:
            for color, value in TRAFFIC_VALUES.items():
                traffic_by_edge[e].loc[:, color] += traffic_by_pixel.loc[:, pixel] == value

    return traffic_by_edge


@if_exist_load_else_do(file_format="joblib")
def get_traffic_by_node(traffic_by_edge, graph, nodes=None):
    """

    :param observed_pollution:
    :param traffic_by_edge:
    :param graph:
    :param nodes:
    :return: traffic_by_node: [#times, #nodes, #traffic colors]
    """
    nodes = list(graph.nodes) if nodes is None else nodes
    deg = compute_adjacency(graph, lambda data: data["length"] * data["lanes"]).toarray().sum(axis=1)
    node2ix = {n: i for i, n in enumerate(graph.nodes)}
    traffic_by_edge_normalization = {e: df.sum(axis=1).max() for e, df in traffic_by_edge.items()}
    # TODO: make it incremental instead of replacing the whole matrix.
    traffic_by_node = np.zeros((len(observed_pollution), len(nodes), len(TRAFFIC_VALUES)))
    for edge, df in traffic_by_edge.items():
        if (edge in graph.edges) and (edge[0] in nodes or edge[1] in nodes):
            for i, color in enumerate(TRAFFIC_VALUES):
                # length is added because we are doing the integral against the P1 elements.
                # a factor of 1/2 may be added too.
                update_val = df.loc[observed_pollution.index, color]
                update_val *= graph.edges[edge]["length"] * graph.edges[edge]["lanes"] / 2

                if edge[0] in nodes:
                    self.traffic_by_node[:, self.node2index[edge[0]], i] += \
                        update_val / deg[node2ix[edge[0]]] / traffic_by_edge_normalization[edge]

                if edge[1] in nodes:
                    self.traffic_by_node[:, self.node2index[edge[1]], i] += \
                        update_val / deg[node2ix[edge[1]]] / traffic_by_edge_normalization[edge]
    return self.traffic_by_node
