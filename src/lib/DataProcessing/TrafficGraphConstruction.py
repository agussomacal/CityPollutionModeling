"""Miscelaneous methods to handle Google traffic data

    We extensively use geopandas. A nice overview can be found in this FOSDEM talk by Joris Van den Bossche:
        https://www.youtube.com/watch?v=uqnA06fqhqk
"""

from collections import defaultdict
from typing import Dict, Tuple

import osmnx as ox
import pandas as pd

from src.lib.DataProcessing.TrafficProcessing import TRAFFIC_VALUES
from src.performance_utils import if_exist_load_else_do


@if_exist_load_else_do(file_format="gml", loader=ox.load_graphml, saver=ox.save_graphml)
def osm_graph(south, north, west, east):
    # list(map(lambda x: x[-1], nx.Graph(graph).to_undirected().edges)).count(1) # 61
    # return nx.Graph(graph).to_undirected()
    return ox.graph_from_bbox(north=north, south=south, east=east, west=west, network_type="drive")


@if_exist_load_else_do(file_format="joblib")
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
    """
    edges_traffic = {e: pd.DataFrame(0, index=traffic_by_pixel.index, columns=TRAFFIC_VALUES.keys()) for e in
                     edges_pixels.keys()}
    for e, pixels in edges_pixels.items():
        for pixel in pixels:
            for color, value in TRAFFIC_VALUES.items():
                edges_traffic[e].loc[:, color] += traffic_by_pixel.loc[:, pixel] == value

    return edges_traffic
