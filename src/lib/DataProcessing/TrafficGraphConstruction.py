from collections import defaultdict
from typing import Dict, Tuple

import osmnx as ox
import pandas as pd

from src.lib.DataProcessing.TrafficProcessing import TRAFFIC_VALUES
from src.performance_utils import timeit, if_exist_load_else_do


@if_exist_load_else_do
def project_pixels2edges(G, traffic_pixels_coords):
    # Projet points into graph edges
    with timeit("Projecting pixels coords to edges of graph"):
        index_edges = ox.distance.get_nearest_edges(G, traffic_pixels_coords.loc["long", :],
                                                    traffic_pixels_coords.loc["lat", :],
                                                    method=None)
        edges_pixels = defaultdict(list)
        for (geom, u, v), pixel_coord in zip(index_edges, traffic_pixels_coords.columns):
            edges_pixels[(u, v)].append(pixel_coord)
    return edges_pixels


@if_exist_load_else_do
def project_traffic_to_edges(G, traffic_by_pixel: pd.DataFrame, edges_pixels: Dict[Tuple, Tuple]):
    """
    traffic_by_pixel: columns: tuple of pixel coords; index: times; values: [0, 1, 2, 3, 4] for [no traffic info, green, yellow...]
    """
    traffic = defaultdict(lambda: pd.DataFrame(0, index=traffic_by_pixel.index, columns=TRAFFIC_VALUES.keys()))
    for e in G.edges:
        for pixel in edges_pixels[(e[1], e[2])]:
            for color, value in TRAFFIC_VALUES.items():
                traffic[(e[0], e[1], e[2])].loc[:, color] += traffic_by_pixel.loc[:, pixel] == value

    return traffic
