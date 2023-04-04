import os

import numpy as np
import osmnx as ox
import pandas as pd

from src.config import traffic_dir
from src.performance_utils import timeit


def project_pixels2edges(G, traffic_pixels_coords, recalculate=False):
    filepath = f"{traffic_dir}/pixel2edge.npy"
    if recalculate or not os.path.exists(filepath):
        # Projet points into graph edges
        with timeit("Projecting pixels coords to edges of graph"):
            index_edges = ox.distance.get_nearest_edges(G, traffic_pixels_coords.loc["long", :],
                                                        traffic_pixels_coords.loc["lat", :],
                                                        method=None)
            np.save(filepath, index_edges)
    else:
        with timeit("Time loading pre-processed data:"):
            index_edges = np.load(filepath)
    return index_edges


def project_traffic_to_edges(G, traffic_by_pixel: pd.DataFrame, traffic_pixels_coords, index_edges):
    """
    traffic_by_pixel: columns: tuple of pixel coords; index: times; values: [0, 1, 2, 3, 4] for [no traffic info, green, yellow...]
    traffic_pixels_coords: columns: tuple of pixel coords; index: [lat, long]; values: lat and long associate with the pixel
    """

    # Read image traffic data
    date = t.strftime("%Y-%m-%d-%H-%M-%S")
    fn = {}
    fn['green'] = str(traffic_dir) + '/' + date + '_green.png'
    fn['orange'] = str(traffic_dir) + '/' + date + '_orange.png'
    fn['red'] = str(traffic_dir) + '/' + date + '_red.png'
    fn['darkred'] = str(traffic_dir) + '/' + date + '_darkred.png'

    data = {}
    for key in fn.keys():
        data[key] = np.sign(np.array(Image.open(fn[key])))

    # Lat-Long of each pixel image
    LAT, LONG = img_to_latlong()

    # Dictionary traffic[edge]=mean_traffic_on_edge
    traffic = {}
    colors = np.array(['green', 'orange', 'red', 'darkred'])
    for e in G.edges:
        for c in range(4):
            traffic[(e[0], e[1], e[2], c)] = []

    k = 0
    for i, j in zip(support_indices[0], support_indices[1]):
        if coordinate_in_box(LAT[i, j], LONG[i, j]):
            edge = tuple(index_edges[k])
            for c in range(4):
                if data[colors[c]][i, j] > 0 and edge in G.edges:
                    traffic[(edge[0], edge[1], e[2], c)].append(data[colors[c]][i, j])
            k = k + 1

    mean_traffic = {}
    for e in G.edges:
        for c in range(4):
            ec = (e[0], e[1], e[2], c)
            mean_traffic[ec] = len(traffic[ec])

    return pd.DataFrame.from_dict(mean_traffic, orient='index', columns=[t])
