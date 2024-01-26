from typing import Callable

import networkx as nx
import numpy as np
import pandas as pd
from scipy.sparse.linalg import gmres
from scipy.spatial.distance import cdist
from tqdm import tqdm

from PerplexityLab.miscellaneous import if_exist_load_else_do
from src.config import city_dir
from src.lib.DataProcessing.TrafficProcessing import TRAFFIC_VALUES
from src.lib.FeatureExtractors.FeatureExtractorsBase import FeatureExtractor


def get_graph_node_positions(graph):
    """

    :param graph:
    :return: [#nodes, #space dimension = 2]
    """
    return np.array([(graph.nodes[n]["x"], graph.nodes[n]["y"]) for n in graph.nodes])


class FEGraphNeighboringTraffic(FeatureExtractor):
    def __init__(self, name="", k_neighbours=1, **kwargs):
        assert k_neighbours >= 1 and isinstance(k_neighbours, int), "k_neighbours should be integer >= 1"
        self.k_neighbours = k_neighbours
        super().__init__(name=name, **kwargs)
        self.position2node_index = dict()
        self.nodes = None
        self.node_positions = None

    def extract_features(self, times, positions: pd.DataFrame, traffic_by_edge, graph, *args, **kwargs) -> np.ndarray:
        graph = self.preprocess_graph(graph)
        self.update_position2node_index(positions, re_update=self.k_neighbours is not None)
        nodes = [self.nodes[self.position2node_index[position]]
                 for position in zip(positions.loc["long"].values, positions.loc["lat"].values)]
        traffic_by_node = self.get_traffic_by_node(times, traffic_by_edge, graph, nodes)
        return np.reshape(traffic_by_node, (len(times), -1))

    def preprocess_graph(self, graph: nx.Graph):
        graph = nx.Graph(graph).to_undirected()
        if len(nx.get_edge_attributes(graph, "length")) != graph.number_of_edges():
            raise Exception("Each edge in Graph should have edge attribute 'length'")
        if len(nx.get_edge_attributes(graph, "lanes")) != graph.number_of_edges():
            raise Exception("Each edge in Graph should have edge attribute 'lanes'")
        self.position2node_index = dict()
        self.nodes = list(graph.nodes)
        self.node_positions = get_graph_node_positions(graph)
        return graph

    def update_position2node_index(self, target_positions: pd.DataFrame, re_update=False):
        for tp in map(tuple, target_positions.values.T):
            if re_update or tp not in self.position2node_index:
                # TODO: instead of looking for the nearset node, predict using the point in the edge
                self.position2node_index[tp] = int(np.argmin(cdist(self.node_positions, np.array([tp])), axis=0))

    def get_traffic_by_node(self, times, traffic_by_edge, graph, nodes):
        """

        :param times:
        :param traffic_by_edge:
        :param graph:
        :param nodes:
        :return: traffic_by_node: times x nodes x num features
        """
        traffic_by_node = np.zeros((len(times), len(nodes), len(TRAFFIC_VALUES) * self.k_neighbours))
        traffic_by_edge_normalization = {e: max((1, df.sum(axis=1).max())) for e, df in traffic_by_edge.items()}

        for j, node in enumerate(nodes):
            depth = {node: 0}
            depth_count = [0] * self.k_neighbours
            for edge in nx.bfs_tree(graph, source=node, depth_limit=1).edges():
                if edge[1] not in depth:
                    depth[edge[1]] = depth[edge[0]] + 1
                if edge in traffic_by_edge:
                    area = graph.edges[edge]["lanes"] * graph.edges[edge]["length"]
                    depth_count[min((depth[edge[0]], depth[edge[1]]))] += area
                    level = np.arange(len(TRAFFIC_VALUES), dtype=int) + depth[edge[0]] * len(TRAFFIC_VALUES)
                    traffic_by_node[:, j, level] += traffic_by_edge[edge].loc[times, :] / traffic_by_edge_normalization[
                        edge] * area
            for d, dc in enumerate(depth_count):
                level = np.arange(len(TRAFFIC_VALUES), dtype=int) + d * len(TRAFFIC_VALUES)
                # it can happen that no neighbouring edges have been with traffic so dc=0
                traffic_by_node[:, j, level] /= max((1, dc))
        return traffic_by_node


# ------------- auxiliary functions ------------ #
def compute_function_on_graph_edges(function, graph, edge_function: Callable):
    nx.set_edge_attributes(graph,
                           {(u, v): edge_function(data) for u, v, data in graph.edges.data()},
                           'weight')
    return function(graph, weight="weight")


def compute_adjacency(graph, edge_function: Callable):
    return compute_function_on_graph_edges(nx.adjacency_matrix, graph, edge_function)


def compute_laplacian_matrix_from_graph(graph, edge_function: Callable):
    # return (nx.laplacian_matrix(graph, weight="weight")).toarray()
    return compute_function_on_graph_edges(nx.laplacian_matrix, graph, edge_function)


def compute_degree_from_graph(graph, edge_function: Callable):
    return compute_function_on_graph_edges(nx.degree, graph, edge_function)


@if_exist_load_else_do(file_format="joblib", description=None)
def get_eigenvectors_of_graph(graph, weight="length"):
    """
    returns:
    eigenvals: in increasing order
    eigenvects: in columns
    """
    L = nx.normalized_laplacian_matrix(graph, weight=weight)
    eigvals, eigvects = np.linalg.eigh(L.toarray())
    return eigvals, eigvects


@if_exist_load_else_do(file_format="joblib", description=None)
def get_HeqMatrices(graph):
    graph = nx.Graph(graph).to_undirected()
    Ms = -compute_laplacian_matrix_from_graph(graph, edge_function=lambda data: data["length"]) / 6
    Ms[np.diag_indices(graph.number_of_nodes())] *= -2
    # diffusion matrix Kd
    Kd = compute_laplacian_matrix_from_graph(graph, edge_function=lambda data: 1.0 / data["length"])
    return Ms, Kd


# --------- diffusion models ---------- #
def diffusion_eq(f, graph, diffusion_coef, absorption_coef, path, recalculate=False):
    Ms, Kd = get_HeqMatrices(path=path, filename=f"GraphMatrices{graph.number_of_nodes()}", graph=graph,
                             recalculate=recalculate)
    if diffusion_coef == 0 and absorption_coef == 0:
        return f / np.diagonal(Ms.toarray()) * (1 / 2)
    else:
        A = diffusion_coef * Kd + absorption_coef * Ms
        return np.array([gmres(A, e, x0=e, maxiter=100)[0] for e in tqdm(f.T)]).T


def label_prop(f, graph, edge_function, lamb=1, iter=10, p=0.5):
    L = compute_laplacian_matrix_from_graph(nx.Graph(graph).to_undirected(), edge_function)
    L.data[np.isnan(L.data)] = 0
    A = compute_adjacency(nx.Graph(graph).to_undirected(), edge_function)
    A.data[np.isnan(A.data)] = 0
    d = np.reshape(np.ravel(A.sum(axis=0)), (-1, 1))
    u = f.copy()
    for i in range(iter):
        # L = A * D ** (-lamb) * D.T ** (lamb - 1)
        u = p * d ** (-lamb) * (L @ (d ** (lamb - 1) * u)) + (1 - p) * f
    return u


def eigen_filter(f, graph, threshold=0.9, recalculate_eigen=False):
    # smoothing
    eigvals, eigvects = get_eigenvectors_of_graph(
        path=city_dir, recalculate=recalculate_eigen,
        filename=f"Eigenvects_V{len(graph)}",
        graph=nx.Graph(graph).to_undirected(), weight="length")
    if isinstance(threshold, float):
        coefs = f.T @ eigvects
        energy = np.cumsum(coefs ** 2)
        energy /= energy[-1]
        return coefs[energy <= threshold] @ eigvects[:, energy <= threshold]
    else:
        return (f.T @ eigvects[:, :threshold]) @ eigvects[:, :threshold]


@if_exist_load_else_do(file_format="joblib", description=None)
def get_traffic_by_node(k_neighbours, times, traffic_by_edge, graph, nodes):
    traffic_by_node = np.zeros((len(times), len(nodes), len(TRAFFIC_VALUES) * k_neighbours))
    traffic_by_edge_normalization = {e: max((1, df.sum(axis=1).max())) for e, df in traffic_by_edge.items()}

    for j, node in enumerate(nodes):
        depth = {node: 0}
        depth_count = [0] * k_neighbours
        for edge in nx.bfs_tree(graph, source=node, depth_limit=1).edges():
            if edge[1] not in depth:
                depth[edge[1]] = depth[edge[0]] + 1
            if edge in traffic_by_edge:
                area = graph.edges[edge]["lanes"] * graph.edges[edge]["length"]
                depth_count[min((depth[edge[0]], depth[edge[1]]))] += area
                level = np.arange(len(TRAFFIC_VALUES), dtype=int) + depth[edge[0]] * len(TRAFFIC_VALUES)
                traffic_by_node[:, j, level] += traffic_by_edge[edge].loc[times, :] / traffic_by_edge_normalization[
                    edge] * area
        for d, dc in enumerate(depth_count):
            level = np.arange(len(TRAFFIC_VALUES), dtype=int) + d * len(TRAFFIC_VALUES)
            # it can happen that no neighbouring edges have been with traffic so dc=0
            traffic_by_node[:, j, level] /= max((1, dc))
    return traffic_by_node
