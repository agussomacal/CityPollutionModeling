import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from PerplexityLab.miscellaneous import if_exist_load_else_do
from src.lib.DataProcessing.TrafficProcessing import TRAFFIC_VALUES
from src.lib.FeatureExtractors.FeatureExtractorsBase import FeatureExtractor


def get_graph_node_positions(graph):
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