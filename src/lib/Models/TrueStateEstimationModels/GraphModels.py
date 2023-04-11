from typing import Callable, List

import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from numpy.linalg import solve

from src.lib.DataProcessing.TrafficProcessing import TRAFFIC_VALUES
from src.lib.Models.BaseModel import BaseModel, mse, GRAD, NONE_OPTIM_METHOD
from src.performance_utils import filter_dict, timeit


class GraphModelBase(BaseModel):

    def __init__(self, name="", loss=mse, optim_method=GRAD, verbose=False, **kwargs):
        # super call
        super().__init__(name=name, loss=loss, optim_method=optim_method, verbose=verbose, **kwargs)
        self.position2node_index = dict()
        self.nodes = None
        self.node_positions = None
        self.node2index = None

    def preprocess_graph(self, graph: nx.Graph):
        graph = nx.Graph(graph).to_undirected()
        if len(nx.get_edge_attributes(graph, "length")) != graph.number_of_edges():
            raise Exception("Each edge in Graph should have edge attribute 'length'")
        self.position2node_index = dict()
        self.node2index = {node: i for i, node in enumerate(graph.nodes)}
        self.nodes = list(graph.nodes)
        self.node_positions = np.array([(graph.nodes[n]["x"], graph.nodes[n]["y"]) for n in graph.nodes])
        return {"graph": graph}

    def state_estimation_core(self, emissions, **kwargs):
        raise Exception("Not implemented.")

    def get_traffic_by_node(self, traffic_by_edge, graph):
        traffic_by_node = np.zeros((len(traffic_by_edge), graph.number_of_nodes(), len(TRAFFIC_VALUES)))
        for e, df in traffic_by_edge.items():
            for i, color in enumerate(TRAFFIC_VALUES.keys()):
                update_val = df[color] * graph.edges[e]["length"] * graph.edges[e]["lanes"]
                traffic_by_node[:, self.node2index[e[0]], i] += update_val
                traffic_by_node[:, self.node2index[e[1]], i] += update_val
        return traffic_by_node

    def get_emissions(self, traffic):
        """

        :param traffic:
        :return: Linear problem.
        """
        return np.einsum("tnp,p", traffic, filter_dict(TRAFFIC_VALUES.keys(), **self.params))

    def state_estimation(self, observed_stations, observed_pollution, traffic, target_positions: pd.DataFrame,
                         **kwargs) -> np.ndarray:
        assert "traffic_by_edge" in kwargs is not None, f"Model {self} does not work if no traffic_by_edge is given."
        kwargs.update(self.preprocess_graph(kwargs["graph"]))
        self.update_position2node_index(target_positions)

        traffic_by_node = self.get_traffic_by_node(kwargs["traffic_by_edge"], kwargs["graph"])
        emissions = self.get_emissions(traffic_by_node)
        solutions = self.state_estimation_core(emissions, **kwargs)

        indexes = [self.position2node_index[position] for position in zip(target_positions.loc["long"].values,
                                                                          target_positions.loc["lat"].values)]
        return solutions[:, indexes]

    def calibrate(self, observed_stations, observed_pollution: pd.DataFrame, traffic, **kwargs):
        if self.optim_method is not NONE_OPTIM_METHOD:
            kwargs.update(self.preprocess_graph(kwargs["graph"]))
            traffic_by_node = self.get_traffic_by_node(kwargs["traffic_by_edge"], kwargs["graph"])
            emissions = self.get_emissions(traffic_by_node)
            return super(GraphModelBase, self).calibrate(self, observed_stations, observed_pollution, traffic,
                                                         emissions=emissions, **kwargs)

    @property
    def domain(self):
        return self.node_positions

    def update_position2node_index(self, target_positions: pd.DataFrame):
        for tp in map(tuple, target_positions.values.T):
            if tp not in self.position2node_index:
                # TODO: instead of looking for the nearset node, predict using the point in the edge
                self.position2node_index[tp] = int(np.argmin(cdist(self.node_positions, np.array([tp])), axis=0))


# ================================================== #
# -------- prepare_graph auxiliary functions ------- #

# ------------- auxiliary functions ------------ #
def compute_laplacian_matrix_from_graph(graph, edge_function: Callable):
    nx.set_edge_attributes(graph,
                           {(u, v): edge_function(data) for u, v, data in graph.edges.data()},
                           'weight')
    return (nx.laplacian_matrix(graph, weight="weight")).toarray()


class HEqStaticModel(GraphModelBase):
    def __init__(self, absorption: float, diffusion: float, traffic_params: List[float], name="",
                 loss=mse, optim_method="lsq", verbose=False, **kwargs):
        # super call
        params = {param_name: tp for param_name, tp in zip(TRAFFIC_VALUES, traffic_params)}
        params.update({"absorption": absorption, "diffusion": diffusion})
        super().__init__(name=name, loss=loss, optim_method=optim_method, verbose=verbose, **params)

    def preprocess_graph(self, graph: nx.Graph):
        preprocess_dict = super(HEqStaticModel, self).preprocess_graph(graph)
        with timeit(f"Creating system matrices for HEqStaticModel."):
            # absorption matrix Ms
            preprocess_dict["Ms"] = -compute_laplacian_matrix_from_graph(preprocess_dict["graph"],
                                                                         edge_function=lambda data: data[
                                                                             "length"]) / 6
            preprocess_dict["Ms"][np.diag_indices(preprocess_dict["graph"].number_of_nodes())] *= -2
            # diffusion matrix Kd
            preprocess_dict["Kd"] = compute_laplacian_matrix_from_graph(preprocess_dict["graph"],
                                                                        edge_function=lambda data: 1.0 / data[
                                                                            "length"])
        preprocess_dict.pop("graph")
        return preprocess_dict

    def state_estimation_core(self, emissions, Kd, Ms, **kwargs):
        return solve(self.params["diffusion"] * Kd + self.params["absorption"] * Ms, emissions.T).T
