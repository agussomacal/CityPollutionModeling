from typing import Callable, List, Union

import networkx as nx
import numpy as np
import pandas as pd
from scipy.linalg import lu_factor, lu_solve
from scipy.sparse.linalg import spsolve
from scipy.spatial.distance import cdist
from numpy.linalg import solve
from tqdm import tqdm

from src.lib.DataProcessing.TrafficProcessing import TRAFFIC_VALUES
from src.lib.Models.BaseModel import BaseModel, mse, GRAD, NONE_OPTIM_METHOD, Bounds, Optim
from src.performance_utils import filter_dict, timeit


class GraphModelBase(BaseModel):

    def __init__(self, name="", loss=mse, optim_method=GRAD, verbose=False, niter=1000, **kwargs):
        # super call
        super().__init__(name=name, loss=loss, optim_method=optim_method, verbose=verbose, niter=niter, **kwargs)
        self.position2node_index = dict()
        self.nodes = None
        self.node_positions = None
        self.node2index = None
        self.last_nodes = None
        self.last_times = None
        self.traffic_by_node = None

    def preprocess_graph(self, graph: nx.Graph):
        graph = nx.Graph(graph).to_undirected()
        if len(nx.get_edge_attributes(graph, "length")) != graph.number_of_edges():
            raise Exception("Each edge in Graph should have edge attribute 'length'")
        if len(nx.get_edge_attributes(graph, "lanes")) != graph.number_of_edges():
            raise Exception("Each edge in Graph should have edge attribute 'lanes'")
        self.position2node_index = dict()
        self.node2index = {node: i for i, node in enumerate(graph.nodes)}
        self.nodes = list(graph.nodes)
        self.node_positions = np.array([(graph.nodes[n]["x"], graph.nodes[n]["y"]) for n in graph.nodes])
        return {"graph": graph}

    def state_estimation_core(self, emissions, **kwargs):
        raise Exception("Not implemented.")

    def get_traffic_by_node(self, observed_pollution, traffic_by_edge, graph):
        # TODO: not recalculate if graph have been already seen.
        traffic_by_node = np.zeros((len(observed_pollution), graph.number_of_nodes(), len(TRAFFIC_VALUES)))
        for e, df in traffic_by_edge.items():
            for i, color in enumerate(TRAFFIC_VALUES):
                # length is added because we are doing the integral against the P1 elements.
                # a factor of 1/2 may be added too.
                update_val = df.loc[observed_pollution.index, color] * graph.edges[e]["length"] * graph.edges[e][
                    "lanes"] / 2
                traffic_by_node[:, self.node2index[e[0]], i] += update_val
                traffic_by_node[:, self.node2index[e[1]], i] += update_val
        return traffic_by_node

    # def get_traffic_by_node(self, observed_pollution, traffic_by_edge, graph):
    #     # if self.last_nodes != set(graph.nodes) or self.last_times != set(list(observed_pollution.index)):
    #     #     self.last_nodes = set(graph.nodes)
    #     #     self.last_times = set(list(observed_pollution.index))
    #         self.traffic_by_node = np.zeros((len(observed_pollution), graph.number_of_nodes(), len(TRAFFIC_VALUES)))
    #         for e, df in traffic_by_edge.items():
    #             for i, color in enumerate(TRAFFIC_VALUES):
    #                 # length is added because we are doing the integral against the P1 elements.
    #                 # a factor of 1/2 may be added too.
    #                 update_val = df.loc[observed_pollution.index, color] * graph.edges[e]["length"] * graph.edges[e][
    #                     "lanes"] / 2
    #                 self.traffic_by_node[:, self.node2index[e[0]], i] += update_val
    #                 self.traffic_by_node[:, self.node2index[e[1]], i] += update_val
    #     return self.traffic_by_node

    def get_emissions(self, traffic_by_node):
        """

        :param traffic:
        :return: Linear problem.
        """
        return np.einsum("tnp,p", traffic_by_node, list(filter_dict(TRAFFIC_VALUES, self.params).values()))

    def state_estimation(self, observed_stations, observed_pollution, traffic, target_positions: pd.DataFrame,
                         **kwargs) -> np.ndarray:
        assert "traffic_by_edge" in kwargs is not None, f"Model {self} does not work if no traffic_by_edge is given."
        kwargs.update(self.preprocess_graph(kwargs["graph"]))
        with timeit("update postitions"):
            self.update_position2node_index(target_positions)
        with timeit("traffic by node"):
            traffic_by_node = self.get_traffic_by_node(observed_pollution, kwargs["traffic_by_edge"], kwargs["graph"])
        with timeit("emissions"):
            emissions = self.get_emissions(traffic_by_node)
        with timeit("solutions"):
            solutions = self.state_estimation_core(emissions, **kwargs)

        with timeit("indexes"):
            indexes = [self.position2node_index[position] for position in zip(target_positions.loc["long"].values,
                                                                              target_positions.loc["lat"].values)]
        return solutions[:, indexes]

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
    POLLUTION_AGNOSTIC = True

    def __init__(self, absorption: Union[float, Optim], diffusion: Union[float, Optim], green=Union[float, Optim],
                 yellow=Union[float, Optim], red=Union[float, Optim], dark_red=Union[float, Optim], name="",
                 loss=mse, optim_method="lsq", verbose=False, niter=1000):
        super().__init__(name=name, loss=loss, optim_method=optim_method, verbose=verbose,
                         absorption=absorption, diffusion=diffusion, green=green, yellow=yellow, red=red,
                         dark_red=dark_red, niter=niter)

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
            # A = self.params["diffusion"] * preprocess_dict["Kd"] + self.params["absorption"] * preprocess_dict["Ms"]
            # preprocess_dict["lu"], preprocess_dict["piv"] = lu_factor(A)
        return preprocess_dict

    def state_estimation_core(self, emissions, Kd, Ms, **kwargs):
        A = self.params["diffusion"] * Kd + self.params["absorption"] * Ms
        return spsolve(A, emissions).T
        # n=2 1.11s/it A calculated each time: 16min A precalculated: 10.3min
        # A precalculated: 30s
        # return np.concatenate(
        #     [spsolve(A, emissions[:, i:i + n]).T for i in
        #      tqdm(range(0, emissions.shape[1], n))])

        # return np.concatenate(
        #     [solve(A, emissions[:, i:i + n]).T for i in
        #      tqdm(range(0, emissions.shape[1], n))])

        # 18 it/s with 1823 samples to perform: 1:40 min.
        # return np.concatenate(
        #     [lu_solve((lu, piv), emissions[:, i:i + n], overwrite_b=True, check_finite=False).T for i in
        #      tqdm(range(0, emissions.shape[1], n))])
        # return lu_solve((lu, piv), emissions, overwrite_b=True, check_finite=False).T
        # return solve(self.params["diffusion"] * Kd + self.params["absorption"] * Ms, emissions).T
