from typing import Callable, Union

import networkx as nx
import numpy as np
import pandas as pd
from scipy.sparse.linalg import gmres
from scipy.spatial.distance import cdist
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from tqdm import tqdm

from PerplexityLab.miscellaneous import filter_dict, timeit
from src.experiments.config_experiments import screenshot_period
from src.lib.DataProcessing.TrafficProcessing import TRAFFIC_VALUES, load_background
from src.lib.FeatureExtractors.ConvolutionFeatureExtractors import FEConvolutionFixedPixels, WaterColor, GreenAreaColor
from src.lib.Models.BaseModel import BaseModel, mse, GRAD, pollution_agnostic
from src.lib.Models.TrueStateEstimationModels.TrafficConvolution import gaussker
from src.lib.Modules import Optim


def check_is_fitted(estimator):
    """Perform is_fitted validation for estimator.

    Checks if the estimator is fitted by verifying the presence of
    fitted attributes (ending with a trailing underscore) and otherwise
    raises a NotFittedError with the given message.

    If an estimator does not set any attributes with a trailing underscore, it
    can define a ``__sklearn_is_fitted__`` method returning a boolean to specify if the
    estimator is fitted or not.

    Parameters
    ----------
    estimator : estimator instance
        Estimator instance for which the check is performed.

    attributes : str, list or tuple of str, default=None
        Attribute name(s) given as string or a list/tuple of strings
        Eg.: ``["coef_", "estimator_", ...], "coef_"``

        If `None`, `estimator` is considered fitted if there exist an
        attribute that ends with a underscore and does not start with double
        underscore.

    msg : str, default=None
        The default error message is, "This %(name)s instance is not fitted
        yet. Call 'fit' with appropriate arguments before using this
        estimator."

        For custom messages if "%(name)s" is present in the message string,
        it is substituted for the estimator name.

        Eg. : "Estimator, %(name)s, must be fitted before sparsifying".

    all_or_any : callable, {all, any}, default=all
        Specify whether all or any of the given attributes must exist.

    Raises
    ------
    TypeError
        If the estimator is a class or not an estimator instance

    NotFittedError
        If the attributes are not found.
    """

    if not hasattr(estimator, "fit"):
        raise TypeError("%s is not an estimator instance." % (estimator))

    if hasattr(estimator, "__sklearn_is_fitted__"):
        fitted = estimator.__sklearn_is_fitted__()
    else:
        fitted = [
            v for v in vars(estimator) if v.endswith("_") and not v.startswith("__")
        ]

    if not fitted:
        return False
    return True


class GraphModelBase(BaseModel):

    def __init__(self, name="", loss=mse, optim_method=GRAD, verbose=False, niter=1000, k_neighbours=None, sigma0=1,
                 **kwargs):
        # super call
        super().__init__(name=name, loss=loss, optim_method=optim_method, verbose=verbose, niter=niter, sigma0=sigma0,
                         **kwargs)
        self.position2node_index = dict()
        self.nodes = None
        self.node_positions = None
        self.node2index = None
        self.last_nodes = None
        self.last_times = None
        self.traffic_by_node = None
        self.k_neighbours = k_neighbours

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

    def get_traffic_by_node(self, observed_pollution, traffic_by_edge, graph, nodes=None):
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
        if self.last_nodes != set(nodes) or self.last_times != set(list(observed_pollution.index)):
            # TODO: make it incremental instead of replacing the whole matrix.
            self.last_nodes = set(nodes)
            self.last_times = set(list(observed_pollution.index))
            self.traffic_by_node = np.zeros((len(observed_pollution), len(nodes), len(TRAFFIC_VALUES)))
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

    def get_emissions(self, traffic_by_node):
        """

        :param traffic:
        :return: Linear problem.
        """
        return np.einsum("tnp,p->tn", traffic_by_node, list(filter_dict(TRAFFIC_VALUES, self.params).values()))

    def state_estimation(self, observed_stations, observed_pollution, traffic, target_positions: pd.DataFrame,
                         **kwargs) -> np.ndarray:
        assert "traffic_by_edge" in kwargs is not None, f"Model {self} does not work if no traffic_by_edge is given."
        kwargs.update(self.preprocess_graph(kwargs["graph"]))
        with timeit("update postitions"):
            self.update_position2node_index(target_positions, re_update=self.k_neighbours is not None)

        with timeit("indexes"):
            indexes = [self.position2node_index[position] for position in zip(target_positions.loc["long"].values,
                                                                              target_positions.loc["lat"].values)]
        if self.k_neighbours is not None and len(indexes) < len(kwargs["graph"]) // 4:
            nodes = set()
            for i in indexes:
                nodes.update(list(nx.single_source_shortest_path_length(kwargs["graph"], list(kwargs["graph"].nodes)[i],
                                                                        cutoff=self.k_neighbours).keys()))
            kwargs["graph"] = nx.subgraph(kwargs["graph"], nodes)
            print(f"After filtering with {self.k_neighbours}-neighbours: "
                  f"{kwargs['graph'].number_of_nodes()} nodes and "
                  f"{kwargs['graph'].number_of_edges()} edges.")
            kwargs.update(self.preprocess_graph(kwargs["graph"]))
            with timeit("update postitions"):
                self.update_position2node_index(target_positions, re_update=True)

            with timeit("indexes"):
                indexes = [self.position2node_index[position] for position in zip(target_positions.loc["long"].values,
                                                                                  target_positions.loc["lat"].values)]

        with timeit("traffic by node"):
            traffic_by_node = self.get_traffic_by_node(observed_pollution, kwargs["traffic_by_edge"], kwargs["graph"])
        with timeit("emissions"):
            emissions = self.get_emissions(traffic_by_node)

        with timeit("solutions"):
            solutions = self.state_estimation_core(emissions, **kwargs)

        return solutions[:, indexes]

    def update_position2node_index(self, target_positions: pd.DataFrame, re_update=False):
        for tp in map(tuple, target_positions.values.T):
            if re_update or tp not in self.position2node_index:
                # TODO: instead of looking for the nearset node, predict using the point in the edge
                self.position2node_index[tp] = int(np.argmin(cdist(self.node_positions, np.array([tp])), axis=0))


# ================================================== #
# -------- prepare_graph auxiliary functions ------- #

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


class GraphEmissionsNeigEdgeModel(GraphModelBase):
    POLLUTION_AGNOSTIC = True

    def __init__(self, model: Pipeline, k_neighbours=1, name="", loss=mse, verbose=False, optim_method=GRAD,
                 extra_regressors=[], **kwargs):
        self.model = model
        assert k_neighbours >= 1 and isinstance(k_neighbours, int), "k_neighbours should be integer >= 1"
        # self.fit_intercept = fit_intercept
        self.extra_regressors = extra_regressors
        super().__init__(
            name=name + "_".join([step[0] for step in model.steps]),
            loss=loss, verbose=verbose,
            k_neighbours=k_neighbours, optim_method=optim_method, **kwargs)

    def get_traffic_by_node(self, times, traffic_by_edge, graph, nodes, nodes_ix):
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

    def reduce_traffic(self, times, target_positions: pd.DataFrame, **kwargs) -> np.ndarray:
        indexes = [self.position2node_index[position]
                   for position in zip(target_positions.loc["long"].values, target_positions.loc["lat"].values)]
        nodes = [list(kwargs["graph"].nodes)[i] for i in indexes]
        return self.get_traffic_by_node(times, kwargs["traffic_by_edge"], kwargs["graph"], nodes, indexes)

    def traffic2pollution(self, times, positions, **kwargs):
        # return np.einsum("tnp,p->tn", traffic_by_node, list(filter_dict(TRAFFIC_VALUES, self.params).values())) + \
        #     self.params["intercept"]
        X = self.prepare_input2model(times=times, positions=positions, **kwargs)
        return self.model.predict(X).reshape((len(times), np.shape(positions)[-1]))
        # X = traffic_by_node.reshape((-1, np.shape(traffic_by_node)[-1]))
        # return self.model.predict(X).reshape(np.shape(traffic_by_node)[:-1])

    def state_estimation(self, observed_stations, observed_pollution, traffic, target_positions: pd.DataFrame,
                         **kwargs) -> np.ndarray:
        """
        traffic_coords: pd.DataFrame with columns the pixel_coord and rows 'lat' and 'long' associated to the pixel
        target_positions: pd.DataFrame with columns the name of the station and rows 'lat' and 'long'
        """
        assert "traffic_by_edge" in kwargs is not None, f"Model {self} does not work if no traffic_by_edge is given."
        kwargs.update(self.preprocess_graph(kwargs["graph"]))
        self.update_position2node_index(observed_stations, re_update=self.k_neighbours is not None)
        self.update_position2node_index(target_positions, re_update=self.k_neighbours is not None)
        # traffic_by_node = self.reduce_traffic(observed_pollution.index, target_positions, **kwargs)
        # return self.traffic2pollution(traffic_by_node)
        return self.traffic2pollution(times=observed_pollution.index, positions=target_positions, **kwargs)

    @pollution_agnostic
    def state_estimation_for_optim(self, observed_stations, observed_pollution, traffic, **kwargs) -> [np.ndarray,
                                                                                                       np.ndarray]:
        """
        traffic_coords: pd.DataFrame with columns the pixel_coord and rows 'lat' and 'long' associated to the pixel
        target_positions: pd.DataFrame with columns the name of the station and rows 'lat' and 'long'
        """
        assert "traffic_by_edge" in kwargs is not None, f"Model {self} does not work if no traffic_by_edge is given."
        kwargs.update(self.preprocess_graph(kwargs["graph"]))
        self.update_position2node_index(observed_stations, re_update=self.k_neighbours is not None)
        if not check_is_fitted(self.model):
            X = self.prepare_input2model(times=observed_pollution.index, positions=observed_stations, **kwargs)
            y = observed_pollution.values.reshape((-1, 1))
            mask = np.all(~np.isnan(X), axis=1) * np.all(~np.isnan(y), axis=1)
            self.model.fit(X[mask], y[mask])
        return self.traffic2pollution(times=observed_pollution.index, positions=observed_stations, **kwargs)

    def prepare_input2model(self, times, positions, **kwargs):
        X = []
        traffic = self.reduce_traffic(times, positions, **kwargs)

        for regressor_name in self.extra_regressors:
            if regressor_name in ["water", "green"]:
                if regressor_name in ["water"]:
                    img = load_background(screenshot_period)
                    regressor = FEConvolutionFixedPixels(name="water", mask=np.all(img == WaterColor, axis=-1),
                                                         x_coords=kwargs["longitudes"], normalize=False,
                                                         y_coords=kwargs["latitudes"], agg_func=np.sum,
                                                         kernel=gaussker, sigma=0.1).extract_features(times,
                                                                                                      positions)[:, :,
                                np.newaxis] * np.mean(traffic, axis=-1, keepdims=True)
                elif regressor_name in ["green"]:
                    img = load_background(screenshot_period)
                    regressor = FEConvolutionFixedPixels(name="green", mask=np.all(img == GreenAreaColor, axis=-1),
                                                         x_coords=kwargs["longitudes"], normalize=False,
                                                         y_coords=kwargs["latitudes"], agg_func=np.sum,
                                                         kernel=gaussker, sigma=0.1).extract_features(times,
                                                                                                      positions)[:, :,
                                np.newaxis] * np.mean(traffic, axis=-1, keepdims=True)

                else:
                    continue
            else:
                regressor = kwargs.get(regressor_name, None)
                if regressor is not None:
                    if regressor_name in ["temperature", "wind"]:
                        regressor.index.intersection(times)
                        regressor = np.transpose([regressor.loc[times, :].values.ravel()] * np.shape(positions)[-1])[:,
                                    :, np.newaxis]

                    else:
                        continue
            X.append(regressor)

        X.append(traffic)
        X = np.concatenate(X, axis=-1)
        X = X.reshape((-1, np.shape(X)[-1]))
        # print(np.shape(X))
        return X


class GraphEmissionsModel(GraphModelBase):
    POLLUTION_AGNOSTIC = True

    def __init__(self, tau: Union[float, Optim] = None, gamma: Union[float, Optim] = None,
                 # green: Union[float, Optim] = None, yellow: Union[float, Optim] = None,
                 # red: Union[float, Optim] = None, dark_red: Union[float, Optim] = None,
                 k_neighbours=1, model=LinearRegression(), name="", loss=mse, verbose=False, optim_method=GRAD,
                 # fit_intercept=False,
                 **kwargs):
        self.model = model
        assert k_neighbours >= 1 and isinstance(k_neighbours, int), "k_neighbours should be integer >= 1"
        # self.fit_intercept = fit_intercept
        super().__init__(
            # name=name + if_true_str(fit_intercept, "_Intercept"),
            name=name + str(model),
            loss=loss, verbose=verbose,
            k_neighbours=k_neighbours, optim_method=optim_method,
            # green=green, yellow=yellow, red=red, dark_red=dark_red,
            tau=tau, gamma=gamma, **kwargs)

    def get_traffic_by_node(self, times, traffic_by_edge, graph, nodes, nodes_ix, tau, gamma=1):
        traffic_by_node = np.zeros((len(times), len(nodes), len(TRAFFIC_VALUES)))
        deg = compute_adjacency(graph, lambda data: data["length"] * data["lanes"]).toarray().sum(axis=1)
        node2ix = {n: i for i, n in enumerate(graph.nodes)}
        traffic_by_edge_normalization = {e: df.sum(axis=1).max() for e, df in traffic_by_edge.items()}

        for c, color in enumerate(TRAFFIC_VALUES):
            for i, t in enumerate(times):
                for j, node in enumerate(nodes):
                    depth = {node: 0}
                    for edge in nx.bfs_tree(graph, source=node, depth_limit=self.k_neighbours).edges():
                        if edge[1] not in depth:
                            depth[edge[1]] = depth[edge[0]] + 1
                        if edge in traffic_by_edge:
                            traffic_by_node[i, j, c] += \
                                tau ** depth[edge[0]] / deg[node2ix[edge[0]]] ** gamma / deg[node2ix[edge[1]]] ** (
                                        1 - gamma) * \
                                traffic_by_edge[edge].loc[t, color] / traffic_by_edge_normalization[edge] * \
                                graph.edges[edge]["lanes"] * graph.edges[edge]["length"]

        return traffic_by_node

    def get_traffic_by_node(self, times, traffic_by_edge, graph, nodes, nodes_ix):
        traffic_by_node = np.zeros((len(times), len(nodes), len(TRAFFIC_VALUES) * self.k_neighbours))
        traffic_by_edge_normalization = {e: df.sum(axis=1).max() for e, df in traffic_by_edge.items()}

        for j, node in enumerate(nodes):
            depth = {node: 0}
            depth_count = [0] * self.k_neighbours
            for edge in nx.bfs_tree(graph, source=node, depth_limit=self.k_neighbours).edges():
                if edge[1] not in depth:
                    depth[edge[1]] = depth[edge[0]] + 1
                if edge in traffic_by_edge:
                    area = graph.edges[edge]["lanes"] * graph.edges[edge]["length"]
                    depth_count[edge[0]] += area
                    level = np.arange(len(TRAFFIC_VALUES), dtype=int) + depth[edge[0]] * len(TRAFFIC_VALUES)
                    traffic_by_node[:, j, level] += traffic_by_edge[edge] / traffic_by_edge_normalization[edge] * area
            for d, dc in enumerate(depth_count):
                level = np.arange(len(TRAFFIC_VALUES), dtype=int) + d * len(TRAFFIC_VALUES)
                traffic_by_node[:, j, level] /= dc
        return traffic_by_node

    def reduce_traffic(self, times, target_positions: pd.DataFrame, **kwargs) -> np.ndarray:
        indexes = [self.position2node_index[position]
                   for position in zip(target_positions.loc["long"].values, target_positions.loc["lat"].values)]
        nodes = [list(kwargs["graph"].nodes)[i] for i in indexes]
        traffic_by_node = self.get_traffic_by_node(times, kwargs["traffic_by_edge"], kwargs["graph"],
                                                   nodes, indexes, self.params["tau"], self.params["gamma"])
        return traffic_by_node  # / np.array([d for _, d in degree])[np.newaxis, indexes, np.newaxis]

    def traffic2pollution(self, traffic_by_node):
        # return np.einsum("tnp,p->tn", traffic_by_node, list(filter_dict(TRAFFIC_VALUES, self.params).values())) + \
        #     self.params["intercept"]
        X = traffic_by_node.reshape((-1, np.shape(traffic_by_node)[-1]))
        return self.model.predict(X).reshape(np.shape(traffic_by_node)[:-1])
        # return np.squeeze([self.model.predict(tbn) for tbn in traffic_by_node])

    def state_estimation(self, observed_stations, observed_pollution, traffic, target_positions: pd.DataFrame,
                         **kwargs) -> np.ndarray:
        """
        traffic_coords: pd.DataFrame with columns the pixel_coord and rows 'lat' and 'long' associated to the pixel
        target_positions: pd.DataFrame with columns the name of the station and rows 'lat' and 'long'
        """
        assert "traffic_by_edge" in kwargs is not None, f"Model {self} does not work if no traffic_by_edge is given."
        kwargs.update(self.preprocess_graph(kwargs["graph"]))
        self.update_position2node_index(observed_stations, re_update=self.k_neighbours is not None)
        self.update_position2node_index(target_positions, re_update=self.k_neighbours is not None)
        traffic_by_node = self.reduce_traffic(observed_pollution.index, target_positions, **kwargs)
        return self.traffic2pollution(traffic_by_node)

    @pollution_agnostic
    def state_estimation_for_optim(self, observed_stations, observed_pollution, traffic, **kwargs) -> [np.ndarray,
                                                                                                       np.ndarray]:
        """
        traffic_coords: pd.DataFrame with columns the pixel_coord and rows 'lat' and 'long' associated to the pixel
        target_positions: pd.DataFrame with columns the name of the station and rows 'lat' and 'long'
        """
        assert "traffic_by_edge" in kwargs is not None, f"Model {self} does not work if no traffic_by_edge is given."
        kwargs.update(self.preprocess_graph(kwargs["graph"]))
        # stations = [station for station in observed_stations.columns if station in kwargs["stations2test"]] \
        #     if "stations2test" in kwargs else observed_stations.columns
        # observed_stations = observed_stations[stations]
        # observed_pollution = observed_pollution[stations]

        self.update_position2node_index(observed_stations, re_update=self.k_neighbours is not None)
        traffic_by_node = self.reduce_traffic(observed_pollution.index, observed_stations, **kwargs)
        # use median to fit to avoid outliers to interfere

        X = traffic_by_node.reshape((-1, np.shape(traffic_by_node)[-1]))
        y = observed_pollution.values.reshape((-1, 1))
        mask = np.all(~np.isnan(X), axis=1) * np.all(~np.isnan(y), axis=1)
        self.model.fit(X[mask], y[mask])
        # self.model.fit(np.nanmedian(traffic_by_node, axis=0), np.nanmean(observed_pollution.values, axis=0))
        # lr = LinearRegression(fit_intercept=self.fit_intercept).fit(np.nanmedian(traffic_by_node, axis=0),
        #                                                             np.nanmean(observed_pollution.values,
        #                                                                        axis=0))
        # self.set_params(intercept=lr.intercept_)
        # self.set_params(**dict(zip(TRAFFIC_VALUES, lr.coef_)))
        return self.traffic2pollution(traffic_by_node)


class HEqStaticModel(GraphModelBase):
    POLLUTION_AGNOSTIC = True

    def __init__(self, absorption: Union[float, Optim], diffusion: Union[float, Optim], green: Union[float, Optim],
                 yellow: Union[float, Optim], red: Union[float, Optim], dark_red: Union[float, Optim],
                 k_neighbours=None, name="", loss=mse, optim_method="lsq", verbose=False, niter=1000, sigma0=1):
        super().__init__(name=name, loss=loss, optim_method=optim_method, verbose=verbose, k_neighbours=k_neighbours,
                         absorption=absorption, diffusion=diffusion, green=green, yellow=yellow, red=red,
                         dark_red=dark_red, niter=niter, sigma0=sigma0)

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
        print(self.params["diffusion"], self.params["absorption"])
        if self.params["diffusion"] == 0 and self.params["absorption"] == 0:
            return emissions / np.diagonal(Ms.toarray()) * (1 / 2)
        else:
            A = self.params["diffusion"] * Kd + self.params["absorption"] * Ms
            return np.array([gmres(A, e, x0=e, maxiter=100)[0] for e in tqdm(emissions)])
        # return spsolve(csr_matrix(A), emissions).T
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
