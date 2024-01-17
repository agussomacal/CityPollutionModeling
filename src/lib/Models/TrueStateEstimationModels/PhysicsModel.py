from pathlib import Path
from typing import Union, List

import cma
import networkx as nx
import numpy as np
import pandas as pd
from scipy.linalg import eigh
from scipy.optimize import minimize
from scipy.sparse.linalg import gmres
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from skopt import gp_minimize
from tqdm import tqdm

from PerplexityLab.miscellaneous import filter_dict, timeit, if_exist_load_else_do
from src.experiments.config_experiments import screenshot_period
from src.lib.DataProcessing.TrafficProcessing import TRAFFIC_VALUES, load_background
from src.lib.FeatureExtractors.ConvolutionFeatureExtractors import FEConvolutionFixedPixels, WaterColor, GreenAreaColor
from src.lib.FeatureExtractors.GraphFeatureExtractors import get_graph_node_positions, compute_adjacency, \
    compute_laplacian_matrix_from_graph
from src.lib.Models.BaseModel import BaseModel, mse, GRAD, pollution_agnostic, loo, CMA, RANDOM, UNIFORM, LOGUNIFORM, \
    NONE_OPTIM_METHOD, BAYES
from src.lib.Models.TrueStateEstimationModels.TrafficConvolution import gaussker
from src.lib.Modules import Optim


@if_exist_load_else_do(file_format="joblib", loader=None, saver=None, description=None, check_hash=False)
def get_absorption_matrix(graph: nx.Graph):
    # absorption matrix Ms
    Ms = -compute_laplacian_matrix_from_graph(graph, edge_function=lambda data: data["length"]) / 6
    Ms[np.diag_indices(graph.number_of_nodes())] *= -2
    return Ms


@if_exist_load_else_do(file_format="joblib", loader=None, saver=None, description=None, check_hash=False)
def get_diffusion_matrix(graph: nx.Graph):
    # diffusion matrix Kd
    Kd = compute_laplacian_matrix_from_graph(graph, edge_function=lambda data: 1.0 / data["length"])
    return Kd


@if_exist_load_else_do(file_format="npy", loader=None, saver=None, description=None, check_hash=False)
def get_geometric_basis(Kd, Ms, k) -> (np.ndarray, ...):
    """

    :param Kd:
    :param Ms:
    :param k:
    :return:
    vk: basis of diffusion (#nodes, k)
    vm: basis of absorption (#nodes, k)
    """
    _, vk = eigh(Kd if isinstance(Kd, np.ndarray) else Kd.todense(), b=None, subset_by_index=[0, k])
    _, vm = eigh(Ms if isinstance(Ms, np.ndarray) else Ms.todense(), b=None, subset_by_index=[0, k])
    return vk, vm


@if_exist_load_else_do(file_format="joblib", loader=None, saver=None, description=None, check_hash=False)
def get_reduced_matrices(Kd, Ms, basis):
    """

    :param Kd:
    :param Ms:
    :param basis: (#nodes, k)
    :return:
    """
    KdROM = basis.T @ Kd @ basis
    MsROM = basis.T @ Ms @ basis
    return KdROM, MsROM


@if_exist_load_else_do(file_format="joblib", loader=None, saver=None, description=None, check_hash=False)
def get_nearest_node_mapping(target_positions: pd.DataFrame, graph: nx.Graph):
    """

    :param Kd:
    :param Ms:
    :param basis: (#nodes, k)
    :return:
    """
    node_positions = get_graph_node_positions(graph)
    position2node_index = {tuple(tp): i for i, tp in enumerate(node_positions)}
    for tp in map(tuple, target_positions.values.T):
        # TODO: instead of looking for the nearset node, predict using the point in the edge
        position2node_index[tp] = int(np.argmin(cdist(node_positions, np.array([tp])), axis=0))
    return position2node_index


@if_exist_load_else_do(file_format="npy", loader=None, saver=None, description=None, check_hash=False)
def get_basis_point_evaluations(basis: np.ndarray, target_positions: pd.DataFrame, position2node_index):
    """

    :param Kd:
    :param Ms:
    :param basis: (#nodes, k)
    :return:
    (m, k)
    """
    return np.array([basis[position2node_index[tp], :] for tp in map(tuple, target_positions.values.T)])


# @if_exist_load_else_do(file_format="npy", loader=None, saver=None, description=None, check_hash=False)
def get_traffic_by_node(times, traffic_by_edge, graph):
    """

    :param observed_pollution:
    :param traffic_by_edge:
    :param graph:
    :param nodes:
    :return: traffic_by_node: [#times, #nodes, #traffic colors]
    """
    node2index = {n: i for i, n in enumerate(graph.nodes)}

    nodes = list(graph.nodes)
    deg = compute_adjacency(graph, lambda data: data["length"] * data["lanes"]).toarray().sum(axis=1)

    # TODO: make it incremental instead of replacing the whole matrix.
    traffic_by_node = np.zeros((len(times), len(nodes), len(TRAFFIC_VALUES)))
    for edge, df in traffic_by_edge.items():
        if (edge in graph.edges) and (edge[0] in nodes or edge[1] in nodes):
            for i, color in enumerate(TRAFFIC_VALUES):
                # length is added because we are doing the integral against the P1 elements.
                # a factor of 1/2 may be added too.
                update_val = df.loc[times, color]
                update_val *= graph.edges[edge]["length"] * graph.edges[edge]["lanes"] / 2

                if edge[0] in nodes:
                    traffic_by_node[:, node2index[edge[0]], i] += \
                        update_val / deg[node2index[edge[0]]]

                if edge[1] in nodes:
                    traffic_by_node[:, node2index[edge[1]], i] += \
                        update_val / deg[node2index[edge[1]]]
    return traffic_by_node


@if_exist_load_else_do(file_format="npy", loader=None, saver=None, description=None, check_hash=False)
def get_traffic_by_node_conv(times, traffic_by_edge, graph, lnei):
    """

    :param times:
    :param traffic_by_edge:
    :param graph:
    :param lnei:
    :return: [#times, #nodes, #traffic colors]
    """
    traffic_by_node = get_traffic_by_node(times, traffic_by_edge, graph)  # [#times, #nodes, #traffic colors]

    node2index = {n: i for i, n in enumerate(graph.nodes)}
    for node in tqdm(graph.nodes(), "new traffic by node"):
        nodes = [node2index[n] for n in nx.bfs_tree(graph, source=node, depth_limit=lnei).nodes()]
        traffic_by_node[:, node2index[node], :] = traffic_by_node[:, nodes, :].mean(axis=1)
    return traffic_by_node


@if_exist_load_else_do(file_format="npy", loader=None, saver=None, description=None, check_hash=False)
def get_basis_traffic_by_node(basis: np.ndarray, traffic_by_node):
    """

    :param basis: (#nodes, k)
    :param traffic_by_node: [#times, #nodes, #traffic colors]
    :return: [#times, #colors, k]
    """
    return np.einsum("nk,tnc->tkc", basis, traffic_by_node)


def extra_regressors(times: pd.DatetimeIndex, positions, extra_regressors, **kwargs):
    X = []

    for regressor_name in extra_regressors:
        if regressor_name in ["water", "green"]:
            if regressor_name in ["water"]:
                img = load_background(screenshot_period)
                regressor = FEConvolutionFixedPixels(name="water", mask=np.all(img == WaterColor, axis=-1),
                                                     x_coords=kwargs["longitudes"], normalize=False,
                                                     y_coords=kwargs["latitudes"], agg_func=np.sum,
                                                     kernel=gaussker, sigma=0.1).extract_features(times,
                                                                                                  positions)[:, :,
                            np.newaxis]  # * np.mean(traffic, axis=-1, keepdims=True)
            elif regressor_name in ["green"]:
                img = load_background(screenshot_period)
                regressor = FEConvolutionFixedPixels(name="green", mask=np.all(img == GreenAreaColor, axis=-1),
                                                     x_coords=kwargs["longitudes"], normalize=False,
                                                     y_coords=kwargs["latitudes"], agg_func=np.sum,
                                                     kernel=gaussker, sigma=0.1).extract_features(times,
                                                                                                  positions)[:, :,
                            np.newaxis]  # * np.mean(traffic, axis=-1, keepdims=True)

            else:
                continue

        elif regressor_name in ["hours", "week"]:
            if regressor_name in ["hours"]:
                t = np.pi * 2 * times.hour / 24
                regressor = PolynomialFeatures(degree=2).fit_transform(np.transpose([np.cos(t), np.sin(t)]))
                regressor = np.transpose([regressor] * np.shape(positions)[-1], axes=(1, 0, 2))
            elif regressor_name in ["week"]:
                weekend_days = 1 * (times.dayofweek > 4)
                labour_days = 1 * (times.dayofweek <= 4)
                regressor = np.transpose([weekend_days, labour_days])
                regressor = np.transpose([regressor] * np.shape(positions)[-1], axes=(1, 0, 2))
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

    X = np.concatenate(X, axis=-1)
    X = X.reshape((-1, np.shape(X)[-1]))
    # print(np.shape(X))
    return X


class BaseSourceModel(BaseModel):

    def __init__(self, path4preprocess: Union[str, Path], graph, spacial_locations: pd.DataFrame, times,
                 traffic_by_edge, extra_regressors=[],
                 name="", loss=mse, optim_method=GRAD,
                 verbose=False, redo_preprocessing=False, cv_in_space=True,
                 source_model=LassoCV(selection="random", positive=False),
                 substract_mean=True, lnei=1,
                 niter=1000, sigma0=1, **kwargs):
        # super call
        super().__init__(name=f"{name}_{substract_mean}_{source_model}",
                         loss=loss, optim_method=optim_method, verbose=verbose, niter=niter, cv_in_space=cv_in_space,
                         sigma0=sigma0, **kwargs)
        self.path4preprocess = path4preprocess
        self.extra_regressors = extra_regressors
        self.source_model = source_model
        self.substract_mean = substract_mean

        graph = nx.Graph(graph).to_undirected()
        if len(nx.get_edge_attributes(graph, "length")) != graph.number_of_edges():
            raise Exception("Each edge in Graph should have edge attribute 'length'")
        if len(nx.get_edge_attributes(graph, "lanes")) != graph.number_of_edges():
            raise Exception("Each edge in Graph should have edge attribute 'lanes'")

        self.node_positions = get_graph_node_positions(graph)
        self.position2node_index = get_nearest_node_mapping(path=path4preprocess, filename=f"pos2node",
                                                            target_positions=spacial_locations,
                                                            graph=graph, recalculate=redo_preprocessing)
        self.times = times
        # shape [#times, #nodes, #traffic colors]
        self.source = get_traffic_by_node_conv(path=path4preprocess, times=times,
                                               traffic_by_edge=traffic_by_edge,
                                               graph=graph, recalculate=redo_preprocessing,
                                               lnei=lnei)

    def get_space_indexes(self, positions):
        return [self.position2node_index[tuple(tp)] for tp in
                (positions.values if isinstance(positions, pd.DataFrame) else positions).T]

    def get_time_indexes(self, observed_pollution):
        return [self.times.index(t) for t in observed_pollution.index]

    def get_source(self, positions, observed_pollution: pd.DataFrame, **kwargs):
        spatial_indexes = self.get_space_indexes(positions)
        times_indexes = self.get_time_indexes(observed_pollution)

        # [#times, #nodes, #traffic colors]
        source = self.source[times_indexes, :, :]
        source[np.isnan(source)] = 0

        # average in space
        avg = np.mean(observed_pollution.values, axis=1, keepdims=True) if self.substract_mean else 0
        return source, avg, spatial_indexes, times_indexes

    def add_extra_regressors_and_reshape(self, source, positions, observed_pollution, **kwargs):
        source = source.reshape((-1, 4))
        if len(self.extra_regressors) > 1:
            extra_data = extra_regressors(observed_pollution.index, positions, self.extra_regressors, **kwargs)
            source = np.concatenate([source, extra_data], axis=1)
        return source


class NodeSourceModel(BaseSourceModel):
    def calibrate(self, observed_stations, observed_pollution: pd.DataFrame, traffic, **kwargs):
        source, avg, spatial_indexes, _ = self.get_source(observed_stations, observed_pollution, **kwargs)
        source = source[:, spatial_indexes, :]
        source = self.add_extra_regressors_and_reshape(source, observed_stations, observed_pollution, **kwargs)
        self.source_model.fit(source, (observed_pollution.values - avg).ravel())

    def state_estimation(self, observed_stations, observed_pollution, traffic, target_positions: pd.DataFrame,
                         **kwargs) -> np.ndarray:
        source, avg, spatial_indexes, times_indexes = self.get_source(target_positions, observed_pollution, **kwargs)
        source = source[:, spatial_indexes, :]
        source = self.add_extra_regressors_and_reshape(source, target_positions, observed_pollution, **kwargs)
        u = self.source_model.predict(source).reshape((len(times_indexes), -1)) + avg

        # final solution
        return u


class ProjectionSourceModel(BaseSourceModel):

    def __init__(self, path4preprocess: Union[str, Path], graph, spacial_locations: pd.DataFrame, times,
                 traffic_by_edge, extra_regressors=[], k_max=None, k=None, std_normalize=False, mean_normalize=True,
                 name="", loss=mse, optim_method=GRAD, cv_in_space=True,
                 verbose=False, redo_preprocessing=False,
                 source_model=LassoCV(selection="random", positive=False),
                 substract_mean=True, lnei=1,
                 niter=1000, sigma0=1, **kwargs):
        # super call
        super().__init__(name=f"{name}_{substract_mean}_{source_model}", path4preprocess=path4preprocess,
                         loss=loss, optim_method=optim_method, verbose=verbose,
                         niter=niter, sigma0=sigma0, cv_in_space=cv_in_space,
                         graph=graph, spacial_locations=spacial_locations,
                         times=times, traffic_by_edge=traffic_by_edge, extra_regressors=extra_regressors,
                         redo_preprocessing=redo_preprocessing, source_model=source_model,
                         substract_mean=substract_mean, lnei=lnei, **kwargs)
        self.k_max = k_max if k is None else k
        self.k = k
        self.pca = None
        self.mse = []
        self.source_mean = None
        self.source_std = None
        self.mean_normalize = mean_normalize
        self.std_normalize = std_normalize  # it works slightly works adding the std

    def init_Vn_space(self, source):
        raise Exception("Not implemented.")
        # self.pca = [PCA(n_components=self.k_max).fit(source[:, :, i]) for i in range(np.shape(source)[-1])]

    def project_to_subspace(self, source, k):
        raise Exception("Not implemented.")

    def fit_source_model_for_a_given_k(self, k, source, spatial_indexes, observed_stations, observed_pollution, avg,
                                       **kwargs):
        s = self.project_to_subspace(source, k)
        s = s[:, spatial_indexes, :] * self.source_std[:, spatial_indexes, :] + self.source_mean[:, spatial_indexes, :]
        s = self.add_extra_regressors_and_reshape(s, observed_stations, observed_pollution, **kwargs)
        self.source_model.fit(s, (observed_pollution.values - avg).ravel())
        return s

    def calibrate(self, observed_stations, observed_pollution: pd.DataFrame, traffic, **kwargs):
        source, avg, spatial_indexes, times_indexes = self.get_source(observed_stations, observed_pollution, **kwargs)
        # source dimensions: [#times, #nodes, #traffic colors]
        self.source_mean = np.mean(source, axis=0, keepdims=True)
        if not self.mean_normalize:
            self.source_mean *= 0
        # it works slightly works adding the std
        if self.std_normalize:
            self.source_std = np.std(source, axis=0, keepdims=True)
            self.source_std[self.source_std == 0] = 1
        else:
            self.source_std = np.ones_like(self.source_mean)
        source -= self.source_mean
        source /= self.source_std
        self.init_Vn_space(source)

        if self.k is None:
            self.mse = []
            for k in tqdm(range(1, self.k_max), desc="Finding optimal number of PCA components."):
                s = self.fit_source_model_for_a_given_k(k, source, spatial_indexes, observed_stations,
                                                        observed_pollution,
                                                        avg, **kwargs)
                # calculate error to optimize k
                u = self.source_model.predict(s).reshape((len(times_indexes), -1)) + avg
                self.mse.append(np.mean((u - observed_pollution.values) ** 2))

            self.k = np.argmin(self.mse) + 1  # we start in k=1 to explore
            print("Best k", self.k)
            print(self.mse)
        self.fit_source_model_for_a_given_k(self.k, source, spatial_indexes, observed_stations, observed_pollution, avg,
                                            **kwargs)

    def state_estimation(self, observed_stations, observed_pollution, traffic, target_positions: pd.DataFrame,
                         **kwargs) -> np.ndarray:
        source, avg, spatial_indexes, times_indexes = self.get_source(target_positions, observed_pollution, **kwargs)
        source = self.project_to_subspace((source - self.source_mean) / self.source_std, self.k)
        source = (source[:, spatial_indexes, :] * self.source_std[:, spatial_indexes, :] +
                  self.source_mean[:, spatial_indexes, :])
        source = self.add_extra_regressors_and_reshape(source, target_positions, observed_pollution, **kwargs)
        u = self.source_model.predict(source).reshape((len(times_indexes), -1)) + avg

        # final solution
        return u


class PCASourceModel(ProjectionSourceModel):
    def __init__(self, path4preprocess: Union[str, Path], graph, spacial_locations: pd.DataFrame, times,
                 traffic_by_edge, extra_regressors=[], k_max=None, k=None, std_normalize=False, mean_normalize=True,
                 name="", loss=mse, optim_method=GRAD, cv_in_space=True,
                 verbose=False, redo_preprocessing=False,
                 source_model=LassoCV(selection="random", positive=False),
                 substract_mean=True, lnei=1,
                 niter=1000, sigma0=1, **kwargs):
        # super call
        super().__init__(name=f"{name}_{substract_mean}_{source_model}", path4preprocess=path4preprocess,
                         loss=loss, optim_method=optim_method, verbose=verbose,
                         niter=niter, sigma0=sigma0, k_max=k_max, k=k, cv_in_space=cv_in_space,
                         mean_normalize=mean_normalize, std_normalize=std_normalize,
                         graph=graph, spacial_locations=spacial_locations,
                         times=times, traffic_by_edge=traffic_by_edge, extra_regressors=extra_regressors,
                         redo_preprocessing=redo_preprocessing, source_model=source_model,
                         substract_mean=substract_mean, lnei=lnei, **kwargs)
        self.pca = None

    def init_Vn_space(self, source):
        self.pca = [PCA(n_components=self.k_max).fit(source[:, :, i]) for i in range(np.shape(source)[-1])]

    def project_to_subspace(self, source, k):
        s = np.zeros_like(source)
        for i in range(np.shape(source)[-1]):
            s[:, :, i] = self.pca[i].transform(source[:, :, i])[:, :k] @ self.pca[i].components_[:k, :]
            # s[:, :, i] = self.pca[i].inverse_transform(self.pca[i].transform(source[:, :, i]))
        return s


class LaplacianSourceModel(ProjectionSourceModel):
    def __init__(self, path4preprocess: Union[str, Path], graph, spacial_locations: pd.DataFrame, times,
                 traffic_by_edge, extra_regressors=[], k_max=None, k=None, std_normalize=False, mean_normalize=True,
                 name="", loss=mse, optim_method=GRAD,
                 verbose=False, redo_preprocessing=False,
                 source_model=LassoCV(selection="random", positive=False),
                 substract_mean=True, lnei=1,
                 niter=1000, sigma0=1, **kwargs):
        # super call
        super().__init__(name=f"{name}_{substract_mean}_{source_model}", path4preprocess=path4preprocess,
                         loss=loss, optim_method=optim_method, verbose=verbose,
                         niter=niter, sigma0=sigma0, k_max=k_max, k=k,
                         mean_normalize=mean_normalize, std_normalize=std_normalize,
                         graph=graph, spacial_locations=spacial_locations,
                         times=times, traffic_by_edge=traffic_by_edge, extra_regressors=extra_regressors,
                         redo_preprocessing=redo_preprocessing, source_model=source_model,
                         substract_mean=substract_mean, lnei=lnei, **kwargs)
        graph = nx.Graph(graph).to_undirected()
        if len(nx.get_edge_attributes(graph, "length")) != graph.number_of_edges():
            raise Exception("Each edge in Graph should have edge attribute 'length'")
        if len(nx.get_edge_attributes(graph, "lanes")) != graph.number_of_edges():
            raise Exception("Each edge in Graph should have edge attribute 'lanes'")

        Kd = get_diffusion_matrix(path=path4preprocess, filename=f"diffusion_matrix", graph=graph,
                                  recalculate=redo_preprocessing)
        Ms = get_absorption_matrix(path=path4preprocess, filename=f"absorption_matrix", graph=graph,
                                   recalculate=redo_preprocessing)
        # [# nodes, k]
        self.B, _ = get_geometric_basis(path=path4preprocess, filename=f"basis_k{k_max}", Kd=Kd, Ms=Ms, k=k_max,
                                        recalculate=redo_preprocessing)

    def project_to_subspace(self, source, k):
        return np.einsum("tkc,dk->tdc", np.einsum("tdc,dk->tkc", source, self.B[:, :k]), self.B[:, :k])

    def init_Vn_space(self, source):
        pass  # it is the self.B of init method.


def calibrate(model: BaseModel, observed_stations, observed_pollution: pd.DataFrame, traffic, **kwargs):
    if len(model.params) == 1 and model.optim_method == CMA:
        model.optim_method = GRAD

    def optim_func(params):
        dictparams = dict(zip(model.bounds, params))
        if model.verbose:
            df = pd.DataFrame(np.reshape(list(dictparams.values()), (1, -1)), columns=list(dictparams.keys()))
            print(f"Params for optimization: \n {df}")

        model.set_params(**dictparams)

        if model.cv_in_space:
            target_pollution, predicted_pollution = \
                list(zip(*[(target_pollution,
                            model.state_estimation(**known_data, **kwargs))
                           for known_data, target_pollution in
                           loo(observed_stations, observed_pollution, traffic, kwargs.get("stations2test", None))]))
            predicted_pollution = np.concatenate(predicted_pollution, axis=0)
            target_pollution = np.concatenate(target_pollution, axis=0)[:, np.newaxis]
        else:
            target_pollution = observed_pollution.values.ravel()
            predicted_pollution = model.state_estimation(observed_stations=observed_stations,
                                                         observed_pollution=observed_pollution,
                                                         target_positions=observed_stations,
                                                         traffic=traffic, **kwargs).ravel()

        loss = model.loss(predicted_pollution, target_pollution)
        model.losses[tuple(params)] = loss
        if model.verbose:
            print(f"loss: {loss}")
        return loss

    model.losses = dict()
    x0 = np.array([model.params[k] for k in model.bounds])
    if model.niter == 1 or len(model.bounds) == 0:
        optim_func(x0)
        optim_params = x0
    elif model.optim_method == CMA:
        optim_params, _ = cma.fmin2(objective_function=optim_func, x0=x0,
                                    sigma0=model.sigma0, eval_initial_x=True,
                                    options={'popsize': 10, 'maxfevals': model.niter})
    elif model.optim_method == GRAD:
        optim_params = minimize(fun=optim_func, x0=x0, bounds=model.bounds.values(),
                                method="L-BFGS-B", options={'maxiter': model.niter}).x
    elif model.optim_method == BAYES:
        optim_params = gp_minimize(optim_func, dimensions=list(model.bounds.values()),
                                   x0=x0.tolist(), n_calls=model.niter).x
    elif model.optim_method in [RANDOM, UNIFORM, LOGUNIFORM]:
        if model.optim_method == UNIFORM:
            sampler = np.linspace
        elif model.optim_method == RANDOM:
            sampler = np.random.uniform
        elif model.optim_method == LOGUNIFORM:
            sampler = lambda d, u, i: np.logspace(np.log10(d), np.log10(u), i)
        else:
            raise Exception(f"Optim method {model.optim_method} not implemented.")
        samples = {k: sampler(bounds.lower, bounds.upper, model.niter).ravel().tolist() for k, bounds in
                   model.bounds.items() if
                   bounds is not None}
        model.losses = {x: optim_func(list({**model.params, **dict(zip(samples.keys(), x))}.values())) for x in
                        tqdm(zip(*samples.values()), desc=f"Training {model}")}
        best_ix = np.argmin(model.losses.values())
        optim_params = [v[best_ix] for v in samples.values()]
    elif model.optim_method == NONE_OPTIM_METHOD:
        return None
    else:
        raise Exception("Not implemented.")

    if len(model.bounds) > 0:
        model.set_params(**dict(zip(model.bounds.keys(), optim_params)))
        model.losses = pd.Series(model.losses.values(), pd.Index(model.losses.keys(), names=model.bounds.keys()),
                                 name="loss")
    print(model, "Optim params: ", model.params)

    model.calibrated = True
    return model


class PhysicsModel(PCASourceModel):
    def __init__(self, path4preprocess: Union[str, Path], graph, spacial_locations: pd.DataFrame, times,
                 traffic_by_edge, extra_regressors=[], k_max=None, k=None, rb_k_max=None, rb_k=None,
                 std_normalize=False, mean_normalize=True, basis="graph_laplacian",
                 name="", loss=mse, optim_method=GRAD, cv_in_space=True,
                 verbose=False, redo_preprocessing=False,
                 source_model=LassoCV(selection="random", positive=False),
                 substract_mean=True, lnei=1,
                 niter=1000, sigma0=1, **kwargs):
        self.rb_k_max = rb_k_max
        self.path4preprocess = path4preprocess
        self.redo_preprocessing = redo_preprocessing
        self.basis = basis

        # super call
        super().__init__(name=f"{name}_{substract_mean}_{source_model}", path4preprocess=path4preprocess,
                         loss=loss, optim_method=optim_method, verbose=verbose,
                         niter=niter, sigma0=sigma0, k_max=k_max, k=k,
                         mean_normalize=mean_normalize, std_normalize=std_normalize,
                         graph=graph, spacial_locations=spacial_locations, cv_in_space=cv_in_space,
                         times=times, traffic_by_edge=traffic_by_edge, extra_regressors=extra_regressors,
                         redo_preprocessing=redo_preprocessing, source_model=source_model,
                         substract_mean=substract_mean, lnei=lnei, rb_k=rb_k, **kwargs)

    def project_source_to_rb(self, basis, source):
        return np.einsum("dk,td->tk", basis, source)

    def calibrate(self, observed_stations, observed_pollution: pd.DataFrame, traffic, **kwargs):
        PCASourceModel.calibrate(self, observed_stations=observed_stations, observed_pollution=observed_pollution,
                                 traffic=traffic, **kwargs)

        graph = nx.Graph(kwargs["graph"]).to_undirected()
        if len(nx.get_edge_attributes(graph, "length")) != graph.number_of_edges():
            raise Exception("Each edge in Graph should have edge attribute 'length'")
        if len(nx.get_edge_attributes(graph, "lanes")) != graph.number_of_edges():
            raise Exception("Each edge in Graph should have edge attribute 'lanes'")

        Kd = get_diffusion_matrix(path=self.path4preprocess, filename=f"diffusion_matrix", graph=graph,
                                  recalculate=self.redo_preprocessing)
        Ms = get_absorption_matrix(path=self.path4preprocess, filename=f"absorption_matrix", graph=graph,
                                   recalculate=self.redo_preprocessing)

        # average in space
        avg = observed_pollution.mean(axis=1).values if self.substract_mean else 0
        source = PCASourceModel.state_estimation(self, observed_stations=observed_stations,
                                                 observed_pollution=observed_pollution,
                                                 traffic=traffic, target_positions=self.node_positions.T, **kwargs)
        self.source_avg = np.mean(source, axis=0, keepdims=True)
        if self.basis == "graph_laplacian":
            # [# nodes, k]
            self.B, _ = get_geometric_basis(path=self.path4preprocess, filename=f"basis_k{self.rb_k_max}", Kd=Kd, Ms=Ms,
                                            k=self.rb_k_max,
                                            recalculate=self.redo_preprocessing)
        elif self.basis == "pca_source":
            # [# nodes, k]
            self.B = PCA(n_components=self.rb_k_max).fit(source - self.source_avg).components_.T
        else:
            raise Exception("Invalid basis")

        self.KdROM, self.MsROM = get_reduced_matrices(path=self.path4preprocess, save=False,
                                                      filename=f"reduced_matrices_k{self.rb_k_max}",
                                                      Kd=Kd,
                                                      Ms=Ms, basis=self.B, recalculate=self.redo_preprocessing)

        model = calibrate(self, observed_stations=observed_stations, observed_pollution=observed_pollution,
                          traffic=traffic, source=source, avg=avg, **kwargs)
        self.set_params(**model.params)
        # super().calibrate(observed_stations, observed_pollution, traffic, **kwargs)

    def state_estimation(self, observed_stations, observed_pollution, traffic, target_positions: pd.DataFrame,
                         **kwargs) -> np.ndarray:
        spatial_unknown_indexes = self.get_space_indexes(target_positions)
        stations_indexes = self.get_space_indexes(observed_stations)
        rb_k = int(self.params["rb_k"])

        # average in space
        avg = observed_pollution.mean(axis=1).values if self.substract_mean else 0

        # source model predictions
        if "source" in kwargs and "avg" in kwargs:  # when calibarting do not re calculate each time.
            source = kwargs["source"]
            avg = kwargs["avg"]
        else:
            source = PCASourceModel.state_estimation(self, observed_stations=observed_stations,
                                                     observed_pollution=observed_pollution,
                                                     traffic=traffic, target_positions=self.node_positions.T, **kwargs)
        # - avg[:, np.newaxis]
        # source -= self.source_avg
        source_k = self.project_source_to_rb(self.B[:, :rb_k], source).T

        # diffusion equation
        Ms_param = self.params["absorption"]
        Kd_param = self.params["diffusion"]
        delta = self.params["delta"]
        alpha = self.params["alpha"]
        # delta = 1
        # alpha = 1
        # Ms_param = 10
        # Kd_param = 10

        At = Kd_param * self.KdROM[:rb_k, :rb_k] + Ms_param * self.MsROM[:rb_k, :rb_k]

        # inverse problem term
        Bz = self.B[stations_indexes, :rb_k].T @ (observed_pollution.T.values - avg)  # learn the correction
        Az = self.B[stations_indexes, :rb_k].T @ self.B[stations_indexes, :rb_k]

        # solving the joint equation
        c = np.linalg.solve((alpha * At + (1 - alpha) * Az), alpha * source_k + (1 - alpha) * Bz)

        # final solution weighted
        u = (delta * np.einsum("dk,kt->td", self.B[spatial_unknown_indexes, :rb_k], c)  # + avg[:, np.newaxis]
             + source[:, spatial_unknown_indexes])

        # u = (delta * (np.einsum("dk,kt->td", self.B[:, :rb_k], c) + avg[:, np.newaxis])
        #      + (1 - delta) * source[:, :])
        # rel = np.abs((u-source)/source*100)
        # rmean = np.mean(rel)
        # rmax = np.max(rel)
        # rmin = np.min(rel)
        # mse=np.mean((u[:, stations_indexes]-observed_pollution.values)**2)
        # msesource=np.mean((source[:, stations_indexes]-observed_pollution.values)**2)
        return u


class ModelsAggregator(BaseModel):
    def __init__(self, models: List[BaseModel], aggregator: Pipeline, name=None):
        super().__init__()
        self.losses = list()
        self.name = name
        self.TRUE_MODEL = np.all([model.TRUE_MODEL for model in models])
        self.aggregator = aggregator
        self.models = models

    @property
    def weights(self):
        if hasattr(self.aggregator.steps[-1], "coef_") and hasattr(self.aggregator.steps[-1], "intercept_"):
            return np.ravel(np.append(self.aggregator.steps[-1].coef_, self.aggregator.steps[-1].intercept_))

    @property
    def model_importance(self):
        weights = self.weights
        if weights is not None:
            return {model_name: weight for model_name, weight in
                    zip(list(map(str, self.models)) + ["intercept"], np.abs(self.weights) / np.abs(self.weights).sum())}

    def __str__(self):
        models_names = ','.join([''.join(filter(lambda c: c.isupper(), str(model))) for model in
                                 self.models]) if self.name is None else self.name
        return f"{self.aggregator}({models_names})"

    def state_estimation_for_each_model(self, observed_stations, observed_pollution, traffic, target_positions,
                                        **kwargs) -> [np.ndarray,
                                                      np.ndarray]:
        return np.transpose([
            model.state_estimation(observed_stations, observed_pollution, traffic, target_positions, **kwargs)
            for model in self.models], axes=(1, 2, 0))

    def state_estimation(self, observed_stations, observed_pollution, traffic, target_positions: pd.DataFrame,
                         **kwargs) -> np.ndarray:
        individual_predictions = self.state_estimation_for_each_model(observed_stations, observed_pollution, traffic,
                                                                      target_positions, **kwargs)
        shape = np.shape(individual_predictions)
        return self.aggregator.predict(individual_predictions.reshape((-1, shape[-1]))).reshape((shape[0], shape[1]))

    def calibrate(self, observed_stations, observed_pollution: pd.DataFrame, traffic, **kwargs):
        if not self.calibrated:
            # calibrate models
            for model in self.models:
                if not model.calibrated:
                    model.calibrate(observed_stations, observed_pollution, traffic, **kwargs)
                self.losses.append(model.losses)
            # find optimal weights for model averaging
            individual_predictions = \
                self.state_estimation_for_each_model(observed_stations, observed_pollution, traffic, observed_stations,
                                                     **kwargs)
            shape = np.shape(individual_predictions)
            self.aggregator.fit(individual_predictions.reshape((-1, shape[-1])),
                                observed_pollution.values.reshape((-1, 1)))
            self.calibrated = True
        print(f"Models importance: {self.model_importance}")
