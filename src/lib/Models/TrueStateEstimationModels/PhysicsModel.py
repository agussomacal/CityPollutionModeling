from pathlib import Path
from typing import Union

import networkx as nx
import numpy as np
import pandas as pd
from scipy.linalg import eigh
from scipy.sparse.linalg import gmres
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.pipeline import Pipeline
from tqdm import tqdm

from PerplexityLab.miscellaneous import filter_dict, timeit, if_exist_load_else_do
from src.experiments.config_experiments import screenshot_period
from src.lib.DataProcessing.TrafficProcessing import TRAFFIC_VALUES, load_background
from src.lib.FeatureExtractors.ConvolutionFeatureExtractors import FEConvolutionFixedPixels, WaterColor, GreenAreaColor
from src.lib.FeatureExtractors.GraphFeatureExtractors import get_graph_node_positions, compute_adjacency, \
    compute_laplacian_matrix_from_graph
from src.lib.Models.BaseModel import BaseModel, mse, GRAD, pollution_agnostic
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
    return np.einsum("nk,tnc->tck", basis, traffic_by_node)


def extra_regressors(times, positions, extra_regressors, **kwargs):
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
                 verbose=False, redo_preprocessing=False,
                 source_model=LassoCV(selection="random", positive=False),
                 substract_mean=True, lnei=1,
                 niter=1000, sigma0=1, **kwargs):
        # super call
        super().__init__(name=f"{name}_{substract_mean}_{source_model}",
                         loss=loss, optim_method=optim_method, verbose=verbose, niter=niter,
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

        self.position2node_index = get_nearest_node_mapping(path=path4preprocess, filename=f"pos2node",
                                                            target_positions=spacial_locations,
                                                            graph=graph, recalculate=redo_preprocessing)
        self.times = times
        # shape [#times, #nodes, #traffic colors]
        self.source = get_traffic_by_node_conv(path=path4preprocess, times=times,
                                               traffic_by_edge=traffic_by_edge,
                                               graph=graph, recalculate=redo_preprocessing,
                                               lnei=lnei)

    def get_source(self, positions, observed_pollution: pd.DataFrame, **kwargs):
        spatial_indexes = [self.position2node_index[tuple(tp)] for tp in positions.values.T]
        times_indexes = [self.times.index(t) for t in observed_pollution.index]

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


class PCASourceModel(BaseSourceModel):

    def __init__(self, path4preprocess: Union[str, Path], graph, spacial_locations: pd.DataFrame, times,
                 traffic_by_edge, extra_regressors=[], k_max=None, k=None, std_normalize=False,
                 name="", loss=mse, optim_method=GRAD,
                 verbose=False, redo_preprocessing=False,
                 source_model=LassoCV(selection="random", positive=False),
                 substract_mean=True, lnei=1,
                 niter=1000, sigma0=1, **kwargs):
        # super call
        super().__init__(name=f"{name}_{substract_mean}_{source_model}", path4preprocess=path4preprocess,
                         loss=loss, optim_method=optim_method, verbose=verbose,
                         niter=niter, sigma0=sigma0,
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
        self.std_normalize = std_normalize  # it works slightly works adding the std

    def project_to_pca_subspace(self, source, k):
        s = np.zeros_like(source)
        for i in range(np.shape(source)[-1]):
            s[:, :, i] = self.pca[i].transform(source[:, :, i])[:, :k] @ self.pca[i].components_[:k, :]
            # s[:, :, i] = self.pca[i].inverse_transform(self.pca[i].transform(source[:, :, i]))
        return s

    def fit_for_a_given_k(self, k, source, spatial_indexes, observed_stations, observed_pollution, avg,
                          **kwargs):
        s = self.project_to_pca_subspace(source, k)
        s = s[:, spatial_indexes, :] * self.source_std[:, spatial_indexes, :] + self.source_mean[:, spatial_indexes, :]
        s = self.add_extra_regressors_and_reshape(s, observed_stations, observed_pollution, **kwargs)
        self.source_model.fit(s, (observed_pollution.values - avg).ravel())
        return s

    def calibrate(self, observed_stations, observed_pollution: pd.DataFrame, traffic, **kwargs):
        source, avg, spatial_indexes, times_indexes = self.get_source(observed_stations, observed_pollution, **kwargs)
        # source dimensions: [#times, #nodes, #traffic colors]
        self.source_mean = np.mean(source, axis=0, keepdims=True)
        # it works slightly works adding the std
        if self.std_normalize:
            self.source_std = np.std(source, axis=0, keepdims=True)
            self.source_std[self.source_std == 0] = 1
        else:
            self.source_std = np.ones_like(self.source_mean)
        source -= self.source_mean
        source /= self.source_std
        self.pca = [PCA(n_components=self.k_max).fit(source[:, :, i]) for i in range(np.shape(source)[-1])]

        if self.k is None:
            self.mse = []
            for k in tqdm(range(1, self.k_max), desc="Finding optimal number of PCA components."):
                s = self.fit_for_a_given_k(k, source, spatial_indexes, observed_stations, observed_pollution,
                                           avg, **kwargs)
                # calculate error to optimize k
                u = self.source_model.predict(s).reshape((len(times_indexes), -1)) + avg
                self.mse.append(np.mean((u - observed_pollution.values) ** 2))

            self.k = np.argmin(self.mse) + 1  # we start in k=1 to explore
            print("Best k", self.k)
            print(self.mse)
        self.fit_for_a_given_k(self.k, source, spatial_indexes, observed_stations, observed_pollution, avg, **kwargs)

    def state_estimation(self, observed_stations, observed_pollution, traffic, target_positions: pd.DataFrame,
                         **kwargs) -> np.ndarray:
        source, avg, spatial_indexes, times_indexes = self.get_source(target_positions, observed_pollution, **kwargs)
        source = self.project_to_pca_subspace((source - self.source_mean) / self.source_std, self.k)
        source = (source[:, spatial_indexes, :] * self.source_std[:, spatial_indexes, :] +
                  self.source_mean[:, spatial_indexes, :])
        source = self.add_extra_regressors_and_reshape(source, target_positions, observed_pollution, **kwargs)
        u = self.source_model.predict(source).reshape((len(times_indexes), -1)) + avg

        # final solution
        return u


class LaplacianSourceModel(BaseSourceModel):
    def __init__(self, path4preprocess: Union[str, Path], graph, spacial_locations: pd.DataFrame, times,
                 traffic_by_edge, extra_regressors=[], k_max=None, k=None, std_normalize=False, mean_normalize=False,
                 name="", loss=mse, optim_method=GRAD,
                 verbose=False, redo_preprocessing=False,
                 source_model=LassoCV(selection="random", positive=False),
                 substract_mean=True, lnei=1,
                 niter=1000, sigma0=1, **kwargs):
        # super call
        super().__init__(name=f"{name}_{substract_mean}_{source_model}", path4preprocess=path4preprocess,
                         loss=loss, optim_method=optim_method, verbose=verbose,
                         niter=niter, sigma0=sigma0,
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
        return np.einsum("tnc,nk,dk->tdc", source, self.B, self.B)

    def fit_for_a_given_k(self, k, source, spatial_indexes, observed_stations, observed_pollution, avg,
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

        if self.k is None:
            self.mse = []
            for k in tqdm(range(1, self.k_max), desc="Finding optimal number of PCA components."):
                s = self.fit_for_a_given_k(k, source, spatial_indexes, observed_stations, observed_pollution,
                                           avg, **kwargs)
                # calculate error to optimize k
                u = self.source_model.predict(s).reshape((len(times_indexes), -1)) + avg
                self.mse.append(np.mean((u - observed_pollution.values) ** 2))

            self.k = np.argmin(self.mse) + 1  # we start in k=1 to explore
            print("Best k", self.k)
            print(self.mse)
        self.fit_for_a_given_k(self.k, source, spatial_indexes, observed_stations, observed_pollution, avg, **kwargs)

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


class PhysicsModel(BaseModel):

    def __init__(self, path4preprocess: Union[str, Path], graph, spacial_locations: pd.DataFrame, times,
                 traffic_by_edge, k_max=10, extra_regressors=[],
                 name="", loss=mse, optim_method=GRAD,
                 verbose=False, redo_preprocessing=False,
                 source_model=LassoCV(selection="random", positive=False), substract_mean=True, lnei=1,
                 niter=1000, sigma0=1, **kwargs):
        # super call
        super().__init__(name=f"{name}_k{k_max}", loss=loss, optim_method=optim_method, verbose=verbose, niter=niter,
                         sigma0=sigma0, **kwargs)
        self.path4preprocess = path4preprocess
        self.k_max = k_max  # number of basis elements.
        self.extra_regressors = extra_regressors
        self.source_model = source_model
        self.substract_mean = substract_mean

        graph = nx.Graph(graph).to_undirected()
        if len(nx.get_edge_attributes(graph, "length")) != graph.number_of_edges():
            raise Exception("Each edge in Graph should have edge attribute 'length'")
        if len(nx.get_edge_attributes(graph, "lanes")) != graph.number_of_edges():
            raise Exception("Each edge in Graph should have edge attribute 'lanes'")

        Kd = get_diffusion_matrix(path=path4preprocess, filename=f"diffusion_matrix", graph=graph,
                                  recalculate=redo_preprocessing)
        Ms = get_absorption_matrix(path=path4preprocess, filename=f"absorption_matrix", graph=graph,
                                   recalculate=redo_preprocessing)
        self.B, _ = get_geometric_basis(path=path4preprocess, filename=f"basis_k{k_max}", Kd=Kd, Ms=Ms, k=k_max,
                                        recalculate=redo_preprocessing)
        self.KdROM, self.MsROM = get_reduced_matrices(path=path4preprocess, filename=f"reduced_matrices_k{k_max}",
                                                      Kd=Kd,
                                                      Ms=Ms, basis=self.B, recalculate=redo_preprocessing)

        self.spacial_locations = spacial_locations
        self.position2node_index = get_nearest_node_mapping(path=path4preprocess, filename=f"pos2node",
                                                            target_positions=spacial_locations,
                                                            graph=graph, recalculate=redo_preprocessing)
        self.wtB = get_basis_point_evaluations(path=path4preprocess, basis=self.B, target_positions=spacial_locations,
                                               position2node_index=self.position2node_index,
                                               recalculate=redo_preprocessing, filename=f"basis_point_eval_k{k_max}", )
        self.times = times
        # shape [#times, #nodes, #traffic colors]
        traffic_by_node = get_traffic_by_node_conv(path=path4preprocess, times=times,
                                                   traffic_by_edge=traffic_by_edge,
                                                   graph=graph, recalculate=redo_preprocessing,
                                                   lnei=lnei)

        # shape(source): times x color x k
        self.source = get_basis_traffic_by_node(path=path4preprocess, basis=self.B, traffic_by_node=traffic_by_node,
                                                recalculate=redo_preprocessing)

    def state_estimation(self, observed_stations, observed_pollution, traffic, target_positions: pd.DataFrame,
                         **kwargs) -> np.ndarray:

        k = min((max((1, int(self.params["k"]))), self.k_max))

        # average in space
        avg = observed_pollution.mean(axis=1).values if self.substract_mean else 0

        # traffic term
        times_indexes = [self.times.index(t) for t in observed_pollution.index]
        source = self.source[times_indexes, :, :k]
        tck = np.shape(source)
        source = self.source_model.predict(
            np.swapaxes(source, 1, 2).reshape((-1, tck[1]))).reshape((tck[0], tck[-1])).T

        # S_params = np.array(list(filter_dict(TRAFFIC_VALUES, self.params).values()))
        # source = np.einsum("tck,c->kt", self.source[times_indexes, :, :], S_params)

        # extra_data = extra_regressors(observed_pollution.index, target_positions, self.extra_regressors, **kwargs)
        # SED_params = np.array(list(filter_dict(self.extra_regressors, self.params).values()))
        # source += np.einsum("te,e", extra_data, SED_params)
        # SED_params = 0
        # source += self.params["intercept"]

        Ms_param = self.params["absorption"]
        Kd_param = self.params["diffusion"]
        At = (Kd_param * self.KdROM + Ms_param * self.MsROM)[:k, :k]

        # inverse problem term
        stations_indexes = [self.spacial_locations.columns.tolist().index(c) for c in observed_pollution.columns]
        Bz = self.wtB[stations_indexes, :k].T @ (observed_pollution.T.values - avg)  # learn the correction
        Az = self.wtB[stations_indexes, :k].T @ self.wtB[stations_indexes, :k]

        # solving the joint equation
        alpha = self.params["alpha"]
        c = np.linalg.solve(
            (alpha * At + (1 - alpha) * Az)[:k, :k],
            (alpha * source + (1 - alpha) * Bz)[:k, :]
        )

        # final solution
        indexes = [self.position2node_index[tuple(tp)] for tp in target_positions.values.T]
        u = np.einsum("ik,kt->it", self.B[:, :k], c) + avg
        um = u.mean() / source.mean()
        return u[indexes, :].T

    def calibrate(self, observed_stations, observed_pollution: pd.DataFrame, traffic, **kwargs):
        known_stations_indexes = [self.position2node_index[tuple(tp)] for tp in observed_stations.values.T]
        k = min((max((1, int(self.params["k"]))), self.k_max))

        # average in space
        avg = observed_pollution.mean(axis=1).values if self.substract_mean else 0

        # source term fitting
        times_indexes = [self.times.index(t) for t in observed_pollution.index]
        source = np.einsum("tck,lk->tlc", self.source[:, :, :k], self.B[:, :k])
        source = source[times_indexes, :, :]
        source = source[:, known_stations_indexes, :].reshape((-1, 4))

        if len(self.extra_regressors) > 1:
            extra_data = extra_regressors(observed_pollution.index, observed_stations, self.extra_regressors, **kwargs)
            source = np.concatenate([source, extra_data], axis=1)

        # To debug: almost all contributions are 0
        # (source[:, :, 0] / observed_pollution.T.values.T)[source[:, :, 0] / observed_pollution.T.values.T > 0]
        self.source_model.fit(source, (observed_pollution.T.values - avg).ravel())
        # print(self.source_model.coef_)
        # self.set_params(**dict(zip(list(TRAFFIC_VALUES.keys()) + self.extra_regressors, lr.coef_.ravel().tolist())))
        # self.set_params(intercept=self.source_model.intercept_)

        super().calibrate(observed_stations, observed_pollution, traffic, **kwargs)
