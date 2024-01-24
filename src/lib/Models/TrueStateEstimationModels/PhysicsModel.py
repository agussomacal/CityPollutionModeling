import copy
from pathlib import Path
from typing import Union, List

import networkx as nx
import numpy as np
import pandas as pd
from scipy.linalg import eigh
from scipy.optimize import minimize, LinearConstraint
from scipy.sparse import identity, diags
from scipy.sparse.linalg import spsolve
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.linear_model import LassoCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from tqdm import tqdm

from PerplexityLab.miscellaneous import if_exist_load_else_do
from src.experiments.config_experiments import screenshot_period
from src.lib.DataProcessing.TrafficProcessing import TRAFFIC_VALUES, load_background
from src.lib.FeatureExtractors.ConvolutionFeatureExtractors import FEConvolutionFixedPixels, WaterColor, GreenAreaColor
from src.lib.FeatureExtractors.GraphFeatureExtractors import get_graph_node_positions, compute_adjacency, \
    compute_laplacian_matrix_from_graph
from src.lib.Models.BaseModel import BaseModel, mse, GRAD, loo, NONE_OPTIM_METHOD, calibrate
from src.lib.Models.SensorDependentModels.BLUEFamily import BLUEModel
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
    graph = nx.Graph(graph).to_undirected()
    if len(nx.get_edge_attributes(graph, "length")) != graph.number_of_edges():
        raise Exception("Each edge in Graph should have edge attribute 'length'")
    if len(nx.get_edge_attributes(graph, "lanes")) != graph.number_of_edges():
        raise Exception("Each edge in Graph should have edge attribute 'lanes'")
    A = compute_adjacency(graph, edge_function=lambda data: data["lanes"] / data["length"])
    A.data[np.isnan(A.data)] = 0
    L = (A - diags(A.sum(axis=0))) @ diags(1 / A.sum(axis=0))
    return L
    # # diffusion matrix Kd
    # Kd = compute_laplacian_matrix_from_graph(graph, edge_function=lambda data: 1.0 / data["length"])
    # return Kd


@if_exist_load_else_do(file_format="joblib", loader=None, saver=None, description=None, check_hash=False)
def get_laplacian_matrix(graph: nx.Graph):
    graph = nx.Graph(graph).to_undirected()
    if len(nx.get_edge_attributes(graph, "length")) != graph.number_of_edges():
        raise Exception("Each edge in Graph should have edge attribute 'length'")
    if len(nx.get_edge_attributes(graph, "lanes")) != graph.number_of_edges():
        raise Exception("Each edge in Graph should have edge attribute 'lanes'")
    A = compute_adjacency(graph, edge_function=lambda data: data["lanes"] / data["length"])
    A.data[np.isnan(A.data)] = 0
    L = (A - diags(A.sum(axis=0))) @ diags(1 / A.sum(axis=0))
    return L


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
    X = X.reshape((-1, np.shape(X)[-1]), order="F")
    # print(np.shape(X))
    return X


def add_extra_regressors_and_reshape(extra_regressors_names, predictions, number_of_instances, positions,
                                     observed_pollution, order="F",
                                     **kwargs):
    predictions = predictions.reshape((-1, number_of_instances), order=order)
    if len(extra_regressors_names) > 1:
        extra_data = extra_regressors(observed_pollution.index, positions, extra_regressors_names, **kwargs)
        predictions = np.concatenate([predictions, extra_data], axis=1)
    return predictions


class BaseSourceModel(BaseModel):

    def __init__(self, path4preprocess: Union[str, Path], graph, spacial_locations: pd.DataFrame, times,
                 traffic_by_edge, extra_regressors=[], train_with_relative_error=False,
                 name="", loss=mse, optim_method=GRAD,
                 verbose=False, redo_preprocessing=False, cv_in_space=True,
                 source_model=LassoCV(selection="random", positive=False),
                 substract_mean=True, lnei=1,
                 niter=1000, sigma0=1, **kwargs):
        # super call
        super().__init__(name=f"{name}_{substract_mean}_{source_model}",
                         loss=loss, optim_method=optim_method, verbose=verbose, niter=niter, cv_in_space=cv_in_space,
                         sigma0=sigma0, **kwargs)
        self.train_with_relative_error = train_with_relative_error
        self.path4preprocess = path4preprocess
        self.redo_preprocessing = redo_preprocessing
        self.extra_regressors = extra_regressors
        self.source_model = source_model
        self.substract_mean = substract_mean

        graph = nx.Graph(graph).to_undirected()
        if len(nx.get_edge_attributes(graph, "length")) != graph.number_of_edges():
            raise Exception("Each edge in Graph should have edge attribute 'length'")
        if len(nx.get_edge_attributes(graph, "lanes")) != graph.number_of_edges():
            raise Exception("Each edge in Graph should have edge attribute 'lanes'")

        self.node_positions = get_graph_node_positions(graph)
        self.position2node_index = get_nearest_node_mapping(path=self.path4preprocess, filename=f"pos2node",
                                                            target_positions=spacial_locations,
                                                            graph=graph, recalculate=self.redo_preprocessing)
        self.times = times
        self.lnei = lnei

    def get_space_indexes(self, positions):
        return [self.position2node_index[tuple(tp)] for tp in
                (positions.values if isinstance(positions, pd.DataFrame) else positions).T]

    def get_time_indexes(self, observed_pollution):
        return [self.times.index(t) for t in observed_pollution.index]

    def get_source(self, positions, observed_pollution: pd.DataFrame, **kwargs):
        spatial_indexes = self.get_space_indexes(positions)
        times_indexes = self.get_time_indexes(observed_pollution)

        # [#times, #nodes, #traffic colors]
        source = get_traffic_by_node_conv(path=self.path4preprocess, times=self.times,
                                          traffic_by_edge=kwargs["traffic_by_edge"],
                                          graph=kwargs["graph"], recalculate=self.redo_preprocessing,
                                          lnei=self.lnei)
        source = source[times_indexes, :, :]
        source[np.isnan(source)] = 0

        # average in space
        avg = np.mean(observed_pollution.values, axis=1, keepdims=True) if self.substract_mean else 0
        return source, avg, spatial_indexes, times_indexes


class NodeSourceModel(BaseSourceModel):
    def calibrate(self, observed_stations, observed_pollution: pd.DataFrame, traffic, **kwargs):
        source, avg, spatial_indexes, _ = self.get_source(observed_stations, observed_pollution, **kwargs)
        source = source[:, spatial_indexes, :]
        source = add_extra_regressors_and_reshape(self.extra_regressors, source, 4, observed_stations,
                                                  observed_pollution, order="F",
                                                  **kwargs)
        self.source_model.fit(source, (observed_pollution.values - avg).ravel(order="F"),
                              sample_weight=1 / observed_pollution.values.ravel(
                                  order="F") if self.train_with_relative_error else None)

    def state_estimation(self, observed_stations, observed_pollution, traffic, target_positions: pd.DataFrame,
                         **kwargs) -> np.ndarray:
        source, avg, spatial_indexes, times_indexes = self.get_source(target_positions, observed_pollution, **kwargs)
        source = source[:, spatial_indexes, :]
        source = add_extra_regressors_and_reshape(self.extra_regressors, source, 4, target_positions,
                                                  observed_pollution, order="F",
                                                  **kwargs)
        u = self.source_model.predict(source).reshape((len(times_indexes), -1), order="F") + avg

        # final solution
        return u


class ProjectionFullSourceModel(BaseSourceModel):

    def __init__(self, path4preprocess: Union[str, Path], graph, spacial_locations: pd.DataFrame, times,
                 traffic_by_edge, kv: Union[int, Optim] = None, ky: Union[int, Optim] = None,
                 kr: Union[int, Optim] = None, kd: Union[int, Optim] = None,
                 extra_regressors=[], k_max=None, train_with_relative_error=False,
                 D0: Union[float, Optim] = 0.0, D1: Union[float, Optim] = 0.0,
                 D2: Union[float, Optim] = 0.0, D3: Union[float, Optim] = 0.0,
                 A0: Union[float, Optim] = 0.0, A1: Union[float, Optim] = 0.0,
                 A2: Union[float, Optim] = 0.0, A3: Union[float, Optim] = 0.0,
                 forward_weight0=1, source_weight0=0,
                 forward_weight1=1, source_weight1=0,
                 forward_weight2=1, source_weight2=0,
                 forward_weight3=1, source_weight3=0,
                 std_normalize=False, mean_normalize=True,
                 name="", loss=mse, optim_method=GRAD, cv_in_space=True, basis="pca",
                 verbose=False, redo_preprocessing=False,
                 source_model=LassoCV(selection="random", positive=False),
                 substract_mean=True, lnei=1,
                 niter=1000, sigma0=1, **kwargs):
        # super call
        super().__init__(name=f"{name}_{substract_mean}_{source_model}", path4preprocess=path4preprocess,
                         loss=loss, optim_method=optim_method, verbose=verbose,
                         train_with_relative_error=train_with_relative_error,
                         niter=niter, sigma0=sigma0, cv_in_space=cv_in_space,
                         graph=graph, spacial_locations=spacial_locations,
                         times=times, traffic_by_edge=traffic_by_edge, extra_regressors=extra_regressors,
                         redo_preprocessing=redo_preprocessing, source_model=source_model,
                         substract_mean=substract_mean, lnei=lnei,
                         kv=kv if isinstance(kv, int) else Optim(kv.start, 1, k_max),
                         ky=ky if isinstance(ky, int) else Optim(ky.start, 1, k_max),
                         kr=kr if isinstance(kr, int) else Optim(kr.start, 1, k_max),
                         kd=kd if isinstance(kd, int) else Optim(kd.start, 1, k_max),
                         D0=D0, D1=D1,
                         D2=D2, D3=D3,
                         A0=A0, A1=A1,
                         A2=A2, A3=A3,
                         forward_weight0=forward_weight0, source_weight0=source_weight0,
                         forward_weight1=forward_weight1, source_weight1=source_weight1,
                         forward_weight2=forward_weight2, source_weight2=source_weight2,
                         forward_weight3=forward_weight3, source_weight3=source_weight3,
                         **kwargs)
        self.k_max = k_max
        self.mse = []
        self.source_mean = None
        self.source_std = None
        self.mean_normalize = mean_normalize
        self.std_normalize = std_normalize  # it works slightly works adding the std
        self.basis = basis
        self.KdROM = [None] * 4
        self.MsROM = [None] * 4

    @staticmethod
    def orthonormalize_base(rb):
        """

        :param rb: [dim of space, number of elements k]
        :return: [dim of space, number of elements k]
        """
        q, r = np.linalg.qr(np.array(rb))
        return q

    def get_Vn_space(self):
        k = [self.params["kv"], self.params["ky"], self.params["kr"], self.params["kd"]]
        for i in range(4):
            pcabasis = self.pca[i].components_[:k[i], :].T
            gbasis = self.laplacian_basis[:k[i], :].T
            if self.basis == "pca":
                yield pcabasis
            elif self.basis == "geometrical":
                yield gbasis
            elif self.basis == "both":
                new_basis = np.concatenate([pcabasis, gbasis], axis=1)
                yield self.orthonormalize_base(new_basis)
            else:
                raise Exception("Not implemented.")

    def project_to_subspace(self, source, observed_stations, observed_pollution, avg):
        known_stations_idx = self.get_space_indexes(observed_stations)
        k = np.array([self.params["kv"], self.params["ky"], self.params["kr"], self.params["kd"]])
        if self.basis == "both":
            k *= 2

        s = np.zeros_like(source)
        for i, B in enumerate(self.get_Vn_space()):
            # B = self.pca[i].components_[:k[i], :].T
            # source_k = self.pca[i].transform(source[:, :, i])[:, :k[i]]
            source_k = source[:, :, i] @ B
            s[:, :, i] = source_k @ B.T

            # Forward problem
            Kd_param, Ms_param = self.params[f"D{i}"], self.params[f"A{i}"]
            At = Kd_param * self.KdROM[i][:k[i], :k[i]] + Ms_param * self.MsROM[i][:k[i], :k[i]]

            # Inverse problem
            Bz = B[known_stations_idx, :].T @ (observed_pollution.values - avg).T  # learn the correction
            Az = B[known_stations_idx, :].T @ B[known_stations_idx, :]

            # solving the joint equation
            alpha = self.params[f"forward_weight{i}"]
            try:
                c = np.linalg.solve((alpha * At + (1 - alpha) * Az), alpha * source_k.T + (1 - alpha) * Bz)
            except:
                # except np.linalg.LinAlgError:
                c = 0 * source_k.T

            # final solution weighted
            delta = self.params[f"source_weight{i}"]
            s[:, :, i] = (1 - delta) * np.einsum("dk,kt->td", B[:, :], c) + delta * s[:, :, i]
        return s

    def calibrate(self, observed_stations, observed_pollution: pd.DataFrame, traffic, **kwargs):
        source, avg, spatial_indexes, times_indexes = self.get_source(observed_stations, observed_pollution, **kwargs)
        # source dimensions: [#times, #nodes, #traffic colors]
        # self.source_mean = np.mean(source - avg[:, :, np.newaxis], axis=0, keepdims=True)
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
        # self.init_Vn_space(source)
        self.pca = [PCA(n_components=self.k_max).fit(source[:, :, i]) for i in range(np.shape(source)[-1])]

        Kd = get_diffusion_matrix(path=self.path4preprocess, filename=f"diffusion_matrix", graph=kwargs["graph"],
                                  recalculate=self.redo_preprocessing, verbose=False)
        Ms = get_absorption_matrix(path=self.path4preprocess, filename=f"absorption_matrix", graph=kwargs["graph"],
                                   recalculate=self.redo_preprocessing, verbose=False)

        self.laplacian_basis, _ = get_geometric_basis(path=self.path4preprocess,
                                                      filename=f"LaplacianModel_basis_k{self.k_max}",
                                                      Kd=Kd, Ms=Ms,
                                                      k=self.k_max,
                                                      recalculate=self.redo_preprocessing, verbose=False)
        self.laplacian_basis = self.laplacian_basis.T
        for i, B in enumerate(self.get_Vn_space()):
            self.KdROM[i], self.MsROM[i] = get_reduced_matrices(
                path=self.path4preprocess, save=False,
                filename=f"reduced_matrices_k{self.k_max * (1 + (self.basis == 'both'))}_color{i}",
                Kd=Kd,
                Ms=Ms, basis=B, recalculate=self.redo_preprocessing, verbose=False)

        model = calibrate(self, observed_stations=observed_stations, observed_pollution=observed_pollution,
                          traffic=traffic, training_mode=True,
                          # source=source,
                          # avg=avg,
                          **kwargs)
        self.set_params(**model.params)

    def state_estimation(self, observed_stations, observed_pollution, traffic, target_positions: pd.DataFrame,
                         **kwargs) -> np.ndarray:
        source, avg, spatial_indexes, times_indexes = self.get_source(target_positions, observed_pollution, **kwargs)
        # substract averages in space and time (normalize)
        # source = (source - avg[:, :, np.newaxis] - self.source_mean) / self.source_std
        source = (source - self.source_mean) / self.source_std
        source = self.project_to_subspace(source, observed_stations, observed_pollution, avg)  # project to subspace
        source = (source[:, spatial_indexes, :] * self.source_std[:, spatial_indexes, :] +
                  self.source_mean[:, spatial_indexes, :])  # re-center without average in space
        source = add_extra_regressors_and_reshape(self.extra_regressors, source, 4, target_positions,
                                                  observed_pollution, order="F",
                                                  **kwargs)
        if kwargs.get("training_mode", False):  # fit to correct average
            target_pollution = kwargs.get("target_pollution", observed_pollution).values
            self.source_model.fit(
                source,
                (target_pollution.reshape(len(times_indexes), -1) - avg).reshape((-1, 1), order="F"),
                sample_weight=1 / target_pollution.ravel(order="F") if self.train_with_relative_error else None)
        u = self.source_model.predict(source).reshape((len(times_indexes), -1),
                                                      order="F") + avg  # add average to predictions.

        # final solution
        return u


class ProjectionAfterSourceModel(BaseModel):

    def __init__(self, source_model: NodeSourceModel, k: Optim, basis="pca",
                 name="", loss=mse, optim_method=GRAD, cv_in_space=True,
                 verbose=False, niter=1000, sigma0=1, **kwargs):
        # super call
        super().__init__(name=f"{name}_{source_model}",
                         loss=loss, optim_method=optim_method, verbose=verbose,
                         niter=niter, sigma0=sigma0, cv_in_space=cv_in_space, k=k, **kwargs)
        self.k_max = k if isinstance(k, int) else k.upper
        self.source_model = source_model
        self.basis = basis

    def calibrate(self, observed_stations, observed_pollution: pd.DataFrame, traffic, **kwargs):
        self.source_model.calibrate(observed_stations=observed_stations, observed_pollution=observed_pollution,
                                    traffic=traffic, **kwargs)

        graph = nx.Graph(kwargs["graph"]).to_undirected()
        if len(nx.get_edge_attributes(graph, "length")) != graph.number_of_edges():
            raise Exception("Each edge in Graph should have edge attribute 'length'")
        if len(nx.get_edge_attributes(graph, "lanes")) != graph.number_of_edges():
            raise Exception("Each edge in Graph should have edge attribute 'lanes'")

        # average in space
        avg = np.mean(observed_pollution.values, axis=1, keepdims=True) if self.source_model.substract_mean else 0
        source = self.source_model.state_estimation(observed_stations=observed_stations,
                                                    observed_pollution=observed_pollution,
                                                    traffic=traffic,
                                                    target_positions=self.source_model.node_positions.T, **kwargs)

        if self.basis == "graph_laplacian":
            Kd = get_diffusion_matrix(path=self.source_model.path4preprocess, filename=f"diffusion_matrix", graph=graph,
                                      recalculate=self.source_model.redo_preprocessing)
            Ms = get_absorption_matrix(path=self.source_model.path4preprocess, filename=f"absorption_matrix",
                                       graph=graph,
                                       recalculate=self.source_model.redo_preprocessing)
            # [# nodes, k]
            self.B, _ = get_geometric_basis(path=self.source_model.path4preprocess,
                                            filename=f"LaplacianModel_basis_k{self.k_max}",
                                            Kd=Kd, Ms=Ms,
                                            k=self.k_max,
                                            recalculate=self.source_model.redo_preprocessing)
        elif self.basis == "pca":
            # [# nodes, k]
            source_avg = np.mean(source, axis=0, keepdims=True)
            self.B = PCA(n_components=self.k_max).fit(source - source_avg).components_.T
        else:
            raise Exception("Invalid basis")

        model = calibrate(self, observed_stations=observed_stations, observed_pollution=observed_pollution,
                          traffic=traffic, source=source, avg=avg, **kwargs)
        self.set_params(**model.params)

    def state_estimation(self, observed_stations, observed_pollution, traffic, target_positions: pd.DataFrame,
                         **kwargs) -> np.ndarray:
        spatial_unknown_indexes = self.source_model.get_space_indexes(target_positions)
        k = int(self.params["k"])

        # source model predictions
        if "source" in kwargs and "avg" in kwargs:  # when calibarting do not re calculate each time.
            source = kwargs["source"]
            avg = kwargs["avg"]
        else:
            # average in space
            avg = np.mean(observed_pollution.values, axis=1, keepdims=True) if self.source_model.substract_mean else 0
            source = self.source_model.state_estimation(observed_stations=observed_stations,
                                                        observed_pollution=observed_pollution,
                                                        traffic=traffic,
                                                        target_positions=self.source_model.node_positions.T, **kwargs)

        u = np.einsum("tk,dk->td",
                      np.einsum("dk,td->tk", self.B[:, :k], source - avg), self.B[:, :k])

        return u[:, spatial_unknown_indexes] + avg


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

        s = add_extra_regressors_and_reshape(self.extra_regressors, s, 4, observed_stations, observed_pollution,
                                             order="C",
                                             **kwargs)
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
            for k in tqdm(range(1, self.k_max), desc="Finding optimal number of Vn components."):
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

        source = add_extra_regressors_and_reshape(self.extra_regressors, source, 4, target_positions,
                                                  observed_pollution, order="C",
                                                  **kwargs)
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
        self.B, _ = get_geometric_basis(path=path4preprocess, filename=f"LaplacianModel_basis_k{k_max}", Kd=Kd, Ms=Ms,
                                        k=k_max, recalculate=redo_preprocessing)

    def project_to_subspace(self, source, k):
        return np.einsum("tkc,dk->tdc", np.einsum("tdc,dk->tkc", source, self.B[:, :k]), self.B[:, :k])

    def init_Vn_space(self, source):
        pass  # it is the self.B of init method.


class SoftDiffusion(BaseModel):
    def __init__(self, path4preprocess: Union[str, Path], source_model: BaseSourceModel, graph,
                 spacial_locations: pd.DataFrame, times,
                 traffic_by_edge, extra_regressors=[], substract_mean=False,
                 name="", loss=mse, optim_method=GRAD, cv_in_space=True,
                 verbose=False, redo_preprocessing=False,
                 niter=1000, sigma0=1, **kwargs):
        self.path4preprocess = path4preprocess
        self.redo_preprocessing = redo_preprocessing
        self.source_model = source_model
        self.substract_mean = substract_mean

        # super call
        super().__init__(name=f"{name}",
                         loss=loss, optim_method=optim_method, verbose=verbose,
                         niter=niter, sigma0=sigma0,
                         cv_in_space=cv_in_space, **kwargs)

    def calibrate(self, observed_stations, observed_pollution: pd.DataFrame, traffic, **kwargs):
        self.source_model.calibrate(observed_stations=observed_stations, observed_pollution=observed_pollution,
                                    traffic=traffic, **kwargs)

        graph = nx.Graph(kwargs["graph"]).to_undirected()
        if len(nx.get_edge_attributes(graph, "length")) != graph.number_of_edges():
            raise Exception("Each edge in Graph should have edge attribute 'length'")
        if len(nx.get_edge_attributes(graph, "lanes")) != graph.number_of_edges():
            raise Exception("Each edge in Graph should have edge attribute 'lanes'")

        # average in space
        avg = np.mean(observed_pollution.values, axis=1, keepdims=True) if self.substract_mean else 0
        source = self.source_model.state_estimation(observed_stations=observed_stations,
                                                    observed_pollution=observed_pollution,
                                                    traffic=traffic,
                                                    target_positions=self.source_model.node_positions.T, **kwargs)

        model = calibrate(self, observed_stations=observed_stations, observed_pollution=observed_pollution,
                          traffic=traffic, source=source, avg=avg, **kwargs)
        self.set_params(**model.params)

    def state_estimation(self, observed_stations, observed_pollution, traffic, target_positions: pd.DataFrame,
                         **kwargs) -> np.ndarray:
        # average in space
        avg = np.mean(observed_pollution.values, axis=1, keepdims=True) if self.substract_mean else 0

        # source model predictions
        if "source" in kwargs and "avg" in kwargs:  # when calibrating do not re calculate each time.
            source = kwargs["source"]
            avg = kwargs["avg"]
        else:
            source = self.source_model.state_estimation(observed_stations=observed_stations,
                                                        observed_pollution=observed_pollution,
                                                        traffic=traffic,
                                                        target_positions=self.source_model.node_positions.T, **kwargs)

        L = get_laplacian_matrix(path=self.path4preprocess, filename=f"laplacian_matrix", graph=kwargs["graph"],
                                 recalculate=self.redo_preprocessing)

        alpha = self.params["alpha"]
        beta = self.params["beta"]
        delta = self.params["delta"]
        # if alpha == 1:
        #     u = source
        # elif alpha == 0:
        #     u = source.mean(axis=1, keepdims=True) @ np.ones((1, np.shape(source)[1]))
        # else:
        # u = spsolve((1 - alpha) * L + alpha * identity(np.shape(L)[0]), alpha * (source - avg).T).T + avg
        try:
            u = spsolve(beta * L + alpha * identity(np.shape(L)[0]), (source - avg).T).T + avg + delta
        except RuntimeError:
            u = 0 * source  # singular matrix or something
        # import matplotlib.pylab as plt
        # import seaborn as sns
        # import pandas as pd
        # alpha = 0.7
        # u = spsolve((1 - alpha) * L + alpha * identity(np.shape(L)[0]), alpha * (source - avg).T).T + avg
        # sns.histplot(pd.DataFrame(np.transpose([source[:, 0], u[:, 0]]), columns=["source", "u"]))
        # # plt.hist(source[:, 0], bins=100)
        # # plt.hist(source[:, 0], bins=100)
        # plt.show()

        spatial_unknown_indexes = self.source_model.get_space_indexes(target_positions)
        return u[:, spatial_unknown_indexes]


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
            self.B, _ = get_geometric_basis(path=self.path4preprocess, filename=f"PhysicsModel_basis_k{self.rb_k_max}",
                                            Kd=Kd, Ms=Ms,
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
    def __init__(self, models: List[BaseModel], aggregator: Pipeline, name=None, weighting="average",
                 train_on_llo=True):
        super().__init__()
        self.losses = list()
        self.name = name
        self.TRUE_MODEL = np.all([model.TRUE_MODEL for model in models])
        self.aggregators = aggregator
        self.models = models
        self.weights = dict()
        self.weighting = weighting
        self.train_on_llo = train_on_llo

    def __str__(self):
        models_names = ','.join([''.join(filter(lambda c: c.isupper(), str(model))) for model in
                                 self.models]) if self.name is None else self.name
        return f"{self.aggregators}({models_names})"

    @staticmethod
    def state_estimation_for_each_model(models, observed_stations, observed_pollution, traffic, target_positions,
                                        **kwargs) -> [np.ndarray,
                                                      np.ndarray]:
        return np.transpose([
            model.state_estimation(observed_stations, observed_pollution, traffic, target_positions, **kwargs)
            for model in models], axes=(1, 2, 0))

    def state_estimation(self, observed_stations, observed_pollution, traffic, target_positions: pd.DataFrame,
                         **kwargs) -> np.ndarray:
        predictions = []
        for i, (known_data, target_pollution) in enumerate(
                loo(observed_stations, observed_pollution, traffic, kwargs.get("stations2test", None))):
            target_position_excluded_station = known_data.pop("target_positions")
            station_name = target_position_excluded_station.columns[0]
            if station_name in self.models.keys():
                individual_predictions = self.state_estimation_for_each_model(self.models[station_name], **known_data,
                                                                              target_positions=target_positions,
                                                                              **kwargs)
                shape = np.shape(individual_predictions)
                predictions.append(self.weights[station_name] * self.aggregators[station_name].predict(
                    individual_predictions.reshape((-1, shape[-1]))).reshape(
                    (shape[0], shape[1])))

        return np.sum(predictions, axis=0)

    def calibrate(self, observed_stations, observed_pollution: pd.DataFrame, traffic, **kwargs):
        if not self.calibrated:
            blue = BLUEModel(name="BLUE", loss=mse, optim_method=NONE_OPTIM_METHOD)
            blue.calibrate(observed_stations, observed_pollution, traffic, **kwargs)

            models = dict()
            aggregators = dict()
            for i, (known_data, target_pollution) in tqdm(enumerate(
                    loo(observed_stations, observed_pollution, traffic, kwargs.get("stations2test", None))),
                    desc="Training individual models and aggregator."):
                target_position_excluded_station = known_data.pop("target_positions")
                station_name = target_position_excluded_station.columns[0]

                # Train individual models
                models[station_name] = [copy.deepcopy(model) for model in self.models]
                for model in models[station_name]:
                    model.calibrate(**known_data, **kwargs)

                # Train aggregator
                aggregators[station_name] = copy.deepcopy(self.aggregators)
                if self.train_on_llo:
                    known_data["target_positions"] = target_position_excluded_station
                    individual_predictions = self.state_estimation_for_each_model(models[station_name], **known_data,
                                                                                  **kwargs)
                    shape = np.shape(individual_predictions)
                    aggregators[station_name].fit(individual_predictions.reshape((-1, shape[-1])),
                                                  target_pollution.values.reshape((-1, 1)))
                    if self.weighting == "average":
                        self.weights[station_name] = 1
                    else:
                        shape = np.shape(individual_predictions)
                        predictions = aggregators[station_name].predict(
                            individual_predictions.reshape((-1, shape[-1]))).reshape(
                            (shape[0], shape[1]))
                        if self.weighting == "BLUE_weighting":
                            blue_prediction = blue.state_estimation(**known_data, **kwargs)
                            self.weights[station_name] = np.mean(
                                (target_pollution.values - blue_prediction.squeeze()) ** 2) / np.mean(
                                (target_pollution.values - predictions.squeeze()) ** 2)
                        elif self.weighting == "mse_weighting":
                            self.weights[station_name] = (
                                    1 / np.mean((target_pollution.values - predictions.squeeze()) ** 2))
                        else:
                            raise Exception("Not implemented.")
                else:
                    # train on observed stations
                    known_data["target_positions"] = known_data["observed_stations"]
                    individual_predictions = self.state_estimation_for_each_model(models[station_name], **known_data,
                                                                                  **kwargs)
                    shape = np.shape(individual_predictions)
                    aggregators[station_name].fit(individual_predictions.reshape((-1, shape[-1])),
                                                  known_data["observed_pollution"].values.reshape((-1, 1)))

                    if self.weighting == "average":
                        self.weights[station_name] = 1
                    else:
                        # weight on llo station
                        known_data["target_positions"] = target_position_excluded_station
                        individual_predictions = self.state_estimation_for_each_model(models[station_name],
                                                                                      **known_data,
                                                                                      **kwargs)
                        shape = np.shape(individual_predictions)
                        predictions = aggregators[station_name].predict(
                            individual_predictions.reshape((-1, shape[-1]))).reshape(
                            (shape[0], shape[1]))
                        if self.weighting == "BLUE_weighting":
                            blue_prediction = blue.state_estimation(**known_data, **kwargs)
                            self.weights[station_name] = np.mean(
                                (target_pollution.values - blue_prediction.squeeze()) ** 2) / np.mean(
                                (target_pollution.values - predictions.squeeze()) ** 2)
                        elif self.weighting == "mse_weighting":
                            self.weights[station_name] = (
                                    1 / np.mean((target_pollution.values - predictions.squeeze()) ** 2))
                        else:
                            raise Exception("Not implemented.")

            self.models = models
            self.aggregators = aggregators
            normalization = np.sum(list(self.weights.values()))
            self.weights = {k: v / normalization for k, v in self.weights.items()}
            print(self.weighting)
            print(sum(self.weights.values()))
            print(self.weights)
            self.calibrated = True


class ModelsAggregatorNoCV(BaseModel):
    def __init__(self, models: List[BaseModel], aggregator, name=None, extra_regressors=[]):
        super().__init__()
        self.losses = list()
        self.name = name
        self.TRUE_MODEL = np.all([model.TRUE_MODEL for model in models])
        self.aggregator = aggregator
        self.models = models
        if isinstance(aggregator, str):
            extra_regressors = []
        self.extra_regressors = extra_regressors

    def __str__(self):
        models_names = ','.join([''.join(filter(lambda c: c.isupper(), str(model))) for model in
                                 self.models]) if self.name is None else self.name
        return f"{self.aggregator}({models_names})"

    @staticmethod
    def state_estimation_for_each_model(models, observed_stations, observed_pollution, traffic, target_positions,
                                        **kwargs) -> [np.ndarray,
                                                      np.ndarray]:
        return np.transpose([
            model.state_estimation(observed_stations, observed_pollution, traffic, target_positions, **kwargs)
            for model in models], axes=(1, 2, 0))

    def state_estimation(self, observed_stations, observed_pollution, traffic, target_positions: pd.DataFrame,
                         **kwargs) -> np.ndarray:
        individual_predictions = self.state_estimation_for_each_model(
            self.models, observed_stations, observed_pollution, traffic,
            target_positions=target_positions, **kwargs)
        shape = np.shape(individual_predictions)
        individual_predictions = add_extra_regressors_and_reshape(
            self.extra_regressors, individual_predictions, shape[-1], target_positions, observed_pollution, order="F",
            **kwargs)
        if isinstance(self.aggregator, np.ndarray):
            u = individual_predictions @ self.aggregator
        else:
            u = self.aggregator.predict(individual_predictions)
        return u.reshape((shape[0], shape[1]), order="F")

    def calibrate(self, observed_stations, observed_pollution: pd.DataFrame, traffic, **kwargs):
        for i, _ in enumerate(self.models):
            self.models[i].calibrate(observed_stations, observed_pollution, traffic, **kwargs)

        individual_predictions = self.state_estimation_for_each_model(
            self.models, observed_stations, observed_pollution, traffic,
            target_positions=observed_stations, **kwargs)
        shape = np.shape(individual_predictions)
        individual_predictions = add_extra_regressors_and_reshape(
            self.extra_regressors, individual_predictions, shape[-1], observed_stations, observed_pollution, order="F",
            **kwargs)

        ground_truth = observed_pollution.values.reshape((-1, 1), order="F")
        if isinstance(self.aggregator, str):
            if self.aggregator == "average":
                self.aggregator = np.ones((len(self.models), 1))
            elif self.aggregator == "weighted_average":
                c = np.corrcoef(individual_predictions)
                self.aggregator = minimize(lambda w: w.T @ c @ w, x0=np.ones((len(self.models), 1)) / len(self.models),
                                           constraints={
                                               "eq:norm": LinearConstraint(np.ones(len(self.models)), lb=0.9, ub=1.1,
                                                                           keep_feasible=False)}).x
            elif self.aggregator == "std":
                error = (individual_predictions - ground_truth) ** 2
                self.aggregator = np.sqrt(1 / np.std(error, axis=0))[:, np.newaxis]
            elif self.aggregator == "cv":
                error = (individual_predictions - ground_truth) ** 2
                self.aggregator = np.sqrt(np.mean(error, axis=0) / np.std(error, axis=0))[:, np.newaxis]
            else:
                raise Exception("Not implemented.")
            self.aggregator /= np.sum(self.aggregator)
            print("Aggregator weights: ", self.aggregator.ravel())
        else:
            self.aggregator.fit(individual_predictions, ground_truth)
