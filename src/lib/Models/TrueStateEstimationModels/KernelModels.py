import inspect
from typing import Dict

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from PerplexityLab.miscellaneous import filter_dict
from src.lib.Models.BaseModel import BaseModel, mse, GRAD
from src.lib.Modules import Optim


# ================ ================ ================ #
#                   Types of kernel                  #
# ================ ================ ================ #
def gaussian_kernel(x, y, sigma):
    """

    @param x: vector 1
    @param y: vector 2
    @param sigma: sigma of the gaussian kernel
    @return:
    """
    # the if condition with the axis is to account for many vectors simultaneously, first dimension is the iterator,
    # the others correspond to the vector.

    return np.exp(-cdist(np.matrix(x), np.matrix(y), metric='euclidean') ** 2 / (2 * sigma ** 2))


def exponential_kernel(x, y, alpha):
    """

    @param x: vector 1
    @param y: vector 2
    @param sigma: sigma of the gaussian kernel
    @return:
    """
    # the if condition with the axis is to account for many vectors simultaneously, first dimension is the iterator,
    # the others correspond to the vector.

    return np.exp(- alpha * cdist(np.matrix(x), np.matrix(y), metric='euclidean'))


def rational_kernel(x, y, alpha, beta):
    """

    @param x: vector 1
    @param y: vector 2
    @param sigma: sigma of the gaussian kernel
    @return:
    """
    # the if condition with the axis is to account for many vectors simultaneously, first dimension is the iterator,
    # the others correspond to the vector.

    return alpha / (beta + cdist(np.matrix(x), np.matrix(y), metric='euclidean'))


# ================ ================ ================ #
#                 Generic KernelModel                #
# ================ ================ ================ #
class KernelModel(BaseModel):

    def __init__(self, kernel_function, name="", loss=mse, optim_method=GRAD, verbose=False,
                 niter=1000, sensor_distrust=0, old_method=False, **kwargs):
        if isinstance(sensor_distrust, (float, int, Optim)):
            sensor_distrust = {"sensor_distrust": sensor_distrust}
        elif isinstance(sensor_distrust, Dict):
            pass
        else:
            raise Exception("The parameter sensor_distrust must be numeric or dict")
        super().__init__(name=name, loss=loss, optim_method=optim_method, verbose=verbose,
                         niter=niter, cv_in_space=True, **sensor_distrust, **kwargs)
        self.kernel_function = kernel_function
        self.kernel_func_param_names = inspect.getfullargspec(kernel_function).args[2:]
        self.observed_locations = None
        self.k_matrix = None
        self.old_method = old_method

    def get_sensor_distrust(self, k_matrix):
        if "sensor_distrust" in self.params.keys():
            return np.repeat(self.params["sensor_distrust"], len(k_matrix))
        else:
            assert len(k_matrix.index) == len(set(k_matrix.index).intersection(
                self.params.keys())), "all the stations named in correlation should be in parameter keys."
            return np.array([self.params[c] for c in k_matrix.columns])  # the good ordering.

    def state_estimation(self, observed_stations, observed_pollution, traffic, target_positions: pd.DataFrame,
                         **kwargs) -> np.ndarray:
        know_stations = observed_pollution.columns

        self.calculate_K_matrix(observed_stations)
        k_matrix = self.k_matrix.loc[know_stations, know_stations]
        sensor_distrust = self.get_sensor_distrust(k_matrix)

        if self.old_method:
            # Old method
            k_matrix.values[np.diag_indices(len(k_matrix))] += sensor_distrust

            umean = zmean = 0  # np.mean(observed_values)
            b = np.linalg.lstsq(k_matrix, (observed_pollution - zmean).T, rcond=None)[0].T
            prediction = umean + b @ self.kernel_eval(observed_stations, target_positions)
        else:
            # TODO: check if not uniform distrust is written this way
            k_matrix = k_matrix @ np.diag(1 - sensor_distrust) + np.diag(sensor_distrust)
            gr = np.diag(1 - sensor_distrust) @ self.kernel_eval(observed_stations, target_positions)

            c = np.linalg.solve(k_matrix, gr)  # solving linear system
            c = c / np.nansum(c)  # normaization to sum 1
            c[np.isnan(c)] = 0

            m = len(know_stations)
            umean = zmean = np.mean(observed_pollution.values, axis=1, keepdims=True)
            prediction = umean + (observed_pollution[know_stations].values - zmean) @ (c - 1.0 / m)
        return prediction

    def calculate_K_matrix(self, observed_stations):
        """

        @param observed_stations: x_i coordinates of observed pollution values
        define the kernel matrix of pairwise evaluations K_ij = k(x_i, x_j)
        """
        if self.observed_locations is None:
            self.observed_locations = observed_stations
        else:
            new_stations = list(set(observed_stations.columns).difference(self.observed_locations.columns))
            self.observed_locations = pd.concat((self.observed_locations, observed_stations[new_stations]), axis=1)
        # TODO: only calculate the distances between the new positions and the new+old.
        self.k_matrix = pd.DataFrame(
            self.kernel_function(self.observed_locations.values.T, self.observed_locations.values.T,
                                 **filter_dict(self.kernel_func_param_names, self.params)),
            columns=self.observed_locations.columns, index=self.observed_locations.columns)

    def kernel_eval(self, observed_stations, target_positions) -> np.ndarray:
        """
        @param observed_stations: x_i where observations exist -> give the kernel representers k_xi
        @param target_positions: spatial points where inference is performed, x_j
        """

        return self.kernel_function(observed_stations.values.T, target_positions.values.T,
                                    **filter_dict(self.kernel_func_param_names, self.params))


# ================ ================ ================ #
#               Specific kernel models               #
# ================ ================ ================ #
class GaussianKernelModel(KernelModel):
    def __init__(self, sigma, sensor_distrust=0, name="", loss=mse, optim_method=GRAD, niter=1000, verbose=False,
                 old_method=False):
        super().__init__(name=name, kernel_function=gaussian_kernel, loss=loss, niter=niter,
                         optim_method=optim_method, sigma=sigma, sensor_distrust=sensor_distrust,
                         verbose=verbose, old_method=old_method)

    def calibrate(self, observed_stations, observed_pollution: pd.DataFrame, traffic, **kwargs):

        if self.params["sigma"] is None or self.params["sensor_distrust"] is None:
            correlation = observed_pollution.corr()
            indexes = np.triu_indices(len(correlation), k=0)
            log_cor = np.log(correlation.values[indexes])
            valid_indexes = ~np.isnan(log_cor)  # when there is negative correlation this is not a good model
            if np.sum(valid_indexes) / len(valid_indexes) < 1:
                print(f"Exponential kernel model assumes positive correlation but some "
                      f"({100 * (1 - np.sum(valid_indexes) / len(valid_indexes))}%) correlations are negative")
            exp_correlations = cdist(observed_stations.values.T, observed_stations.values.T)[indexes][valid_indexes]
            beta, alpha = np.ravel(
                np.linalg.lstsq(np.transpose([np.ones(np.sum(valid_indexes)), exp_correlations ** 2]),
                                np.reshape(log_cor[valid_indexes], (-1, 1)), rcond=-1)[0])
            sensor_distrust = (1 - np.exp(beta))
            sensor_distrust[sensor_distrust > 1] = 1.0
            sensor_distrust[sensor_distrust < 0] = 0.0
            self.set_params(sigma=np.sqrt(np.abs(1 / alpha)), sensor_distrust=sensor_distrust)

        super().calibrate(observed_stations, observed_pollution, traffic, **kwargs)


class ExponentialKernelModel(KernelModel):
    def __init__(self, alpha=None, sensor_distrust=0, name="", loss=mse, optim_method=GRAD, niter=1000,
                 verbose=False, old_method=False):
        super().__init__(name=name, kernel_function=exponential_kernel, loss=loss, niter=niter,
                         optim_method=optim_method, alpha=alpha, sensor_distrust=sensor_distrust,
                         verbose=verbose, old_method=old_method)

    def calibrate(self, observed_stations, observed_pollution: pd.DataFrame, traffic, **kwargs):

        if self.params["alpha"] is None or self.params["sensor_distrust"] is None:
            correlation = observed_pollution.corr()
            indexes = np.triu_indices(len(correlation), k=0)
            log_cor = np.log(correlation.values[indexes])
            valid_indexes = ~np.isnan(log_cor)  # when there is negative correlation this is not a good model
            if np.sum(valid_indexes) / len(valid_indexes) < 1:
                print(f"Exponential kernel model assumes positive correlation but some "
                      f"({100 * (1 - np.sum(valid_indexes) / len(valid_indexes))}%) correlations are negative")
            exp_correlations = cdist(observed_stations.values.T, observed_stations.values.T)[indexes][valid_indexes]
            beta, alpha = np.ravel(np.linalg.lstsq(np.transpose([np.ones(np.sum(valid_indexes)), exp_correlations]),
                                                   np.reshape(log_cor[valid_indexes], (-1, 1)), rcond=-1)[0])
            self.set_params(alpha=-alpha)
            if ("sensor_distrust" not in self.params) or (self.params["sensor_distrust"] is None) or len(
                    self.params) < 3:
                sensor_distrust = (1 - np.exp(beta))
                sensor_distrust = 1.0 if sensor_distrust > 1 else sensor_distrust
                sensor_distrust = 0.0 if sensor_distrust < 0 else sensor_distrust
                self.set_params(sensor_distrust=sensor_distrust)

        super().calibrate(observed_stations, observed_pollution, traffic, **kwargs)


# ================ ================ ================ #
#                 Generic DistanceModel                #
# ================ ================ ================ #
class DistanceModel(BaseModel):

    def __init__(self, kernel_function, name="", loss=mse, optim_method=GRAD, verbose=False,
                 niter=1000, **kwargs):
        super().__init__(name=name, loss=loss, optim_method=optim_method, verbose=verbose,
                         niter=niter, cv_in_space=True, **kwargs)
        self.kernel_function = kernel_function
        self.kernel_func_param_names = inspect.getfullargspec(kernel_function).args[2:]

    def state_estimation(self, observed_stations, observed_pollution, traffic, target_positions: pd.DataFrame,
                         **kwargs) -> np.ndarray:
        d = self.kernel_eval(observed_stations, target_positions)
        d /= np.sum(d)
        avg = np.mean(observed_pollution.values, axis=1, keepdims=True)
        return (observed_pollution.values - avg) @ d + avg

    def kernel_eval(self, observed_stations, target_positions) -> np.ndarray:
        """
        @param observed_stations: x_i where observations exist -> give the kernel representers k_xi
        @param target_positions: spatial points where inference is performed, x_j
        """

        return self.kernel_function(observed_stations.values.T, target_positions.values.T,
                                    **filter_dict(self.kernel_func_param_names, self.params))
