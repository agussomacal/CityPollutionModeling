import datetime
import inspect
from typing import List, Dict, Union

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from src.lib.Models.BaseModel import BaseModel, mse, GRAD
from PerplexityLab.miscellaneous import filter_dict


# ================ ================ ================ #
#                   Types of kernel                  #
# ================ ================ ================ #
def gaussian_kernel(x, y, sigma, beta):
    """

    @param x: vector 1
    @param y: vector 2
    @param sigma: sigma of the gaussian kernel
    @return:
    """
    # the if condition with the axis is to account for many vectors simultaneously, first dimension is the iterator,
    # the others correspond to the vector.

    return np.exp(beta - cdist(np.matrix(x), np.matrix(y), metric='euclidean') ** 2 / sigma ** 2)


def exponential_kernel(x, y, alpha, beta):
    """

    @param x: vector 1
    @param y: vector 2
    @param sigma: sigma of the gaussian kernel
    @return:
    """
    # the if condition with the axis is to account for many vectors simultaneously, first dimension is the iterator,
    # the others correspond to the vector.

    return np.exp(beta - alpha * cdist(np.matrix(x), np.matrix(y), metric='euclidean'))


# ================ ================ ================ #
#                 Generic KernelModel                #
# ================ ================ ================ #
class KernelModel(BaseModel):

    def __init__(self, kernel_function, distrust=0, name="", loss=mse, optim_method=GRAD, verbose=False,
                 niter=1000, **kwargs):
        super().__init__(name=name, loss=loss, optim_method=optim_method, distrust=distrust, verbose=verbose,
                         niter=niter, **kwargs)
        self.kernel_function = kernel_function
        self.kernel_func_param_names = inspect.getfullargspec(kernel_function).args[2:]
        self.observed_locations = None
        self.k_matrix = None

    @property
    def distrust(self):
        return self.params["distrust"]

    def state_estimation(self, observed_stations, observed_pollution, traffic, target_positions: pd.DataFrame,
                         **kwargs) -> np.ndarray:
        self.calculate_K_matrix(observed_stations)
        k_matrix = self.k_matrix.loc[observed_stations.columns, observed_stations.columns]
        k_matrix.values[np.diag_indices(len(k_matrix))] += self.distrust
        # print(self, ": ", np.linalg.cond(K))

        umean = zmean = 0  # np.mean(observed_values)
        b = np.linalg.lstsq(k_matrix, (observed_pollution - zmean).T, rcond=None)[0].T
        prediction = umean + b @ self.kernel_eval(observed_stations, target_positions)
        # TODO: what to do when observed_stations has nans?
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
    def __init__(self, sigma, beta, distrust=0, name="", loss=mse, optim_method=GRAD, niter=1000, verbose=False):
        super().__init__(name=name, kernel_function=gaussian_kernel, loss=loss, niter=niter,
                         optim_method=optim_method, sigma=sigma, beta=beta, distrust=distrust, verbose=verbose)

    def calibrate(self, observed_stations, observed_pollution: pd.DataFrame, traffic, **kwargs):

        if self.params["sigma"] is None or self.params["beta"] is None:
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
            self.set_params(sigma=np.sqrt(np.abs(1 / alpha)), beta=np.exp(beta))

        super().calibrate(observed_stations, observed_pollution, traffic, **kwargs)


class ExponentialKernelModel(KernelModel):
    def __init__(self, alpha=None, beta=None, distrust=0, name="", loss=mse, optim_method=GRAD, niter=1000,
                 verbose=False):
        super().__init__(name=name, kernel_function=exponential_kernel, loss=loss, niter=niter,
                         optim_method=optim_method, alpha=alpha, beta=beta, distrust=distrust, verbose=verbose)

    def calibrate(self, observed_stations, observed_pollution: pd.DataFrame, traffic, **kwargs):

        if self.params["alpha"] is None or self.params["beta"] is None:
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
            self.set_params(alpha=-alpha, beta=np.exp(beta))

        super().calibrate(observed_stations, observed_pollution, traffic, **kwargs)
