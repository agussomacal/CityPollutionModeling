import copy
import datetime
from typing import List, Dict, Union

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler

from src.lib.Models.BaseModel import BaseModel, loo, pollution_dependent, mse, NONE_OPTIM_METHOD, pollution_agnostic, \
    split_loo
from PerplexityLab.miscellaneous import filter_dict, if_true_str


class SummaryModel(BaseModel):
    def __init__(self, summary_statistic="mean"):
        assert summary_statistic in ["mean", "median"], "summary_statistic should be one of 'mean', 'median'"
        self.summary_statistic = summary_statistic
        super().__init__(name=summary_statistic)


class SnapshotMeanModel(SummaryModel):
    def calibrate(self, observed_stations, observed_pollution, traffic, **kwargs):
        self.calibrated = True
        return self

    def state_estimation(self, observed_stations, observed_pollution, traffic, target_positions,
                         **kwargs) -> np.ndarray:
        summary_function = getattr(np, 'nan' + self.summary_statistic)
        return summary_function(observed_pollution.values, axis=1)[:, np.newaxis] * \
            np.ones((1, np.shape(target_positions)[1]))


class SnapshotPCAModel(SummaryModel, BaseModel):
    def __init__(self, name="", n_components=1, summary_statistic="mean", loss=mse, optim_method=NONE_OPTIM_METHOD,
                 niter=1000, verbose=False, sigma0=1):
        SummaryModel.__init__(self, summary_statistic=summary_statistic)
        BaseModel.__init__(self, name=name, loss=loss, optim_method=optim_method, niter=niter,
                           verbose=verbose,
                           sigma0=sigma0, n_components=n_components)
        self.zscore = StandardScaler()
        self.pca = None
        self.vectors = None

    @pollution_agnostic
    def state_estimation_for_optim(self, observed_stations, observed_pollution, traffic, **kwargs) -> [np.ndarray,
                                                                                                       np.ndarray]:
        not_nan = ~np.isnan(observed_pollution).any(axis=1)
        self.pca = PCA(n_components=int(self.params["n_components"]))
        self.pca.fit(self.zscore.fit_transform(observed_pollution)[not_nan, :])
        self.vectors = (self.pca.singular_values_[:, np.newaxis] * self.pca.components_) / np.sqrt(
            (self.pca.singular_values_ ** 2).sum())
        return self.state_estimation(observed_stations, observed_pollution, traffic, observed_stations, **kwargs)

    def state_estimation(self, observed_stations, observed_pollution, traffic, target_positions,
                         **kwargs) -> np.ndarray:
        summary_function = getattr(np, 'nan' + self.summary_statistic)
        preds = []
        for i in range(len(observed_pollution)):
            pollution_t = observed_pollution.iloc[[i], :]
            pollution_t = pollution_t.dropna(thresh=1, axis=1)
            # pollution_t[~pollution_t.isna()]
            # pollution_t.loc[:, ~np.isnan(pollution_t.values.ravel())]
            xy, x_ind, y_ind = np.intersect1d(self.zscore.feature_names_in_, pollution_t.columns, return_indices=True)

            new_zscore = copy.deepcopy(self.zscore)
            new_zscore.mean_ = self.zscore.mean_[x_ind]
            new_zscore.scale_ = self.zscore.scale_[x_ind]
            new_zscore.feature_names_in_ = self.zscore.feature_names_in_[x_ind]
            new_zscore.n_features_in_ = len(x_ind)

            new_pca = copy.deepcopy(self.pca)
            new_pca.components_ = self.pca.components_[:, x_ind]
            new_pca.mean_ = self.pca.mean_[x_ind]
            new_pca.n_features = len(x_ind)
            new_pca.n_features_in_ = len(x_ind)

            stabilized_observations = new_zscore.inverse_transform(
                new_pca.transform(new_zscore.transform(pollution_t[xy])) @ self.vectors[:, x_ind])

            preds.append(summary_function(stabilized_observations, axis=1))
        return np.array(preds) * np.ones((1, np.shape(target_positions)[1]))


class SnapshotBLUEModel(BaseModel):
    def __init__(self, name="", sensor_distrust=0.0, loss=mse, optim_method=NONE_OPTIM_METHOD,
                 niter=1000, verbose=False, sigma0=1):
        super().__init__(name=name, loss=loss, optim_method=optim_method, niter=niter,
                         verbose=verbose, sigma0=sigma0, sensor_distrust=sensor_distrust)
        self.zscore = StandardScaler()
        self.correlation = None

    @property
    def sensor_distrust(self):
        if "sensor_distrust" in self.params.keys():
            return np.repeat(self.params["sensor_distrust"], len(self.correlation))
        else:
            assert len(self.correlation.index) == len(set(self.correlation.index).intersection(
                self.params.keys())), "all the stations named in correlation should be in parameter keys."
            return np.array([self.params[c] for c in self.correlation.columns])  # the good ordering.

    @pollution_agnostic
    def state_estimation_for_optim(self, observed_stations, observed_pollution, traffic, **kwargs) -> [np.ndarray,
                                                                                                       np.ndarray]:
        self.correlation = observed_pollution.corr()
        # distrust should be positive otherwise it means there is more confidence than 0-noise case
        self.correlation.values[np.diag_indices(len(self.correlation))] += np.abs(self.sensor_distrust)
        self.zscore.fit(observed_pollution)
        return self.state_estimation(observed_stations, observed_pollution, traffic, observed_stations, **kwargs)

    def state_estimation(self, observed_stations, observed_pollution, traffic, target_positions,
                         **kwargs) -> np.ndarray:
        preds = []
        for i in range(len(observed_pollution)):
            pollution_t = observed_pollution.iloc[[i], :]
            pollution_t = pollution_t.dropna(thresh=1, axis=1)
            # pollution_t[~pollution_t.isna()]
            # pollution_t.loc[:, ~np.isnan(pollution_t.values.ravel())]
            xy, x_ind, y_ind = np.intersect1d(self.zscore.feature_names_in_, pollution_t.columns, return_indices=True)

            new_zscore = copy.deepcopy(self.zscore)
            new_zscore.mean_ = self.zscore.mean_[x_ind]
            new_zscore.scale_ = self.zscore.scale_[x_ind]
            new_zscore.feature_names_in_ = self.zscore.feature_names_in_[x_ind]
            new_zscore.n_features_in_ = len(x_ind)

            p = pd.Series(new_zscore.transform(pollution_t[xy]).ravel(), index=xy)
            preds.append([])
            for ix in range(len(xy)):
                known_stations = xy.tolist()
                known_stations.pop(ix)
                b = self.correlation.loc[known_stations, xy[ix]].values
                c = self.correlation.loc[known_stations, known_stations].values
                # a = np.linalg.solve(c, b)
                a = Ridge(alpha=1.0, tol=0.1, fit_intercept=False).fit(c, b).coef_.T
                # a = np.linalg.lstsq(c, b, rcond=-1)[0]
                # a = scipy.linalg.solve(c, b, assume_a='pos')  # singular matrix
                preds[-1].append(p[known_stations] @ a)
            preds[-1] = np.mean(new_zscore.inverse_transform(np.reshape(preds[-1], (1, -1))))
        return np.array(preds)[:, np.newaxis] * np.ones((1, np.shape(target_positions)[1]))


class SnapshotWeightedModel(BaseModel):
    def __init__(self, positive=True, fit_intercept=False, **kwargs):
        self.positive = positive
        self.fit_intercept = fit_intercept
        super().__init__(
            name=if_true_str(positive, "positive", "_") + if_true_str(fit_intercept, "fit_intercept", "_"),
            **kwargs)

    def state_estimation(self, observed_stations, observed_pollution, traffic, target_positions,
                         **kwargs) -> np.ndarray:
        weight = pd.concat([pd.Series(filter_dict(observed_stations.columns, self.params))] * len(observed_pollution),
                           axis=1).T
        p_up = observed_pollution.shape[1] / (~observed_pollution.isna().values).sum(axis=1)

        return np.nansum(observed_pollution.values * weight.values * p_up[:, np.newaxis], axis=1, keepdims=True)

        # pred = (observed_pollution @ pd.Series(filter_dict(observed_stations.columns, self.params))).values \
        #     .reshape((-1, np.shape(target_positions)[1]))

    def calibrate(self, observed_stations, observed_pollution: pd.DataFrame, traffic, **kwargs) -> [np.ndarray,
                                                                                                    np.ndarray]:
        """
        traffic_coords: pd.DataFrame with columns the pixel_coord and rows 'lat' and 'long' associated to the pixel
        target_positions: pd.DataFrame with columns the name of the station and rows 'lat' and 'long'
        """
        target, pollution = \
            list(zip(*[(target_pollution, pd.concat((known_data["observed_pollution"], target_pollution), axis=1))
                       for known_data, target_pollution in
                       loo(observed_stations, observed_pollution, traffic, kwargs.get("stations2test", None))]))
        target = pd.concat(target).values
        pollution = pd.concat(pollution)
        pollution_columns = pollution.columns
        pollution = pollution.values
        pollution[
            list(range(len(pollution))), list(map(lambda x: x[0].tolist().index(x[1]), zip(pollution, target)))] = 0
        pollution[np.isnan(pollution)] = 0

        lr = LinearRegression(fit_intercept=self.fit_intercept, positive=self.positive)
        lr.fit(pollution, target)  # , sample_weight=1 / observed_pollution[pollution_columns].std()
        self.set_params(**dict(zip(pollution_columns, lr.coef_)))
        self.calibrated = True
        return self


class SnapshotWeightedStd(BaseModel):
    def state_estimation(self, observed_stations, observed_pollution, traffic, target_positions,
                         **kwargs) -> np.ndarray:
        weight = pd.concat([pd.Series(filter_dict(observed_stations.columns, self.params))] * len(observed_pollution),
                           axis=1).T
        weight.values[observed_pollution.isna().values] = 0
        return np.nansum(observed_pollution.values * weight.values / weight.sum(axis=1).values[:, np.newaxis], axis=1,
                         keepdims=True)

    def calibrate(self, observed_stations, observed_pollution: pd.DataFrame, traffic, **kwargs) -> [np.ndarray,
                                                                                                    np.ndarray]:
        """
        traffic_coords: pd.DataFrame with columns the pixel_coord and rows 'lat' and 'long' associated to the pixel
        target_positions: pd.DataFrame with columns the name of the station and rows 'lat' and 'long'
        """
        self.set_params(**(1 / observed_stations.var()).to_dict())
        self.calibrated = True
        return self


class GlobalMeanModel(SummaryModel):
    def __init__(self, summary_statistic="mean", global_mean=0):
        super(GlobalMeanModel, self).__init__(summary_statistic=summary_statistic)
        self.set_params(global_mean=global_mean)

    def calibrate(self, observed_stations, observed_pollution, traffic, **kwargs):
        summary_function = getattr(np, 'nan' + self.summary_statistic)
        self.set_params(global_mean=summary_function(observed_pollution.values))
        self.calibrated = True
        return self

    def state_estimation(self, observed_stations, observed_pollution, traffic, target_positions,
                         **kwargs) -> np.ndarray:
        return np.ones((len(observed_pollution), np.shape(target_positions)[1])) * self.params["global_mean"]
