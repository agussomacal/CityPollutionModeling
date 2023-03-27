import datetime
from typing import List, Dict, Union

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from src.lib.Models.BaseModel import BaseModel, loo
from src.performance_utils import filter_dict


class SummaryModel(BaseModel):
    def __init__(self, summary_statistic="mean"):
        assert summary_statistic in ["mean", "median"], "summary_statistic should be one of 'mean', 'median'"
        self.summary_statistic = summary_statistic
        super().__init__(name=summary_statistic)


class SnapshotMeanModel(SummaryModel):
    def calibrate(self, observed_stations, observed_pollution, traffic, **kwargs):
        return self

    def state_estimation(self, observed_stations, observed_pollution, traffic, target_positions,
                         **kwargs) -> np.ndarray:
        summary_function = getattr(np, 'nan' + self.summary_statistic)
        return summary_function(observed_pollution.values, axis=1)[:, np.newaxis] * \
            np.ones((1, np.shape(target_positions)[1]))


class SnapshotWeightedModel(BaseModel):

    def state_estimation(self, observed_stations, observed_pollution, traffic, target_positions,
                         **kwargs) -> np.ndarray:
        return (observed_pollution @ pd.Series(filter_dict(observed_stations.columns, **self.params))).values

    def calibrate(self, observed_stations, observed_pollution, traffic, **kwargs) -> [np.ndarray, np.ndarray]:
        """
        traffic_coords: pd.DataFrame with columns the pixel_coord and rows 'lat' and 'long' associated to the pixel
        target_positions: pd.DataFrame with columns the name of the station and rows 'lat' and 'long'
        """

        target, pollution = \
            list(zip(*[(target_pollution, pd.concat((known_data["observed_pollution"], target_pollution), axis=1))
                       for known_data, target_pollution in loo(observed_stations, observed_pollution, traffic)]))
        target = pd.concat(target).values
        pollution = pd.concat(pollution)
        pollution_columns = pollution.columns
        pollution = pollution.values
        pollution[
            list(range(len(pollution))), list(map(lambda x: x[0].tolist().index(x[1]), zip(pollution, target)))] = 0
        pollution[np.isnan(pollution)] = 0

        lr = LinearRegression(fit_intercept=False, positive=True).fit(pollution, target)
        self.set_params(**dict(zip(pollution_columns, lr.coef_)))
        return self


class GlobalMeanModel(SummaryModel):
    def __init__(self, summary_statistic="mean", global_mean=0):
        super(GlobalMeanModel, self).__init__(summary_statistic=summary_statistic)
        self.set_params(global_mean=global_mean)

    def calibrate(self, observed_stations, observed_pollution, traffic, **kwargs):
        summary_function = getattr(np, 'nan' + self.summary_statistic)
        self.set_params(global_mean=summary_function(observed_pollution.values))
        return self

    def state_estimation(self, observed_stations, observed_pollution, traffic, target_positions,
                         **kwargs) -> np.ndarray:
        return np.ones((len(observed_pollution), np.shape(target_positions)[1])) * self.params["global_mean"]
