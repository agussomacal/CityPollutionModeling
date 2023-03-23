import datetime
from typing import List, Dict, Union

import numpy as np

from src.lib.Models.BaseModel import BaseModel


class SummaryModel(BaseModel):
    def __init__(self, summary_statistic="mean"):
        assert summary_statistic in ["mean", "median"], "summary_statistic should be one of 'mean', 'median'"
        self.summary_statistic = summary_statistic
        super().__init__(name=summary_statistic)


class SnapshotMeanModel(SummaryModel):
    def calibrate(self, observed_pollution, traffic, target_positions, **kwargs):
        return self

    def state_estimation(self, observed_pollution, traffic, target_positions, **kwargs) -> np.ndarray:
        summary_function = getattr(np, 'nan'+self.summary_statistic)
        return summary_function(observed_pollution.values, axis=1)[:, np.newaxis] * \
               np.ones((1, np.shape(target_positions)[1]))


class GlobalMeanModel(SummaryModel):
    def __init__(self, summary_statistic="mean", global_mean=0):
        super(GlobalMeanModel, self).__init__(summary_statistic=summary_statistic)
        self.set_params(global_mean=global_mean)

    def calibrate(self, observed_pollution, traffic, target_positions, **kwargs):
        summary_function = getattr(np, 'nan'+self.summary_statistic)
        self.set_params(global_mean=summary_function(observed_pollution.values))
        return self

    def state_estimation(self, observed_pollution, traffic, target_positions, **kwargs) -> np.ndarray:
        return np.ones((len(observed_pollution), np.shape(target_positions)[1])) * self.params["global_mean"]
