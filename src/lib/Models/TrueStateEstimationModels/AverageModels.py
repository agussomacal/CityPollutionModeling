import datetime
from typing import List, Dict, Union

import numpy as np

from src.lib.Models.BaseModel import BaseModel


class SnapshotMeanModel(BaseModel):

    def __init__(self):
        super().__init__()

    def calibrate(self, observed_pollution, traffic, target_positions, **kwargs):
        return self

    def state_estimation(self, observed_pollution, traffic, target_positions, **kwargs) -> np.ndarray:
        return observed_pollution.values.mean(axis=1)[:, np.newaxis] * np.ones((1, np.shape(target_positions)[1]))


class GlobalMeanModel(BaseModel):

    def __init__(self, global_mean=0):
        super(GlobalMeanModel, self).__init__()
        self.set_params(global_mean=global_mean)

    def calibrate(self, observed_pollution, traffic, target_positions, **kwargs):
        self.set_params(global_mean=np.mean(observed_pollution))
        return self

    def state_estimation(self, observed_pollution, traffic, target_positions, **kwargs) -> np.ndarray:
        return np.ones(np.shape(target_positions)[:2]) * self.params["global_mean"]
