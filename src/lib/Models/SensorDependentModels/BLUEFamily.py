import datetime
from collections import defaultdict
from typing import List, Dict, Union

import numpy as np
import pandas as pd
import scipy
from scipy.spatial.distance import cdist
from sklearn.linear_model import Ridge

from src.lib.Models.BaseModel import BaseModel, mse, NONE_OPTIM_METHOD
from src.lib.Modules import Optim


class BLUEModel(BaseModel):
    TRUE_MODEL = False
    POLLUTION_AGNOSTIC = False

    def __init__(self, name="", sensor_distrust=0.0, loss=mse, optim_method=NONE_OPTIM_METHOD, niter=1000,
                 verbose=False):
        if isinstance(sensor_distrust, (float, int, Optim)):
            sensor_distrust = {"sensor_distrust": sensor_distrust}
        elif isinstance(sensor_distrust, Dict):
            pass
        else:
            raise Exception("The parameter sensor_distrust must be numeric or dict")
        super().__init__(name=name, loss=loss, optim_method=optim_method, verbose=verbose,
                         niter=niter, cv_in_space=True, **sensor_distrust)
        self.correlation_dict = dict()
        self.correlation = None
        self.obs_mean = None

    @property
    def sensor_distrust(self):
        if "sensor_distrust" in self.params.keys():
            return np.repeat(self.params["sensor_distrust"], len(self.correlation))
        else:
            assert len(self.correlation.index) == len(set(self.correlation.index).intersection(
                self.params.keys())), "all the stations named in correlation should be in parameter keys."
            return np.array([self.params[c] for c in self.correlation.columns])  # the good ordering.

    def calibrate(self, observed_stations, observed_pollution: pd.DataFrame, traffic, **kwargs):
        observations = pd.concat((observed_pollution, kwargs["target_observation"]), axis=1)
        self.correlation = observations.corr()
        self.obs_mean = observations.mean(axis=0)
        super().calibrate(observed_stations, observed_pollution, traffic, **kwargs)

    def state_estimation(self, observed_stations, observed_pollution, traffic, target_positions: pd.DataFrame,
                         **kwargs) -> np.ndarray:
        know_stations = observed_pollution.columns
        unknown_stations = target_positions.columns

        correlation = self.correlation.copy()
        # distrust should be positive otherwise it means there is more confidence than 0-noise case
        correlation.values[np.diag_indices(len(self.correlation))] += np.abs(self.sensor_distrust)
        c = correlation.loc[know_stations, know_stations]
        b = correlation.loc[know_stations, unknown_stations].values

        a = np.linalg.solve(c, b)
        # a = Ridge(alpha=1.0, tol=0.1, fit_intercept=False).fit(c, b).coef_.T
        # a = np.linalg.lstsq(c, b, rcond=-1)[0]
        # a = scipy.linalg.solve(c, b, assume_a='pos')  # singular matrix
        return (observed_pollution - self.obs_mean[know_stations]).values @ a + self.obs_mean[unknown_stations].values[
                                                                                np.newaxis, :]
