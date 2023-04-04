from datetime import datetime

import numpy as np
import pandas as pd

from src.lib.Models.BaseModel import BaseModel, mse


class CosinusModel(BaseModel):
    def __init__(self, t0: datetime, amplitude, period, phase, name="", loss=mse, optim_method="lsq",
                 verbose=False):
        self.t0 = t0
        super(CosinusModel, self).__init__(name=name, loss=loss, optim_method=optim_method, verbose=verbose)
        self.set_params(amplitude=amplitude, period=period, phase=phase)

    def calibrate(self, observed_stations, observed_pollution, traffic: pd.DataFrame, **kwargs):
        pass

    def state_estimation(self, observed_stations, observed_pollution, traffic, target_positions: pd.DataFrame,
                         **kwargs) -> np.ndarray:
        average_pollution = self.params["amplitude"] * np.cos(
            (observed_pollution.index - self.t0)/ np.timedelta64(1, 'h') * 2 * np.pi /
            self.params["period"] + self.params["phase"])
        return average_pollution.values[:, np.newaxis] * np.ones((1, np.shape(target_positions)[1]))
