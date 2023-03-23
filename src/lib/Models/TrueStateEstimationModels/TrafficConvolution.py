import datetime
from typing import List, Dict, Union
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

from src.lib.Models.BaseModel import BaseModel

from src.lib.Models.TrueStateEstimationModels.AverageModels import SummaryModel

from src.lib.DataProcessing.TrafficProcessing import TRAFFIC_VALUES


def reduce_traffic_to_colors(traffic: pd.DataFrame, kernel=1):
    return pd.DataFrame.from_dict({color: (kernel * (traffic == v)).sum(axis=1) for color, v in TRAFFIC_VALUES.items()})


class TrafficMeanModel(SummaryModel):
    def __init__(self, summary_statistic="mean", **kwargs):
        super(TrafficMeanModel, self).__init__(summary_statistic=summary_statistic)
        assert len(set(kwargs.keys()).union(TRAFFIC_VALUES.keys())) == len(TRAFFIC_VALUES), \
            f"parameters for {self} are {TRAFFIC_VALUES.keys()}, instead {kwargs.keys()} given"
        self.set_params(**kwargs)

    def calibrate(self, observed_pollution, traffic, target_positions, **kwargs):
        summary_function = getattr(np, 'nan' + self.summary_statistic)
        average_pollution_by_time = summary_function(observed_pollution.values, axis=1)

        reduced_traffic = reduce_traffic_to_colors(traffic, kernel=1)
        lr = LinearRegression(fit_intercept=False).fit(reduced_traffic.values, average_pollution_by_time)
        self.set_params(**dict(zip(reduced_traffic.columns, lr.coef_)))
        return self

    def state_estimation(self, observed_pollution, traffic, target_positions, **kwargs) -> np.ndarray:
        reduced_traffic = reduce_traffic_to_colors(traffic, kernel=1)
        estimated_average_pollution = reduced_traffic @ pd.Series(self.params)
        return estimated_average_pollution.values[:, np.newaxis] * np.ones((1, np.shape(target_positions)[1]))
