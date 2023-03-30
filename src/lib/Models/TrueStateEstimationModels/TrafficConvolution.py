import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from src.lib.DataProcessing.TrafficProcessing import TRAFFIC_VALUES
from src.lib.Models.BaseModel import BaseModel, NONE_OPTIM_METHOD, mse, loo
from src.lib.Models.TrueStateEstimationModels.AverageModels import SummaryModel
from src.performance_utils import partial_filter, filter_dict


def gaussker(dist, sigma):
    return np.exp(-dist / sigma ** 2)


def reduce_traffic_to_colors(traffic: pd.DataFrame, kernel=1):
    return pd.DataFrame.from_dict({color: (kernel * (traffic == v)).sum(axis=1) for color, v in TRAFFIC_VALUES.items()})


class TrafficMeanModel(SummaryModel):
    def __init__(self, summary_statistic="mean", **kwargs):
        super(TrafficMeanModel, self).__init__(summary_statistic=summary_statistic)
        assert len(set(kwargs.keys()).union(TRAFFIC_VALUES.keys())) == len(TRAFFIC_VALUES), \
            f"parameters for {self} are {TRAFFIC_VALUES.keys()}, instead {kwargs.keys()} given"
        self.set_params(**kwargs)

    def calibrate(self, observed_stations, observed_pollution, traffic, **kwargs):
        summary_function = getattr(np, 'nan' + self.summary_statistic)
        average_pollution_by_time = summary_function(observed_pollution.values, axis=1)

        reduced_traffic = reduce_traffic_to_colors(traffic, kernel=1)
        lr = LinearRegression(fit_intercept=False).fit(reduced_traffic.values, average_pollution_by_time)
        self.set_params(**dict(zip(reduced_traffic.columns, lr.coef_)))
        return self

    def state_estimation(self, observed_stations, observed_pollution, traffic, target_positions,
                         **kwargs) -> np.ndarray:
        reduced_traffic = reduce_traffic_to_colors(traffic, kernel=1)
        estimated_average_pollution = reduced_traffic @ pd.Series(self.params)
        return estimated_average_pollution.values[:, np.newaxis] * np.ones((1, np.shape(target_positions)[1]))


class TrafficConvolutionModel(BaseModel):
    def __init__(self, conv_kernel=gaussker, normalize=False, loss=mse, optim_method=NONE_OPTIM_METHOD, niter=1000,
                 verbose=False, **kwargs):
        self.conv_kernel = conv_kernel
        self.normalize = normalize
        for k in TRAFFIC_VALUES.keys():
            if k not in kwargs.keys():
                kwargs[k] = 0
        super(TrafficConvolutionModel, self).__init__(
            name=f"{conv_kernel.__name__}{'norma' if normalize else ''}",
            loss=loss,
            optim_method=optim_method,
            niter=niter,
            verbose=verbose,
            **kwargs
        )
        # assert len(set(kwargs.keys()).union(TRAFFIC_VALUES.keys())) == len(TRAFFIC_VALUES), \
        #     f"parameters for {self} are {TRAFFIC_VALUES.keys()}, instead {kwargs.keys()} given"
        # self.set_params(**kwargs)

    def convolve(self, traffic, target_positions, traffic_coords) -> pd.DataFrame:
        """
        traffic_coords: pd.DataFrame with columns the pixel_coord and rows 'lat' and 'long' associated to the pixel
        target_positions: pd.DataFrame with columns the name of the station and rows 'lat' and 'long'
        """
        dist = np.sqrt(
            ((traffic_coords.loc[["long", "lat"], :] -
              target_positions.loc[["long", "lat"], :].values.reshape((2, 1)))
             ** 2).sum())
        kernel = partial_filter(self.conv_kernel, dist=dist, **self.params)
        reduced_traffic = reduce_traffic_to_colors(traffic, kernel=kernel)
        if self.normalize:
            reduced_traffic /= kernel.sum(axis=1)
        return reduced_traffic

    def state_estimation(self, observed_stations, observed_pollution, traffic, target_positions:pd.DataFrame,
                         traffic_coords, **kwargs) -> np.ndarray:
        """
        traffic_coords: pd.DataFrame with columns the pixel_coord and rows 'lat' and 'long' associated to the pixel
        target_positions: pd.DataFrame with columns the name of the station and rows 'lat' and 'long'
        """
        estimated_average_pollution = []
        for name, tp in target_positions.items():
            reduced_traffic = self.convolve(traffic=traffic, target_positions=pd.DataFrame(tp),
                                            traffic_coords=traffic_coords)
            estimated_average_pollution.append(reduced_traffic @ pd.Series(filter_dict(reduced_traffic.columns, **self.params)))
        return pd.concat(estimated_average_pollution, axis=1).values

    def state_estimation_for_optim(self, observed_stations, observed_pollution, traffic, **kwargs) -> [np.ndarray,
                                                                                                       np.ndarray]:
        """
        traffic_coords: pd.DataFrame with columns the pixel_coord and rows 'lat' and 'long' associated to the pixel
        target_positions: pd.DataFrame with columns the name of the station and rows 'lat' and 'long'
        """
        target_pollution, reduced_traffic = \
            list(zip(*[(target_pollution, partial_filter(self.convolve, **{**kwargs, **known_data}))
                       for known_data, target_pollution in loo(observed_stations, observed_pollution, traffic)]))
        target_pollution = pd.concat(target_pollution)
        reduced_traffic = pd.concat(reduced_traffic)

        lr = LinearRegression(fit_intercept=False).fit(reduced_traffic.values, target_pollution)
        self.set_params(**dict(zip(reduced_traffic.columns, lr.coef_)))
        estimated_average_pollution = reduced_traffic @ pd.Series(filter_dict(reduced_traffic.columns, **self.params))
        return target_pollution, estimated_average_pollution.values
