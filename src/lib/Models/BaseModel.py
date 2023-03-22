from collections import namedtuple

import pandas as pd
from sklearn.model_selection import LeaveOneOut
import cma
import numpy as np
from scipy.optimize import minimize

CMA_OPTIM_METHOD = "cma"
BFGS_OPTIM_METHOD = "bfgs"
BFGS_PAR_OPTIM_METHOD = "bfgs_parallel"
NONE_OPTIM_METHOD = None

Bounds = namedtuple("Bounds", "lower upper")


def split_by_station(unknown_station, observed_stations, observed_pollution, traffic):
    known_stations = observed_stations.columns.to_list()
    known_stations.remove(unknown_station)
    valid_times = ~observed_pollution[unknown_station].isna()
    if len(valid_times) > 0:
        return {"observed_stations": observed_stations[known_stations],
                "observed_pollution": observed_pollution.loc[valid_times, known_stations],
                "traffic": traffic.loc[valid_times, :],
                "target_positions": observed_stations[[unknown_station]]}, \
            observed_pollution.loc[valid_times, unknown_station]


def loo(observed_stations, observed_pollution, traffic):
    loo = LeaveOneOut()
    for _, unknown_station_ix in loo.split(observed_stations.columns):
        unknown_station = observed_stations.columns[unknown_station_ix].to_list().pop()
        split = split_by_station(unknown_station, observed_stations, observed_pollution, traffic)
        if split is not None:
            yield split


def mse(x, y):
    """

    @param x: vector 1
    @param y: vector 2
    @return:
    """

    return np.nanmean((x - y) ** 2)


class BaseModel:
    TRUE_MODEL = True

    def __init__(self, name="", loss=mse, optim_method=NONE_OPTIM_METHOD, verbose=False, **kwargs):
        self.name = name
        self.loss = loss
        self.optim_method = optim_method
        self.verbose = verbose
        self._params = dict()
        self.bounds = dict()
        for k, v in kwargs.items():
            if v is None or isinstance(v, Bounds):
                self.bounds[k] = v
            else:
                self.set_params(**{k: v})

    @property
    def params(self):
        return self._params

    def set_params(self, **kwargs):
        self._params.update(kwargs)

    def __str__(self):
        return str(self.__class__.__name__) + self.name

    def state_estimation(self, observed_stations, observed_pollution, traffic, target_positions,
                         **kwargs) -> np.ndarray:
        raise Exception("Not implemented.")

    def state_estimation_for_optim(self, observed_stations, observed_pollution, traffic, target_positions,
                                   **kwargs) -> np.ndarray:
        return self.state_estimation(
            observed_stations=observed_stations,
            observed_pollution=observed_pollution,
            traffic=traffic,
            target_positions=target_positions,
            **kwargs
        )

    def calibrate(self, observed_stations, observed_pollution: pd.DataFrame, traffic, **kwargs):
        if len(self.params) == 1 and self.optim_method == CMA_OPTIM_METHOD:
            self.optim_method = BFGS_OPTIM_METHOD

        def optim_func(params):
            if self.verbose:
                print("Params for optimization: ", params)
            self.set_params(**dict(zip(self.params.keys(), optim_params)))
            target_pollution, predicted_pollution = \
                list(zip(*[(target_pollution, self.state_estimation_for_optim(**known_data, **kwargs))
                           for known_data, target_pollution in loo(observed_stations, observed_pollution, traffic)]))

            loss = self.loss(np.concatenate(predicted_pollution, axis=0), np.concatenate(target_pollution, axis=0))
            if self.verbose:
                print(loss)
            return loss

        x0 = np.array(list([self.params[k] for k in self.bounds]))
        if self.optim_method == CMA_OPTIM_METHOD:
            optim_params, _ = cma.fmin2(objective_function=optim_func, x0=x0,
                                        sigma0=1, eval_initial_x=True,
                                        options={'popsize': 10, 'maxfevals': 1000})
        elif self.optim_method == BFGS_OPTIM_METHOD:
            optim_params = minimize(fun=optim_func, x0=x0, bounds=self.bounds,
                                    method="L-BFGS-B" if all([b is None for b in self.bounds.values()]) else 'SLSQP',
                                    options={'maxiter': 1000}).x
        elif self.optim_method == NONE_OPTIM_METHOD:
            return None
        else:
            raise Exception("Not implemented.")
        print("Optim params: ", self.params)
        self.set_params(**dict(zip(self.params.keys(), optim_params)))
        return self
