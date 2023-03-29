import logging
from collections import namedtuple
from typing import List, Union, Dict

import cma
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.model_selection import LeaveOneOut
from tqdm import tqdm

from src.performance_utils import if_true_str

CMA_OPTIM_METHOD = "cma"
BFGS_OPTIM_METHOD = "bfgs"
BFGS_PAR_OPTIM_METHOD = "bfgs_parallel"
RANDOM = "random"
UNIFORM = "uniform"
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

    def __init__(self, name="", loss=mse, optim_method=NONE_OPTIM_METHOD, niter=1000, verbose=False,
                 **kwargs):
        self.name = name
        self.loss = loss
        self.niter = niter
        self.optim_method = optim_method
        self.verbose = verbose
        self._params = dict()
        self.bounds = dict()
        for k, v in kwargs.items():
            if v is None or isinstance(v, Bounds):
                self.bounds[k] = v
                # self.set_params(**{k: np.mean(v)})  # starting value the center
                self.set_params(**{k: v.lower})  # starting value the center
            else:
                self.set_params(**{k: v})
        self.losses = dict()

    @property
    def params(self):
        return self._params

    def set_params(self, **kwargs):
        self._params.update(kwargs)

    def __str__(self):
        return str(self.__class__.__name__) + self.name

    def state_estimation(self, observed_stations, observed_pollution, traffic, target_positions: pd.DataFrame,
                         **kwargs) -> np.ndarray:
        """
        target_positions: pd.DataFrame with columns the name of the station and rows 'lat' and 'long'
        """
        raise Exception("Not implemented.")

    def state_estimation_for_optim(self, observed_stations, observed_pollution, traffic, **kwargs) -> [np.ndarray,
                                                                                                       np.ndarray]:
        target_pollution, predicted_pollution = \
            list(zip(*[(target_pollution,
                        self.state_estimation(**known_data, target_pollution=target_pollution, **kwargs))
                       for known_data, target_pollution in loo(observed_stations, observed_pollution, traffic)]))

        return np.concatenate(predicted_pollution, axis=0), np.concatenate(target_pollution, axis=0)

    def calibrate(self, observed_stations, observed_pollution: pd.DataFrame, traffic, **kwargs):
        if len(self.params) == 1 and self.optim_method == CMA_OPTIM_METHOD:
            self.optim_method = BFGS_OPTIM_METHOD

        def optim_func(params):
            if self.verbose:
                logging.info(f"Params for optimization: {params}")
            self.set_params(**dict(zip(self.params, params)))
            target_pollution, predicted_pollution = self.state_estimation_for_optim(
                observed_stations, observed_pollution, traffic, **kwargs)
            # target_pollution, predicted_pollution = \
            #     list(zip(*[(target_pollution,
            #                 self.state_estimation_for_optim(**known_data, target_pollution=target_pollution, **kwargs))
            #                for known_data, target_pollution in loo(observed_stations, observed_pollution, traffic)]))

            loss = self.loss(target_pollution, predicted_pollution)
            # self.losses[tuple(params)] = loss
            if self.verbose:
                logging.info(loss)
            return loss

        if self.optim_method == CMA_OPTIM_METHOD:
            raise Exception("Not implemente, bounds or params?")
            x0 = np.array(list([self.params[k] for k in self.bounds]))
            optim_params, _ = cma.fmin2(objective_function=optim_func, x0=x0,
                                        sigma0=1, eval_initial_x=True,
                                        options={'popsize': 10, 'maxfevals': self.niter})
            self.set_params(**dict(zip(self.params.keys(), optim_params)))
        elif self.optim_method == BFGS_OPTIM_METHOD:
            raise Exception("Not implemente, bounds or params?")
            x0 = np.array(list([self.params[k] for k in self.bounds]))
            optim_params = minimize(fun=optim_func, x0=x0, bounds=self.bounds,
                                    method="L-BFGS-B" if all([b is None for b in self.bounds.values()]) else 'SLSQP',
                                    options={'maxiter': self.niter}).x

            self.set_params(**dict(zip(self.params.keys(), optim_params)))
        elif self.optim_method in [RANDOM, UNIFORM]:
            sampler = np.linspace if UNIFORM else np.random.uniform
            samples = {k: sampler(*bounds, self.niter).ravel().tolist() for k, bounds in self.bounds.items() if
                       bounds is not None}
            self.losses = {x: optim_func(list({**self.params, **dict(zip(samples.keys(), x))}.values())) for x in
                           tqdm(zip(*samples.values()), desc=f"Training {self}")}
            best_ix = np.argmin(self.losses.values())
            self.set_params(**{k: v[best_ix] for k, v in samples.items()})

        elif self.optim_method == NONE_OPTIM_METHOD:
            return None
        else:
            raise Exception("Not implemented.")
        logging.info("Optim params: ", self.params)
        return self


class ModelsSequenciator:
    def __init__(self, models: List[BaseModel], name=None):
        # assert all([isinstance(model, BaseModel) for model in models])
        self.name = name
        self.models = models
        self.TRUE_MODEL = np.all([model.TRUE_MODEL for model in models])

    @property
    def params(self):
        return {str(model): model.params for model in self.models}

    def __str__(self):
        return '_'.join([str(model) for model in self.models]) if self.name is None else self.name

    def state_estimation(self, observed_stations, observed_pollution, traffic, target_positions: pd.DataFrame,
                         **kwargs) -> np.ndarray:
        predictions = np.zeros((len(observed_pollution), np.shape(target_positions)[1]))
        observed_pollution_i = observed_pollution.copy()
        for i, model in enumerate(self.models):
            predictions += model.state_estimation(observed_stations, observed_pollution_i, traffic, target_positions,
                                                  **kwargs)
            # get the residuals on the observed values
            # the following lines are necessary for models that relly on the names of the sensors and are not properly
            # state stimation methods.
            # only actualize if it is not the las model
            if observed_pollution_i is not None and i < len(self.models) - 1:
                observed_pollution_i -= pd.concat([pd.DataFrame(model.state_estimation(**known_data, **kwargs),
                                                                index=known_data["observed_pollution"].index,
                                                                columns=[target_pollution.name])
                                                   for known_data, target_pollution in
                                                   loo(observed_stations, observed_pollution_i, traffic)], axis=1)
                # observed_pollution_i -= pd.DataFrame(
                #     model.state_estimation(observed_stations, observed_pollution_i, traffic,
                #                            observed_stations, **kwargs),
                #     index=observed_pollution_i.index,
                #     columns=observed_pollution_i.columns)

        # predictions[predictions < 0] = 0  # non negative values; pollution is positive quantity.
        return predictions

    def calibrate(self, observed_stations, observed_pollution: pd.DataFrame, traffic, **kwargs):
        observed_pollution_i = observed_pollution.copy()
        for i, model in enumerate(self.models):
            model.calibrate(observed_stations, observed_pollution, traffic, **kwargs)

            if i < len(self.models) - 1:  # only actualize if it is not the las model
                observed_pollution_i -= pd.concat([pd.DataFrame(model.state_estimation(**known_data, **kwargs),
                                                                index=known_data["observed_pollution"].index,
                                                                columns=[target_pollution.name])
                                                   for known_data, target_pollution in
                                                   loo(observed_stations, observed_pollution_i, traffic)], axis=1)
        return self


class ModelsAverager(BaseModel):
    def __init__(self, models: List[BaseModel], positive=False, fit_intercept=False,
                 weights: Union[Dict, List, np.ndarray] = None, name=None, n_alphas=None):
        super().__init__()
        self.name = name
        self.TRUE_MODEL = np.all([model.TRUE_MODEL for model in models])

        self.models = models
        self.n_alphas = n_alphas
        if n_alphas is None:
            self.lr = LinearRegression(positive=positive, fit_intercept=fit_intercept)
        else:
            self.lr = LassoCV(positive=positive, fit_intercept=fit_intercept, n_alphas=n_alphas)
        if weights is not None:
            if isinstance(weights, Dict):
                self.lr.coef_ = np.array([weights[str(model)] for model in self.models])
                self.lr.intercept_ = weights["intercept"]
            elif isinstance(weights, (List, np.ndarray)):
                self.lr.coef_ = np.array(weights[:-1])
                self.lr.intercept_ = weights[-1]

    @property
    def weights(self):
        return np.ravel(np.append(self.lr.coef_, self.lr.intercept_))

    @property
    def model_importance(self):
        return {model_name: weight for model_name, weight in
                zip(list(map(str, self.models)) + ["intercept"], np.abs(self.weights) / np.abs(self.weights).sum())}

    @property
    def params(self):
        return {**{str(model): model.params for model in self.models}, **{"weights": self.weights.tolist()}}

    def __str__(self):
        models_names = ','.join([''.join(filter(lambda c: c.isupper(), str(model))) for model in
                                 self.models]) if self.name is None else self.name
        return f"{if_true_str(self.lr.fit_intercept, 'c', '', '+')}{'LASSO' if self.n_alphas else 'LR'}{if_true_str(self.lr.positive, '+')}({models_names})"

    def state_estimation_concatenated(self, observed_stations, observed_pollution, traffic,
                                      target_positions: pd.DataFrame,
                                      **kwargs) -> np.ndarray:
        return np.concatenate([
            model.state_estimation(observed_stations, observed_pollution, traffic, target_positions, **kwargs).reshape(
                (-1, 1))
            for model in self.models], axis=1)

    def state_estimation(self, observed_stations, observed_pollution, traffic, target_positions: pd.DataFrame,
                         **kwargs) -> np.ndarray:
        individual_predictions = self.state_estimation_concatenated(observed_stations, observed_pollution, traffic,
                                                                    target_positions, **kwargs)
        return (individual_predictions @ self.lr.coef_ + self.lr.intercept_).reshape((len(observed_pollution), -1))

    def state_estimation_for_optim(self, observed_stations, observed_pollution, traffic, **kwargs) -> [np.ndarray,
                                                                                                       np.ndarray]:
        target_pollution, predicted_pollution = \
            list(zip(*[(target_pollution,
                        self.state_estimation_concatenated(**known_data, target_pollution=target_pollution, **kwargs))
                       for known_data, target_pollution in loo(observed_stations, observed_pollution, traffic)]))

        return np.concatenate(predicted_pollution, axis=0), np.concatenate(target_pollution, axis=0)

    def calibrate(self, observed_stations, observed_pollution: pd.DataFrame, traffic, **kwargs):
        if not hasattr(self.lr, "coefs_") or not hasattr(self.lr, "intercept_"):
            # calibrate models
            for model in self.models:
                model.calibrate(observed_stations, observed_pollution, traffic, **kwargs)
            # find optimal weights for model averaging
            individual_predictions, target = \
                self.state_estimation_for_optim(observed_stations, observed_pollution, traffic, **kwargs)
            not_nan = ~np.any(np.isnan(individual_predictions), axis=1) & ~np.isnan(target)
            self.lr.fit(individual_predictions[not_nan], target[not_nan])
        print(f"Models importance: {self.model_importance}")
