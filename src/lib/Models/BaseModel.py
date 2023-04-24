from collections import namedtuple
from typing import List, Union, Dict

import cma
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression, LassoCV
from tqdm import tqdm

from PerplexityLab.miscellaneous import if_true_str

CMA = "cma"
GRAD = "bfgs"
BFGS_PAR_OPTIM_METHOD = "bfgs_parallel"
RANDOM = "random"
UNIFORM = "uniform"
LOGUNIFORM = "loguniform"
NONE_OPTIM_METHOD = None

Bounds = namedtuple("Bounds", "lower upper")
Optim = namedtuple("Optim", "start lower upper", defaults=[0, None, None])


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


def split_loo(observed_stations):
    for unknown_station_ix in range(observed_stations.shape[1]):
        yield unknown_station_ix


def pollution_agnostic(state_estimation4optim):
    def decorated(self, observed_stations, observed_pollution, traffic, **kwargs) -> [np.ndarray, np.ndarray]:
        stations2test = kwargs.get("stations2test", observed_pollution.columns)
        order = [unknown_station_ix for unknown_station_ix in split_loo(observed_pollution) if
                 observed_pollution.columns[unknown_station_ix] in stations2test]

        se = state_estimation4optim(
            self,
            observed_stations=observed_stations.iloc[:, order],
            observed_pollution=observed_pollution.iloc[:, order],
            traffic=traffic,
            # target_positions=observed_stations.iloc[:, order],
            **kwargs
        )

        se = se.reshape((-1, 1), order="F")
        target = observed_pollution.iloc[:, order].values.reshape((-1, 1), order="F")
        valid_indexes = ~np.isnan(target)
        return se[valid_indexes, np.newaxis], target[valid_indexes, np.newaxis]

    return decorated


def loo(observed_stations, observed_pollution, traffic, stations2test=None):
    for unknown_station_ix in split_loo(observed_stations):
        unknown_station = observed_stations.columns[unknown_station_ix]
        if stations2test is None or unknown_station in stations2test:
            split = split_by_station(unknown_station, observed_stations, observed_pollution, traffic)
            if split is not None:
                yield split


def pollution_dependent(state_estimation4optim):
    def decorated(self, observed_stations, observed_pollution, traffic, **kwargs) -> [np.ndarray, np.ndarray]:
        target_pollution, predicted_pollution = \
            list(zip(*[(target_pollution,
                        state_estimation4optim(self, **known_data, target_pollution=target_pollution, **kwargs))
                       for known_data, target_pollution in
                       loo(observed_stations, observed_pollution, traffic, kwargs.get("stations2test", None))]))

        return np.concatenate(predicted_pollution, axis=0), np.concatenate(target_pollution, axis=0)[:, np.newaxis]

    return decorated


def mse(x, y):
    """

    @param x: vector 1
    @param y: vector 2
    @return:
    """

    return np.nanmean((x - y) ** 2)


def medianse(x, y):
    """

    @param x: vector 1
    @param y: vector 2
    @return:
    """

    return np.nanmedian((x - y) ** 2)


class BaseModel:
    TRUE_MODEL = True
    POLLUTION_AGNOSTIC = False

    def __init__(self, name="", loss=mse, optim_method=NONE_OPTIM_METHOD, niter=1000, verbose=False, sigma0=1,
                 **kwargs):
        self.name = name
        self.loss = loss
        self.niter = niter
        self.optim_method = optim_method
        self.sigma0 = sigma0  # for CMA
        self.verbose = verbose
        self._params = dict()
        self.bounds = dict()
        for k, v in kwargs.items():
            if isinstance(v, Optim):
                self.bounds[k] = Bounds(lower=v.lower, upper=v.upper)
                # self.set_params(**{k: np.mean(v)})  # starting value the center
                self.set_params(**{k: v.start})  # starting value the center
            else:
                self.set_params(**{k: v})
        self.losses = dict()
        self.calibrated = False

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

    @pollution_dependent
    def state_estimation_for_optim(self, observed_stations, observed_pollution, traffic, target_positions,
                                   **kwargs) -> [np.ndarray, np.ndarray]:
        return self.state_estimation(observed_stations, observed_pollution, traffic, target_positions, **kwargs)

    def calibrate(self, observed_stations, observed_pollution: pd.DataFrame, traffic, **kwargs):
        if len(self.params) == 1 and self.optim_method == CMA:
            self.optim_method = GRAD

        def optim_func(params):
            if self.verbose:
                print(f"Params for optimization: {params}")
            self.set_params(**dict(zip(self.bounds, params)))
            predicted_pollution, target_pollution = self.state_estimation_for_optim(
                observed_stations, observed_pollution, traffic, **kwargs)
            loss = self.loss(predicted_pollution, target_pollution)
            self.losses[tuple(params)] = loss
            if self.verbose:
                print(loss)
            return loss

        self.losses = dict()
        x0 = np.array([self.params[k] for k in self.bounds])
        if self.niter == 1 or len(self.bounds) == 0:
            optim_func(x0)
            optim_params = x0
        elif self.optim_method == CMA:
            optim_params, _ = cma.fmin2(objective_function=optim_func, x0=x0,
                                        sigma0=1, eval_initial_x=True,
                                        options={'popsize': 10, 'maxfevals': self.niter})
        elif self.optim_method == GRAD:
            optim_params = minimize(fun=optim_func, x0=x0, bounds=self.bounds.values(),
                                    method="L-BFGS-B", options={'maxiter': self.niter}).x

        elif self.optim_method in [RANDOM, UNIFORM, LOGUNIFORM]:
            if self.optim_method == UNIFORM:
                sampler = np.linspace
            elif self.optim_method == RANDOM:
                sampler = np.random.uniform
            elif self.optim_method == LOGUNIFORM:
                sampler = lambda d, u, i: np.logspace(np.log10(d), np.log10(u), i)
            else:
                raise Exception(f"Optim method {self.optim_method} not implemented.")
            samples = {k: sampler(bounds.lower, bounds.upper, self.niter).ravel().tolist() for k, bounds in
                       self.bounds.items() if
                       bounds is not None}
            self.losses = {x: optim_func(list({**self.params, **dict(zip(samples.keys(), x))}.values())) for x in
                           tqdm(zip(*samples.values()), desc=f"Training {self}")}
            best_ix = np.argmin(self.losses.values())
            optim_params = [v[best_ix] for v in samples.values()]
        elif self.optim_method == NONE_OPTIM_METHOD:
            return None
        else:
            raise Exception("Not implemented.")

        if len(self.bounds) > 0:
            self.set_params(**dict(zip(self.bounds.keys(), optim_params)))
            self.losses = pd.Series(self.losses.values(), pd.Index(self.losses.keys(), names=self.bounds.keys()),
                                    name="loss")
        print("Optim params: ", self.params)
        self.calibrated = True
        return self


class ModelsSequenciator(BaseModel):
    def __init__(self, models: List[BaseModel], name=None, **kwargs):
        super().__init__(name, **kwargs)
        self.name = name
        self.models = models
        self.TRUE_MODEL = np.all([model.TRUE_MODEL for model in models])
        self.amp = [1.0] * len(models)
        self.losses = list()
        self.calibrated = False

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
            preds_i = self.amp[i] * model.state_estimation(observed_stations, observed_pollution_i, traffic,
                                                           target_positions,
                                                           **kwargs)
            predictions += preds_i
            # get the residuals on the observed values
            # the following lines are necessary for models that relly on the names of the sensors and are not properly
            # state stimation methods.
            # only actualize if it is not the las model
            if observed_pollution_i is not None and i < len(self.models) - 1:
                observed_pollution_i -= preds_i
                # observed_pollution_i -= self.amp[i] * pd.concat(
                #     [pd.DataFrame(model.state_estimation(**known_data, **kwargs),
                #                   index=known_data["observed_pollution"].index,
                #                   columns=[target_pollution.name])
                #      for known_data, target_pollution in
                #      loo(observed_stations, observed_pollution_i, traffic)], axis=1)
                # observed_pollution_i -= pd.DataFrame(
                #     model.state_estimation(observed_stations, observed_pollution_i, traffic,
                #                            observed_stations, **kwargs),
                #     index=observed_pollution_i.index,
                #     columns=observed_pollution_i.columns)

        # predictions[predictions < 0] = 0  # non negative values; pollution is positive quantity.
        return predictions

    @pollution_agnostic
    def state_estimation_for_optim(self, observed_stations, observed_pollution, traffic, **kwargs) -> [np.ndarray,
                                                                                                       np.ndarray]:
        return self.state_estimation(observed_stations, observed_pollution, traffic,
                                     target_positions=observed_stations, **kwargs)
        # return reorder4agnostic(state_estimation, observed_pollution, kwargs.get("stations2test", None))

    def calibrate(self, observed_stations, observed_pollution: pd.DataFrame, traffic, **kwargs):
        observed_pollution_i = observed_pollution.copy()
        for i, model in enumerate(self.models):
            # model.calibrate(observed_stations, observed_pollution, traffic, **kwargs)
            model.calibrate(observed_stations, observed_pollution_i, traffic, **kwargs)
            self.losses.append(model.losses)
            self.amp[i] = 1 if i == 0 else \
                np.nanmedian((observed_pollution_i.values / model.state_estimation(observed_stations,
                                                                                   observed_pollution,
                                                                                   traffic,
                                                                                   observed_stations,
                                                                                   **kwargs)))

            if i < len(self.models) - 1:  # only actualize if it is not the las model
                observed_pollution_i -= self.amp[i] * model.state_estimation(observed_stations, observed_pollution,
                                                                             traffic, observed_stations, **kwargs)
                # observed_pollution_i -= self.amp[i]*pd.concat([pd.DataFrame(model.state_estimation(**known_data, **kwargs),
                #                                                 index=known_data["observed_pollution"].index,
                #                                                 columns=[target_pollution.name])
                #                                    for known_data, target_pollution in
                #                                    loo(observed_stations, observed_pollution_i, traffic)], axis=1)
        print("Amplitudes: ", self.amp)
        self.calibrated = True
        return self


class ModelsAverager(BaseModel):
    def __init__(self, models: List[BaseModel], positive=False, fit_intercept=False,
                 weights: Union[Dict, List, np.ndarray] = None, name=None, n_alphas=None):
        super().__init__()
        self.losses = list()
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

    def state_estimation(self, observed_stations, observed_pollution, traffic, target_positions: pd.DataFrame,
                         **kwargs) -> np.ndarray:
        individual_predictions = np.array([
            model.state_estimation(observed_stations, observed_pollution, traffic, target_positions, **kwargs)
            for model in self.models])
        return np.einsum("m...,m", individual_predictions, self.lr.coef_.ravel()) + self.lr.intercept_

    def state_estimation_for_each_model(self, observed_stations, observed_pollution, traffic, **kwargs) -> [np.ndarray,
                                                                                                            np.ndarray]:
        predicted_pollution, target_pollution = list(zip(*[
            model.state_estimation_for_optim(observed_stations, observed_pollution, traffic, **kwargs)
            for model in self.models]))
        return np.concatenate(predicted_pollution, axis=1), np.concatenate(target_pollution, axis=1)

    def calibrate(self, observed_stations, observed_pollution: pd.DataFrame, traffic, **kwargs):
        if not hasattr(self.lr, "coefs_") or not hasattr(self.lr, "intercept_"):
            # calibrate models
            for model in self.models:
                if not model.calibrated:
                    model.calibrate(observed_stations, observed_pollution, traffic, **kwargs)
                self.losses.append(model.losses)
            # find optimal weights for model averaging
            individual_predictions, target = \
                self.state_estimation_for_each_model(observed_stations, observed_pollution, traffic, **kwargs)
            assert np.all(target == target[:, [0]], axis=1).all(), "All targets have to be equal."
            self.lr.fit(individual_predictions, target.mean(axis=1, keepdims=True))
            self.calibrated = True
        print(f"Models importance: {self.model_importance}")
