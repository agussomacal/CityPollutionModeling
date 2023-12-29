from typing import List

import cma
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.pipeline import Pipeline
from tqdm import tqdm

from src.lib.Modules import Bounds, Optim

CMA = "cma"
GRAD = "bfgs"
BFGS_PAR_OPTIM_METHOD = "bfgs_parallel"
RANDOM = "random"
UNIFORM = "uniform"
LOGUNIFORM = "loguniform"
NONE_OPTIM_METHOD = None


def split_by_station(unknown_station, observed_stations, observed_pollution, traffic, filterna=True):
    known_stations = observed_stations.columns.to_list()
    known_stations.remove(unknown_station)
    valid_times = ~observed_pollution[unknown_station].isna() if filterna else observed_pollution.index
    if len(valid_times) > 0:
        return {"observed_stations": observed_stations[known_stations],
                "observed_pollution": observed_pollution.loc[valid_times, known_stations],
                "traffic": traffic.loc[valid_times, :],
                "target_positions": observed_stations[[unknown_station]]}, \
            observed_pollution.loc[valid_times, unknown_station]


def split_loo(observed_stations):
    for unknown_station_ix in range(observed_stations.shape[1]):
        yield unknown_station_ix


def loo(observed_stations, observed_pollution, traffic, stations2test=None, filterna=True):
    for unknown_station_ix in split_loo(observed_stations):
        unknown_station = observed_stations.columns[unknown_station_ix]
        if stations2test is None or unknown_station in stations2test:
            split = split_by_station(unknown_station, observed_stations, observed_pollution, traffic, filterna)
            if split is not None:
                yield split


def pollution_agnostic(state_estimation4optim):
    """
    LOO for models that don't make use of pollution information (like traffic average).
    """

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


def apply_threshold(values, lower_threshold, upper_threshold):
    values[values < lower_threshold] = lower_threshold
    values[values > upper_threshold] = upper_threshold
    return values


def filter_by_quantile(values, lower_quantile, upper_quantile):
    q = np.quantile(values, [lower_quantile, upper_quantile], axis=0)
    return values[(values > q[0]) & (values < q[1])]


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
            dictparams = dict(zip(self.bounds, params))
            if self.verbose:
                df = pd.DataFrame(np.reshape(list(dictparams.values()), (1, -1)), columns=list(dictparams.keys()))
                print(f"Params for optimization: \n {df}")
            self.set_params(**dictparams)
            predicted_pollution, target_pollution = self.state_estimation_for_optim(
                observed_stations, observed_pollution, traffic, **kwargs)
            loss = self.loss(predicted_pollution, target_pollution)
            self.losses[tuple(params)] = loss
            if self.verbose:
                print(f"loss: {loss}")
            return loss

        self.losses = dict()
        x0 = np.array([self.params[k] for k in self.bounds])
        if self.niter == 1 or len(self.bounds) == 0:
            optim_func(x0)
            optim_params = x0
        elif self.optim_method == CMA:
            optim_params, _ = cma.fmin2(objective_function=optim_func, x0=x0,
                                        sigma0=self.sigma0, eval_initial_x=True,
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
        print(self, "Optim params: ", self.params)

        # dbvsuvbsd
        self.calibrated = True
        return self


class ModelsSequenciator(BaseModel):
    def __init__(self, models: List[BaseModel], transition_model: List[Pipeline], name=None, **kwargs):
        super().__init__(name, **kwargs)
        self.name = name
        self.models = models
        self.transition_model = transition_model
        self.TRUE_MODEL = np.all([model.TRUE_MODEL for model in models])
        self.amp = [1.0] * len(models)
        self.losses = list()
        self.calibrated = False

    @property
    def params(self):
        return {str(model): model.params for model in self.models}

    def __str__(self):
        return '_'.join([str(model) for model in self.models]) if self.name is None else self.name

    def single_state_estimation(self, observed_stations, observed_pollution, traffic, target_positions: pd.DataFrame,
                                **kwargs) -> np.ndarray:
        predictions_i = np.zeros((len(observed_pollution), np.shape(target_positions)[1]))
        observed_pollution_i = observed_pollution.copy()
        observed_stations_i = observed_stations.copy()
        for i, model in enumerate(self.models):
            predictions_i += self.transition_model[i].predict(
                model.state_estimation(observed_stations=observed_stations_i,
                                       observed_pollution=observed_pollution_i,
                                       traffic=traffic,
                                       target_positions=target_positions,
                                       **kwargs).reshape((-1, 1))).reshape(np.shape(predictions_i))
            # get the residuals on the observed values
            # the following lines are necessary for models that relly on the names of the sensors and are not properly
            # state estimation methods.
            # only actualize if it is not the las model
            if observed_pollution_i is not None and i < len(self.models) - 1:
                observed_pollution_i -= self.transition_model[i].predict(
                    model.state_estimation(observed_stations=observed_stations_i,
                                           observed_pollution=observed_pollution_i,
                                           traffic=traffic,
                                           target_positions=observed_stations_i,
                                           **kwargs).reshape((-1, 1))).reshape(np.shape(observed_pollution_i))
        return predictions_i

    def state_estimation(self, observed_stations, observed_pollution, traffic, target_positions: pd.DataFrame,
                         **kwargs) -> np.ndarray:
        # the prediction is the average of predictions
        return np.nanmean([self.single_state_estimation(
            observed_stations=known_data["observed_stations"],
            observed_pollution=known_data["observed_pollution"],
            traffic=known_data["traffic"],
            target_positions=target_positions, **kwargs) for known_data, target_pollution in
            loo(observed_stations, observed_pollution, traffic, filterna=False)], axis=0)

    @pollution_dependent
    def state_estimation_for_optim(self, observed_stations, observed_pollution, traffic, **kwargs) -> [np.ndarray,
                                                                                                       np.ndarray]:
        return self.single_state_estimation(observed_stations, observed_pollution, traffic, **kwargs)

    def calibrate(self, observed_stations, observed_pollution: pd.DataFrame, traffic, **kwargs):
        observed_pollution_i = observed_pollution.copy()
        for i, model in enumerate(self.models):
            # model.calibrate(observed_stations, observed_pollution, traffic, **kwargs)
            model.calibrate(observed_stations, observed_pollution_i, traffic, **kwargs)
            self.losses.append(model.losses)

            preds_i = pd.concat(
                [pd.DataFrame(model.state_estimation(**known_data, **kwargs),
                              index=known_data["observed_pollution"].index,
                              columns=[target_pollution.name])
                 for known_data, target_pollution in
                 loo(observed_stations, observed_pollution_i, traffic)], axis=1)
            X = preds_i.values.reshape((-1, 1))
            y = observed_pollution.values.reshape((-1, 1))
            mask = np.all(~np.isnan(X), axis=1) * np.all(~np.isnan(y), axis=1)
            self.transition_model[i].fit(X[mask], y[mask])

            if i < len(self.models) - 1:  # only actualize if it is not the las model
                mask = np.all(~np.isnan(X), axis=1)
                observed_pollution_i.values[mask.reshape(np.shape(observed_pollution_i))] -= self.transition_model[
                    i].predict(X[mask]).ravel()
        self.calibrated = True
        return self


class ModelsAggregator(BaseModel):
    def __init__(self, models: List[BaseModel], aggregator: Pipeline, name=None):
        super().__init__()
        self.losses = list()
        self.name = name
        self.TRUE_MODEL = np.all([model.TRUE_MODEL for model in models])
        self.aggregator = aggregator
        self.models = models

    @property
    def weights(self):
        if hasattr(self.aggregator.steps[-1], "coef_") and hasattr(self.aggregator.steps[-1], "intercept_"):
            return np.ravel(np.append(self.aggregator.steps[-1].coef_, self.aggregator.steps[-1].intercept_))

    @property
    def model_importance(self):
        weights = self.weights
        if weights is not None:
            return {model_name: weight for model_name, weight in
                    zip(list(map(str, self.models)) + ["intercept"], np.abs(self.weights) / np.abs(self.weights).sum())}

    def __str__(self):
        models_names = ','.join([''.join(filter(lambda c: c.isupper(), str(model))) for model in
                                 self.models]) if self.name is None else self.name
        return f"{self.aggregator}({models_names})"

    def state_estimation(self, observed_stations, observed_pollution, traffic, target_positions: pd.DataFrame,
                         **kwargs) -> np.ndarray:
        individual_predictions = np.array([
            model.state_estimation(observed_stations, observed_pollution, traffic, target_positions, **kwargs).squeeze()
            for model in self.models]).T
        return self.aggregator.predict(individual_predictions)

    def state_estimation_for_each_model(self, observed_stations, observed_pollution, traffic, **kwargs) -> [np.ndarray,
                                                                                                            np.ndarray]:
        predicted_pollution, target_pollution = list(zip(*[
            model.state_estimation_for_optim(observed_stations, observed_pollution, traffic, **kwargs)
            for model in self.models]))
        return np.concatenate(predicted_pollution, axis=1), np.concatenate(target_pollution, axis=1)

    def calibrate(self, observed_stations, observed_pollution: pd.DataFrame, traffic, **kwargs):
        if not self.calibrated:
            # calibrate models
            for model in self.models:
                if not model.calibrated:
                    model.calibrate(observed_stations, observed_pollution, traffic, **kwargs)
                self.losses.append(model.losses)
            # find optimal weights for model averaging
            individual_predictions, target = \
                self.state_estimation_for_each_model(observed_stations, observed_pollution, traffic, **kwargs)
            assert np.all(target == target[:, [0]], axis=1).all(), "All targets have to be equal."
            self.aggregator.fit(individual_predictions, target.mean(axis=1, keepdims=True))
            self.calibrated = True
        print(f"Models importance: {self.model_importance}")
