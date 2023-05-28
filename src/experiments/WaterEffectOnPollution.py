from collections import defaultdict
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

from PerplexityLab.DataManager import DataManager
from PerplexityLab.visualization import save_fig
from src.config import results_dir
from src.experiments.PreProcess import latitudes, longitudes, station_coordinates, pollution_past, \
    pollution_future, longer_distance
from src.experiments.config_experiments import screenshot_period
from src.lib.DataProcessing.TrafficProcessing import load_background, road_color
from src.lib.FeatureExtractors.ConvolutionFeatureExtractors import WaterColor, GreenAreaColor
from src.lib.Models.TrueStateEstimationModels.TrafficConvolution import gaussker

if __name__ == "__main__":
    experiment_name = "WaterEffectOnPollution"

    data_manager = DataManager(
        path=results_dir,
        name=experiment_name,
        country_alpha_code="FR",
        trackCO2=True
    )

    with data_manager.track_emissions("PreprocessesTrafficPollution"):
        img = load_background(screenshot_period)
        np.sum(img == WaterColor)
        np.sum(img == GreenAreaColor)

        roads_mask = np.all(img == road_color, axis=-1)
        water_mask = np.all(img == WaterColor, axis=-1)
        greenarea_mask = np.all(img == GreenAreaColor, axis=-1)
        plt.imshow(roads_mask + 2 * water_mask + 3 * greenarea_mask, cmap="tab10")

        water_featurer = FEConvolutionFixedPixels(water_mask, longitudes, latitudes, metric="euclidean")
        water_features = pd.Series(
            water_featurer.by_convolution(points=station_coordinates.loc[["long", "lat"], :].values.T,
                                          kernel=partial(gaussker, sigma=0.1), agg_func=np.sum),
            index=station_coordinates.columns)

        greenarea_featurer = FEConvolutionFixedPixels(greenarea_mask, longitudes, latitudes, metric="euclidean")
        greenarea_features = pd.Series(
            greenarea_featurer.by_convolution(points=station_coordinates.loc[["long", "lat"], :].values.T,
                                              kernel=partial(gaussker, sigma=0.1), agg_func=np.sum),
            index=station_coordinates.columns)

    with save_fig(data_manager.path, "Water_effect_on_average_pollution.png"):
        plt.plot(water_features, pollution_past.mean()[water_features.index], "o", label="past")
        plt.plot(water_features, pollution_future.mean()[water_features.index], "o", label="future")
        plt.xlabel("Water feature")
        plt.ylabel("average of pollution")
        plt.legend()

    with save_fig(data_manager.path, "Water_effect_on_std_pollution.png"):
        plt.plot(water_features, pollution_past.std()[water_features.index], "o", label="past")
        plt.plot(water_features, pollution_future.std()[water_features.index], "o", label="future")
        plt.xlabel("Water feature")
        plt.ylabel("std of pollution")
        plt.legend()

    with save_fig(data_manager.path, "Water_effect_on_std_pollution.png"):
        plt.plot(water_features,
                 (pollution_past - pollution_past.mean(axis=1).values.reshape((-1, 1))).std()[water_features.index],
                 "o")
        # plt.plot(water_features, pollution_past.median()[water_features.index], "o")
        plt.xlabel("Water feature")
        plt.ylabel("std of pollution")

    # ---------- Models analysis ---------- #
    models = [LinearRegression(), Pipeline([("Poly", PolynomialFeatures(degree=5)), ("LR", LassoCV())]),
              MLPRegressor(hidden_layer_sizes=(5, 5), activation="logistic"), RandomForestRegressor()]
    correlation_madian = defaultdict(list)
    correlation_std = defaultdict(list)
    sigmas = np.linspace(0, longer_distance, 10)
    for sigma in sigmas:
        water_featurer = FEImageStaticConvolution(water_mask, longitudes, latitudes, metric="euclidean")
        water_features = pd.Series(
            water_featurer.by_convolution(points=station_coordinates.loc[["long", "lat"], :].values.T,
                                          kernel=partial(gaussker, sigma=sigma), agg_func=np.sum),
            index=station_coordinates.columns)

        greenarea_featurer = FEImageStaticConvolution(greenarea_mask, longitudes, latitudes, metric="euclidean")
        greenarea_features = pd.Series(
            greenarea_featurer.by_convolution(points=station_coordinates.loc[["long", "lat"], :].values.T,
                                              kernel=partial(gaussker, sigma=sigma), agg_func=np.sum),
            index=station_coordinates.columns)

        street_featurer = FEImageStaticConvolution(roads_mask, longitudes, latitudes, metric="euclidean")
        street_features = pd.Series(
            greenarea_featurer.by_convolution(points=station_coordinates.loc[["long", "lat"], :].values.T,
                                              kernel=partial(gaussker, sigma=sigma), agg_func=np.sum),
            index=station_coordinates.columns)

        # target_median = pollution_past.median()[water_features.index].values.reshape((-1, 1))
        target_median = pollution_past[water_features.index].values.reshape((-1, 1))
        query = np.concatenate([
            np.reshape([water_features] * len(pollution_past), (-1, 1)),
            np.reshape([greenarea_features] * len(pollution_past), (-1, 1)),
            np.reshape([street_features] * len(pollution_past), (-1, 1))
        ], axis=1)
        notnan_train = np.ravel(~np.isnan(target_median))
        # target_median_test = pollution_future.median()[water_features.index].values.reshape((-1, 1))
        target_median_test = pollution_future[water_features.index].values.reshape((-1, 1))
        query_test = np.concatenate([
            np.reshape([water_features] * len(pollution_future), (-1, 1)),
            np.reshape([greenarea_features] * len(pollution_future), (-1, 1)),
            np.reshape([street_features] * len(pollution_future), (-1, 1))
        ], axis=1)
        notnan_test = np.ravel(~np.isnan(target_median_test))

        target_std = pollution_past.std()[water_features.index].values.reshape((-1, 1))
        target_std_test = pollution_future.std()[water_features.index].values.reshape((-1, 1))
        query_std = np.transpose([water_features, greenarea_features, street_features])
        for model in models:
            model.fit(query[notnan_train, :], target_median[notnan_train])
            correlation_madian[str(model)].append(
                np.corrcoef(target_median_test[notnan_test].ravel(), model.predict(query_test[notnan_test, :]).ravel())[
                    0, 1])

            model.fit(query_std, target_std)
            correlation_std[str(model)].append(
                np.corrcoef(target_std_test.ravel(), model.predict(query_std).ravel())[0, 1])

    with save_fig(data_manager.path, "Models_correlation.png"):
        fig, ax = plt.subplots(ncols=2, figsize=(12, 8))
        for model, corr in correlation_madian.items():
            ax[0].plot(sigmas, corr, label=model)
        ax[0].set_xlabel("Convolution parameter")
        ax[0].set_ylabel("Correlation coefficient")
        ax[0].legend()

        for model, corr in correlation_std.items():
            ax[1].plot(sigmas, corr, label=model)
        ax[1].set_xlabel("Convolution parameter")
        ax[1].set_ylabel("Correlation coefficient")
        ax[1].legend()
