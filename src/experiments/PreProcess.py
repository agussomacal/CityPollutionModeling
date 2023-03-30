import os.path
import os.path
import time

import joblib
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.spatial.distance import cdist

import src.config as config
from src.DataManager import DataManager
from src.experiments.config_experiments import screenshot_period, recalculate_traffic_by_pixel, nrows2load_traffic_data, \
    proportion_of_past_times, shuffle, server
from src.lib.DataProcessing.PollutionPreprocess import get_pollution, get_stations_lat_long, filter_pollution_dates
from src.lib.DataProcessing.Prepare4Experiments import get_traffic_pollution_data_per_hour
from src.lib.DataProcessing.TrafficProcessing import save_load_traffic_by_pixel_data, get_traffic_pixel_coords, \
    load_background
from src.lib.Models.BaseModel import BaseModel, split_by_station, Bounds
from src.performance_utils import timeit
from src.viz_utils import save_fig


def plot_stations_in_map(background, station_coordinates, lat, long):
    plt.close("all")
    plt.imshow(background)

    x = [np.argmin((l - long[0, :]) ** 2) for l in station_coordinates.T.long]
    y = [np.argmin((l - lat[:, 0]) ** 2) for l in station_coordinates.T.lat]
    plt.scatter(x, y, s=25, c="r", marker="x", edgecolors="k")
    for pos_x, pos_y, station_name in zip(x, y, station_coordinates.columns):
        plt.text(pos_x + 25, pos_y + 25, station_name, {'size': 7, "color": "red"})
    plt.tight_layout()


def plot_pairwise_info(pirewise_info):
    plt.figure(figsize=np.shape(pirewise_info))
    # mask = np.zeros_like(pirewise_info, dtype=np.bool)
    # mask[np.tril_indices_from(mask)] = True
    sns.heatmap(pirewise_info, annot=True, fmt=".2f", linewidth=.5, cmap="viridis",
                square=True)  # , mask=mask)
    plt.tight_layout()


def split_data_in_time(traffic_by_pixel, pollution, proportion_of_past_times, average=False, shuffle=False):
    traffic_per_hour, pollution_per_hour = get_traffic_pollution_data_per_hour(traffic_by_pixel, pollution, average)
    n_past = int(proportion_of_past_times * len(pollution_per_hour))
    print(f"Times in the past for training: {n_past}")
    print(f"Times in the future for testing: {len(pollution_per_hour) - n_past}")

    ix = np.random.choice(len(pollution_per_hour), replace=False) if shuffle else np.arange(len(pollution_per_hour))
    ix = np.array(ix, dtype=int)

    traffic_past = traffic_per_hour.iloc[ix[:n_past]]
    traffic_future = traffic_per_hour.iloc[ix[n_past:]]

    pollution_past = pollution_per_hour.iloc[ix[:n_past]]
    pollution_future = pollution_per_hour.iloc[ix[n_past:]]
    return traffic_past, pollution_past, traffic_future, pollution_future


# ----- Setting data for experiment ----- #
experiment_name = "PreAnalysis"

data_manager = DataManager(
    path=config.results_dir,
    name=experiment_name
)

preprocessing_data_path = f"{data_manager.path}/preprocessing_data{nrows2load_traffic_data}{shuffle}.compressed"
if os.path.exists(preprocessing_data_path):
    with timeit("Time loading pre-processed data:"):
        traffic_past, pollution_past, traffic_future, pollution_future, station_coordinates = joblib.load(
            preprocessing_data_path)
else:
    station_coordinates = get_stations_lat_long()
    traffic_by_pixel = save_load_traffic_by_pixel_data(
        screenshot_period=screenshot_period, recalculate=recalculate_traffic_by_pixel,
        nrows2load_traffic_data=nrows2load_traffic_data, workers=1, chunksize=None)
    pollution = get_pollution(date_start=traffic_by_pixel.index.min(), date_end=traffic_by_pixel.index.max())
    latitudes, longitudes, traffic_pixels_coords = get_traffic_pixel_coords(screenshot_period, traffic_by_pixel)
    pollution, station_coordinates = filter_pollution_dates(pollution, station_coordinates, traffic_by_pixel,
                                                            traffic_pixels_coords)

    # ----- split data in time to train and test -----
    traffic_past, pollution_past, traffic_future, pollution_future = \
        split_data_in_time(traffic_by_pixel, pollution, proportion_of_past_times)
    with timeit("Time saving pre-processed data"):
        joblib.dump((traffic_past, pollution_past, traffic_future, pollution_future, station_coordinates),
                    filename=preprocessing_data_path)

latitudes, longitudes, traffic_pixels_coords = get_traffic_pixel_coords(screenshot_period, traffic_future)
lat_bounds = Bounds(np.min(latitudes), np.max(latitudes))
long_bounds = Bounds(np.min(longitudes), np.max(longitudes))
longer_distance = np.sqrt(np.diff(lat_bounds) ** 2 + np.diff(long_bounds) ** 2)

distance_between_stations_pixels = pd.DataFrame(cdist(station_coordinates.loc[["long", "lat"],:].values.T,
                                                       traffic_pixels_coords.loc[["long", "lat"],:].values.T),
                                                columns=traffic_pixels_coords.columns,
                                                index=station_coordinates.columns)


# ----- Defining Experiment ----- #
def train_test_model(model: BaseModel):
    def decorated_func(station):
        # in train time use the past
        data_known, _ = split_by_station(unknown_station=station, observed_stations=station_coordinates,
                                         observed_pollution=pollution_past, traffic=traffic_past)
        data_known.pop("target_positions")  # this are reserved only when testing.
        t0 = time.time()
        model.calibrate(**data_known, traffic_coords=traffic_pixels_coords,
                        distance_between_stations_pixels=distance_between_stations_pixels)
        t_to_fit = time.time() - t0
        # in test time use the future
        data_known, data_unknown = split_by_station(unknown_station=station, observed_stations=station_coordinates,
                                                    observed_pollution=pollution_future, traffic=traffic_future)

        t0 = time.time()
        estimation = model.state_estimation(**data_known, traffic_coords=traffic_pixels_coords,
                                            distance_between_stations_pixels=distance_between_stations_pixels)
        t_to_estimate = time.time() - t0

        return {
            "losses": model.losses,
            "model_params": model.params,
            "estimation": estimation,
            "error": ((estimation - data_unknown.values.ravel()) ** 2).ravel(),
            "time_to_fit": t_to_fit,
            "time_to_estimate": t_to_estimate,
        }

    decorated_func.__name__ = str(model)
    return decorated_func


if __name__ == "__main__":
    # ----- insight on the data -----
    with save_fig(data_manager.path, "AvailableStations_InPeriod.png"):
        plot_stations_in_map(load_background(screenshot_period), station_coordinates, latitudes, longitudes)

    distance_between_stations = pd.DataFrame(cdist(station_coordinates.values.T, station_coordinates.values.T),
                                             columns=station_coordinates.columns, index=station_coordinates.columns)
    with save_fig(data_manager.path, "StationsDistance.png"):
        plot_pairwise_info(distance_between_stations)
    with save_fig(data_manager.path, "StationsCorrelation.png"):
        plot_pairwise_info(pollution_past.corr())
    with save_fig(data_manager.path, "StationsCorrelationVSDistance.png"):
        plt.scatter(distance_between_stations.values.ravel(), pollution_past.corr().values.ravel())
