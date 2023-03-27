import logging
import time

import numpy as np
import pandas as pd
import psutil
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.spatial.distance import cdist
from spiderplot import spiderplot

import src.config as config
from src.DataManager import DataManager
from src.LabPipeline import LabPipeline
from src.lib.DataProcessing.PollutionPreprocess import get_pollution, get_stations_lat_long, filter_pollution_dates
from src.lib.DataProcessing.Prepare4Experiments import get_traffic_pollution_data_per_hour
from src.lib.DataProcessing.TrafficProcessing import save_load_traffic_by_pixel_data, get_traffic_pixel_coords, \
    load_background
from src.lib.Models.BaseModel import BaseModel, split_by_station, Bounds, mse, UNIFORM
from src.lib.Models.TrueStateEstimationModels.AverageModels import SnapshotMeanModel, GlobalMeanModel
from src.lib.Models.TrueStateEstimationModels.TrafficConvolution import TrafficMeanModel, TrafficConvolutionModel, \
    gaussker
from src.viz_utils import save_fig, generic_plot


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
    mask = np.zeros_like(pirewise_info, dtype=np.bool)
    mask[np.tril_indices_from(mask)] = True
    sns.heatmap(pirewise_info, annot=True, fmt=".2f", linewidth=.5, cmap="viridis",
                square=True, mask=mask)
    plt.tight_layout()


def split_data_in_time(traffic_by_pixel, pollution, proportion_of_past_times, average=False):
    traffic_per_hour, pollution_per_hour = get_traffic_pollution_data_per_hour(traffic_by_pixel, pollution, average)
    n_past = int(proportion_of_past_times * len(pollution_per_hour))
    print(f"Times in the past for training: {n_past}")
    print(f"Times in the future for testing: {len(pollution_per_hour) - n_past}")

    traffic_past = traffic_per_hour[:n_past]
    traffic_future = traffic_per_hour[n_past:]

    pollution_past = pollution_per_hour[:n_past]
    pollution_future = pollution_per_hour[n_past:]
    return traffic_past, pollution_past, traffic_future, pollution_future


if __name__ == "__main__":
    experiment_name = "ModelComparison"

    data_manager = DataManager(
        path=config.results_dir,
        name=experiment_name
    )

    # Create and configure logger
    logging.basicConfig(
        level=logging.INFO,
        # handlers=[logging.FileHandler(f"{data_manager.path}/experiment.log"), logging.StreamHandler()],
        filename=f"{data_manager.path}/experiment.log",
        format='%(asctime)s %(message)s',
        filemode='w')
    # define a Handler which writes INFO messages or higher to the sys.stderr
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    # set a format which is simpler for console use
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    # tell the handler to use this format
    console.setFormatter(formatter)
    # add the handler to the root logger
    logging.getLogger().addHandler(console)

    sns.set_theme()

    # ----- parameters of experiment ----- #
    recalculate_traffic_by_pixel = False
    proportion_of_past_times = 0.8
    screenshot_period = 15

    RAM = psutil.virtual_memory().total / 1000000000
    if RAM > 100:  # if run in server
        nrows2load_traffic_data = None  # None 1000
        num_cores = 25
    else:
        nrows2load_traffic_data = 12  # None 1000
        num_cores = 10

    # ----- Setting data for experiment ----- #
    station_coordinates = get_stations_lat_long()
    traffic_by_pixel = save_load_traffic_by_pixel_data(
        screenshot_period=screenshot_period, recalculate=recalculate_traffic_by_pixel,
        nrows2load_traffic_data=nrows2load_traffic_data, workers=1, chunksize=None)
    pollution = get_pollution(date_start=traffic_by_pixel.index.min(), date_end=traffic_by_pixel.index.max())
    latitudes, longitudes, traffic_pixels_coords = get_traffic_pixel_coords(screenshot_period, traffic_by_pixel)
    pollution, station_coordinates = filter_pollution_dates(pollution, station_coordinates, traffic_by_pixel,
                                                            traffic_pixels_coords)
    lat_bounds = Bounds(np.min(latitudes), np.max(latitudes))
    long_bounds = Bounds(np.min(longitudes), np.max(longitudes))
    longer_distance = np.sqrt(np.diff(lat_bounds) ** 2 + np.diff(long_bounds) ** 2)

    # ----- insight on the data -----
    with save_fig(data_manager.path, "AvailableStations_InPeriod.png"):
        plot_stations_in_map(load_background(screenshot_period), station_coordinates, latitudes, longitudes)

    distance_between_stations = pd.DataFrame(cdist(station_coordinates.values.T, station_coordinates.values.T),
                                             columns=station_coordinates.columns, index=station_coordinates.columns)
    with save_fig(data_manager.path, "StationsDistance.png"):
        plot_pairwise_info(distance_between_stations)
    with save_fig(data_manager.path, "StationsCorrelation.png"):
        plot_pairwise_info(pollution.corr())
    with save_fig(data_manager.path, "StationsCorrelationVSDistance.png"):
        plt.scatter(distance_between_stations.values.ravel(), pollution.corr().values.ravel())

    # ----- split data in time to train and test -----
    traffic_past, pollution_past, traffic_future, pollution_future = \
        split_data_in_time(traffic_by_pixel, pollution, proportion_of_past_times)


    def llo4test(station):
        _, data_unknown = split_by_station(unknown_station=station, observed_stations=station_coordinates,
                                           observed_pollution=pollution_future, traffic=traffic_future)
        return {
            "future_pollution": data_unknown
        }


    # ----- Defining Experiment ----- #
    def train_test_model(model: BaseModel):
        def decorated_func(station):
            # in train time use the past
            data_known, _ = split_by_station(unknown_station=station, observed_stations=station_coordinates,
                                             observed_pollution=pollution_past, traffic=traffic_past)
            data_known.pop("target_positions")  # this are reserved only when testing.
            t0 = time.time()
            model.calibrate(**data_known, traffic_coords=traffic_pixels_coords)
            t_to_fit = time.time() - t0
            # in test time use the future
            data_known, data_unknown = split_by_station(unknown_station=station, observed_stations=station_coordinates,
                                                        observed_pollution=pollution_future, traffic=traffic_future)

            t0 = time.time()
            estimation = model.state_estimation(**data_known, traffic_coords=traffic_pixels_coords)
            t_to_estimate = time.time() - t0

            return {
                "model_params": model.params,
                "estimation": estimation,
                # "error": ((estimation - data_unknown.values.ravel()) ** 2).ravel(),
                "time_to_fit": t_to_fit,
                "time_to_estimate": t_to_estimate,
            }

        decorated_func.__name__ = str(model)
        return decorated_func


    models = [SnapshotMeanModel(summary_statistic="mean"),
              SnapshotMeanModel(summary_statistic="median"),
              GlobalMeanModel(),
              TrafficMeanModel(summary_statistic="mean"),
              TrafficMeanModel(summary_statistic="median")]

    lab = LabPipeline()
    lab.define_new_block_of_functions("true_values", llo4test)
    lab.define_new_block_of_functions("model", *list(map(train_test_model, models)))
    lab.execute(
        data_manager,
        num_cores=num_cores,
        forget=False,
        recalculate=False,
        save_on_iteration=None,
        station=station_coordinates.columns.to_list()
    )

    # ----- Plotting results ----- #

    generic_plot(data_manager, x="station", y="mse", label="model", plot_func=spiderplot,
                 mse=lambda estimation, future_pollution:
                 np.sqrt(((estimation - future_pollution.values.ravel()).ravel() ** 2).mean()))

    generic_plot(data_manager, x="station", y="error", label="model", plot_func=sns.boxenplot,
                 error=lambda estimation, future_pollution:
                 np.abs((estimation - future_pollution.values.ravel()).ravel()))
    # generic_plot(data_manager, x="station", y="mse", label="model", plot_func=spiderplot,
    #              mse=lambda estimation, station:
    #              np.sqrt(((estimation - split_by_station(
    #                  unknown_station=station, observed_stations=station_coordinates,
    #                  observed_pollution=pollution_future, traffic=traffic_future)[1].values[:,
    #                                     np.newaxis]).ravel() ** 2).mean()))
    #
    # generic_plot(data_manager, x="station", y="error", label="model", plot_func=sns.boxenplot,
    #              error=lambda estimation, station:
    #              np.abs((estimation - split_by_station(
    #                  unknown_station=station, observed_stations=station_coordinates,
    #                  observed_pollution=pollution_future, traffic=traffic_future)[1].values[:,
    #                                   np.newaxis]).ravel()))

    generic_plot(data_manager, x="model", y="time_to_fit", plot_func=sns.boxenplot)
    generic_plot(data_manager, x="model", y="time_to_estimate", plot_func=sns.boxenplot)

    # Conv
    print(longer_distance)
    models = [TrafficConvolutionModel(conv_kernel=gaussker, normalize=False, sigma=Bounds(0, 2 * longer_distance),
                                      loss=mse, optim_method=UNIFORM, niter=10, verbose=True)]

    lab = LabPipeline()
    lab.define_new_block_of_functions("model", *list(map(train_test_model, models)))
    lab.execute(
        data_manager,
        num_cores=num_cores,
        forget=False,
        recalculate=False,
        save_on_iteration=None,
        station=station_coordinates.columns.to_list()
    )

    # ----- Plotting results ----- #
    generic_plot(data_manager, x="station", y="mse", label="model", plot_func=spiderplot,
                 mse=lambda estimation, station:
                 np.sqrt(((estimation - split_by_station(
                     unknown_station=station, observed_stations=station_coordinates,
                     observed_pollution=pollution_future, traffic=traffic_future)[1].values[:,
                                        np.newaxis]).ravel() ** 2).mean()))

    generic_plot(data_manager, x="station", y="error", label="model", plot_func=sns.boxenplot,
                 error=lambda estimation, station:
                 np.abs((estimation - split_by_station(
                     unknown_station=station, observed_stations=station_coordinates,
                     observed_pollution=pollution_future, traffic=traffic_future)[1].values[:,
                                      np.newaxis]).ravel()))

    generic_plot(data_manager, x="model", y="time_to_fit", plot_func=sns.boxenplot)
    generic_plot(data_manager, x="model", y="time_to_estimate", plot_func=sns.boxenplot)
