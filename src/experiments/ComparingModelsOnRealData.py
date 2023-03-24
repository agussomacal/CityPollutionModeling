import time
from pathlib import Path

import numpy as np
import psutil
import seaborn as sns
from matplotlib import pyplot as plt
from spiderplot import spiderplot

import src.config as config
from src.DataManager import DataManager
from src.LabPipeline import LabPipeline
from src.lib.DataProcessing.PollutionPreprocess import get_pollution, get_stations_lat_long, filter_pollution_dates
from src.lib.DataProcessing.Prepare4Experiments import get_traffic_pollution_data_per_hour
from src.lib.DataProcessing.TrafficProcessing import save_load_traffic_by_pixel_data, get_traffic_pixel_coords, \
    load_background
from src.lib.Models.BaseModel import BaseModel, split_by_station
from src.lib.Models.TrueStateEstimationModels.AverageModels import SnapshotMeanModel, GlobalMeanModel
from src.lib.Models.TrueStateEstimationModels.TrafficConvolution import TrafficMeanModel
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
        nrows2load_traffic_data = 300  # None 1000
        num_cores = 10

    # ----- Setting data for experiment ----- #
    plots_dir = Path.joinpath(config.results_dir, "ScreenshotsAnalysis")
    plots_dir.mkdir(parents=True, exist_ok=True)

    station_coordinates = get_stations_lat_long()
    traffic_by_pixel = save_load_traffic_by_pixel_data(
        screenshot_period=screenshot_period, recalculate=recalculate_traffic_by_pixel,
        nrows2load_traffic_data=nrows2load_traffic_data, workers=1, chunksize=None)
    pollution = get_pollution(date_start=traffic_by_pixel.index.min(), date_end=traffic_by_pixel.index.max())
    latitudes, longitudes, traffic_pixels_coords = get_traffic_pixel_coords(screenshot_period, traffic_by_pixel)
    pollution, station_coordinates = filter_pollution_dates(pollution, station_coordinates, traffic_by_pixel,
                                                            traffic_pixels_coords)

    with save_fig(plots_dir, "AvailableStations_InPeriod.png"):
        plot_stations_in_map(load_background(screenshot_period), station_coordinates, latitudes, longitudes)

    traffic_past, pollution_past, traffic_future, pollution_future = \
        split_data_in_time(traffic_by_pixel, pollution, proportion_of_past_times)


    # ----- Defining Experiment ----- #
    def train_test_model(model: BaseModel):
        def decorated_func(station):
            # in train time use the past
            data_known, data_unknown = split_by_station(unknown_station=station, observed_stations=station_coordinates,
                                                        observed_pollution=pollution_past, traffic=traffic_past)
            t0 = time.time()
            model.calibrate(**data_known)
            t_to_fit = time.time() - t0
            # in test time use the future
            data_known, data_unknown = split_by_station(unknown_station=station, observed_stations=station_coordinates,
                                                        observed_pollution=pollution_future, traffic=traffic_future)

            t0 = time.time()
            estimation = model.state_estimation(**data_known)
            t_to_estimate = time.time() - t0

            return {
                "model_params": model.params,
                "estimation": estimation,
                "time_to_fit": t_to_fit,
                "time_to_estimate": t_to_estimate,
            }

        decorated_func.__name__ = str(model)
        return decorated_func


    data_manager = DataManager(
        path=config.results_dir,
        name="NewPipeline"
    )
    lab = LabPipeline()
    models = [
        SnapshotMeanModel(summary_statistic="mean"),
        SnapshotMeanModel(summary_statistic="median"),
        GlobalMeanModel(),
        TrafficMeanModel(summary_statistic="mean"),
        TrafficMeanModel(summary_statistic="median")
    ]
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
