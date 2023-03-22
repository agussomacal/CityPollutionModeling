from functools import partial
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
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
    recalculate_traffic_by_pixel = False
    proportion_of_past_times = 0.7
    nrows2load_traffic_data = 20  # None 1000
    screenshot_period = 15

    sns.set_theme()

    plots_dir = Path.joinpath(config.results_dir, "ScreenshotsAnalysis")
    plots_dir.mkdir(parents=True, exist_ok=True)

    traffic_by_pixel = save_load_traffic_by_pixel_data(screenshot_period=screenshot_period,
                                                       recalculate=recalculate_traffic_by_pixel,
                                                       nrows2load_traffic_data=nrows2load_traffic_data,
                                                       workers=1)
    pollution = get_pollution(date_start=traffic_by_pixel.index.min(), date_end=traffic_by_pixel.index.max())
    station_coordinates = get_stations_lat_long()
    latitudes, longitudes, traffic_pixels_coords = get_traffic_pixel_coords(screenshot_period, traffic_by_pixel)
    pollution, station_coordinates = filter_pollution_dates(pollution, station_coordinates, traffic_by_pixel,
                                                            traffic_pixels_coords)

    with save_fig(plots_dir, "AvailableStations_InPeriod.png"):
        plot_stations_in_map(load_background(screenshot_period), station_coordinates, latitudes, longitudes)

    traffic_past, pollution_past, traffic_future, pollution_future = \
        split_data_in_time(traffic_by_pixel, pollution, proportion_of_past_times)


    def train_test_model(model: BaseModel):
        def decorated_func(station):
            # in train time use the past
            data_known, data_unknown = split_by_station(unknown_station=station, observed_stations=station_coordinates,
                                                        observed_pollution=pollution_past, traffic=traffic_past)
            model.calibrate(**data_known)
            # in test time use the future
            data_known, data_unknown = split_by_station(unknown_station=station, observed_stations=station_coordinates,
                                                        observed_pollution=pollution_future, traffic=traffic_future)

            return {
                "model_params": model.params,
                "estimation": model.state_estimation(**data_known)
            }

        decorated_func.__name__ = str(model)
        return decorated_func


    data_manager = DataManager(
        path=config.results_dir,
        name="NewPipeline"
    )
    lab = LabPipeline()
    models = [SnapshotMeanModel(), GlobalMeanModel()]
    lab.define_new_block_of_functions("model", *list(map(train_test_model, models)))
    lab.execute(
        data_manager,
        num_cores=2,
        forget=False,
        recalculate=False,
        save_on_iteration=None,
        station=station_coordinates.columns.to_list()
    )
    generic_plot(data_manager, x="station", y="se", label="model", plot_func=spiderplot,
                 se=lambda estimation, station:
                 ((estimation - split_by_station(
                     unknown_station=station, observed_stations=station_coordinates,
                     observed_pollution=pollution_future, traffic=traffic_future)[1].values[:,
                                np.newaxis]).ravel() ** 2).mean())

    generic_plot(data_manager, x="station", y="se", label="model", plot_func=sns.barplot,
                 se=lambda estimation, station:
                 ((estimation - split_by_station(
                     unknown_station=station, observed_stations=station_coordinates,
                     observed_pollution=pollution_future, traffic=traffic_future)[1].values[:,
                                np.newaxis]).ravel() ** 2).mean())
