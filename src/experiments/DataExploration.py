from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

import src.config as config
from src.lib.DataProcessing.PollutionPreprocess import get_pollution, get_stations_lat_long, filter_pollution_dates
from src.lib.DataProcessing.TrafficProcessing import save_load_traffic_by_pixel_data, get_traffic_pixel_coords, \
    load_background
from src.viz_utils import save_fig


def plot_stations_in_map(background, station_coordinates, lat, long):
    plt.close("all")
    plt.imshow(background)

    x = [np.argmin((l - long[0, :]) ** 2) for l in station_coordinates.long]
    y = [np.argmin((l - lat[:, 0]) ** 2) for l in station_coordinates.lat]
    plt.scatter(x, y, s=25, c="r", marker="x", edgecolors="k")
    for pos_x, pos_y, station_name in zip(x, y, station_coordinates.index):
        plt.text(pos_x + 25, pos_y + 25, station_name, {'size': 7, "color": "red"})
    plt.tight_layout()


if __name__ == "__main__":
    recalculate_traffic_by_pixel = False
    proportion_of_past_times = 0.7
    nrows2load_traffic_data = 10  # None 1000
    screenshot_period = 15

    plots_dir = Path.joinpath(config.results_dir, "ScreenshotsAnalysis")
    plots_dir.mkdir(parents=True, exist_ok=True)

    traffic_by_pixel = save_load_traffic_by_pixel_data(screenshot_period=screenshot_period,
                                                       recalculate=recalculate_traffic_by_pixel,
                                                       nrows2load_traffic_data=nrows2load_traffic_data,
                                                       filename="Traffic_by_PixelDate")
    pollution = get_pollution(date_start=traffic_by_pixel.index.min(), date_end=traffic_by_pixel.index.max())
    station_coordinates = get_stations_lat_long()
    latitudes, longitudes, traffic_pixels_coords = get_traffic_pixel_coords(screenshot_period, traffic_by_pixel)
    pollution, station_coordinates = filter_pollution_dates(pollution, station_coordinates, traffic_by_pixel,
                                                            traffic_pixels_coords)

    with save_fig(plots_dir, "AvailableStations_InPeriod.png"):
        plot_stations_in_map(load_background(screenshot_period), station_coordinates, latitudes, longitudes)
