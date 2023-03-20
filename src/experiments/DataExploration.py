from datetime import datetime
from pathlib import Path

import numpy as np

import pandas as pd

import config
from lib.DataProcessing.SeleniumScreenshots import traffic_screenshots_folder
from lib.DataProcessing.TrafficProcessing import save_load_traffic_by_pixel_data


def load_pollution_file(year):
    pollution = pd.read_csv(f"{config.observations_dir}/{year}_NO2.csv", index_col=0, header=2)[3:]
    pollution.index = pd.to_datetime(pollution.index).tz_localize(None)
    pollution = pollution.astype(float)
    return pollution


def get_pollution(date_start: datetime, date_end: datetime):
    pollution = pd.concat([load_pollution_file(year=year) for year in range(date_start.year, date_end.year + 1)])
    return pollution.loc[(pollution.index >= date_start) & (pollution.index <= date_end)]


if __name__ == "__main__":
    recalculate_traffic_by_pixel = False
    proportion_of_past_times = 0.7
    nrows2load_traffic_data = 1000  # None 1000
    screenshot_period = 15

    stations_traffic_dir = traffic_screenshots_folder(screenshot_period)
    traffic_by_pixel = save_load_traffic_by_pixel_data(screenshot_period=screenshot_period,
                                                       recalculate=recalculate_traffic_by_pixel,
                                                       nrows2load_traffic_data=nrows2load_traffic_data,
                                                       filename="Traffic_by_PixelDate")
    pollution = get_pollution(date_start=traffic_by_pixel.index.min(), date_end=traffic_by_pixel.index.max())
