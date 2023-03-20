import glob
from pathlib import Path

import numpy as np
import pandas as pd
from imageio.v2 import imread
from scipy import sparse
from tqdm import tqdm

import config
from lib.DataProcessing.SeleniumScreenshots import get_info_from_name, traffic_screenshots_folder
from src.performance_utils import get_map_function, timeit

TRAFFIC_TS_LAG = -1  # 1 hour less because GMT+1 vs GMT+0

TRAFFIC_VALUES = {
    "green": 1,
    "yellow": 2,
    "red": 3,
    "really_bad": 4
}

TRAFFIC_COLORS = {
    "green": (99, 214, 104),
    "yellow": (255, 151, 77),
    "red": (242, 60, 50),
    "really_bad": (129, 31, 31)
}
road_color = (255, 255, 255)


def load_image(filepath, shape=(800, 1000)):
    image = imread(filepath)
    sx, sy, _ = np.shape(image)
    return image[(sx // 2 - shape[0] // 2):(sx // 2 + 400), (sy // 2 - 500):(sy // 2 + 500), :-1]


def filter_image_by_colors(img, colors_dict, color_width=2):
    filtered_image = np.zeros(np.shape(img)[:-1])
    for name, value in colors_dict.items():
        filtered_image[np.all((img >= np.array(TRAFFIC_COLORS[name]) - color_width) *
                              (img <= np.array(TRAFFIC_COLORS[name]) + color_width),
                              axis=-1)] = value
    return filtered_image


def process_traffic_images(screenshot_period=15):
    stations_traffic_dir = traffic_screenshots_folder(screenshot_period)
    images = []
    dates = []
    for date, image in tqdm(get_map_function(workers=1)(
            lambda im_path: (get_info_from_name(im_path)[-1], load_image(im_path)),
            glob.glob(f"{stations_traffic_dir}/Screenshot*.png")), desc="Loading images"):
        images.append(sparse.csr_matrix(filter_image_by_colors(image, TRAFFIC_VALUES, color_width=1)))
        dates.append(date)

    # temporal lag
    dates = pd.to_datetime(dates) + pd.tseries.frequencies.to_offset(f"{TRAFFIC_TS_LAG}H")
    dates = dates.map(lambda x: pd.to_datetime(x).floor('15T'))  # round in minutes

    pixels_with_traffic = sum((img > 0 for img in images)).nonzero()

    traffic = pd.DataFrame(0, index=dates, columns=list(zip(*pixels_with_traffic)))
    for date, img in tqdm(zip(dates, images), "Creating traffic pixel summary DataFrame."):
        traffic.loc[date, list(zip(*img.nonzero()))] = img.data
    traffic.sort_index(inplace=True)
    return traffic


def save_load_traffic_by_pixel_data(screenshot_period=15, recalculate=False, nrows2load_traffic_data=None,
                                    filename="Traffic_by_PixelDate"):
    if recalculate:
        with timeit("Extracting traffic pixels from images: "):
            traffic_by_pixel = process_traffic_images(screenshot_period=screenshot_period)
            # 2Gb and slower to save than to load images and calculate.
        print("Saving traffic_by_pixel pixel summary DataFrame.")
        with timeit("Saving extracted traffic pixels: "):
            traffic_by_pixel.to_csv(f"{config.traffic_dir}/{filename}.zip", compression="zip")
        # np.unique(traffic_by_pixel.values.ravel())   # array([0., 1., 2., 3., 4.]) There is exactly 4 colors.
    else:
        def str2tuple(s):
            x, y = s.split(",")
            return int(x[1:]), int(y[1:-1])

        with timeit("Loading traffic pixels data: "):
            traffic_by_pixel = pd.read_csv(f"{config.traffic_dir}/{filename}.zip",
                                           compression="zip", nrows=nrows2load_traffic_data,
                                           low_memory=True, index_col=0)
        traffic_by_pixel.index = pd.to_datetime(traffic_by_pixel.index)
        traffic_by_pixel.columns = traffic_by_pixel.columns.map(str2tuple)
    return traffic_by_pixel
