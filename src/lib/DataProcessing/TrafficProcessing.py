import glob
import os.path
from pathlib import Path

import numpy as np
import pandas as pd
from imageio.v2 import imread
from scipy import sparse
from tqdm import tqdm

import src.config as config
from src.lib.DataProcessing.SeleniumScreenshots import get_info_from_name, traffic_screenshots_folder
from src.performance_utils import get_map_function, timeit

TRAFFIC_TS_LAG = -1  # 1 hour less because GMT+1 vs GMT+0

TRAFFIC_VALUES = {
    "green": 1,
    "yellow": 2,
    "red": 3,
    "dark_red": 4
}

TRAFFIC_COLORS = {
    "green": (99, 214, 104),
    "yellow": (255, 151, 77),
    "red": (242, 60, 50),
    "dark_red": (129, 31, 31)
}
road_color = (255, 255, 255)


def load_image(filepath, shape=(800, 1000)):
    image = imread(filepath)
    sx, sy, _ = np.shape(image)
    return image[(sx // 2 - shape[0] // 2):(sx // 2 + 400), (sy // 2 - 500):(sy // 2 + 500), :-1]


def load_background(screenshot_period):
    # Load background image
    return load_image(glob.glob(f"{traffic_screenshots_folder(screenshot_period)}/Background_*.png")[0])


# =============== =============== =============== #
#              Process traffic images
# =============== =============== =============== #
def filter_image_by_colors(img, colors_dict, color_width=2):
    filtered_image = np.zeros(np.shape(img)[:-1])
    for name, value in colors_dict.items():
        filtered_image[np.all((img >= np.array(TRAFFIC_COLORS[name]) - color_width) *
                              (img <= np.array(TRAFFIC_COLORS[name]) + color_width),
                              axis=-1)] = value
    return filtered_image


def process_traffic_images(screenshot_period=15, workers=1):
    stations_traffic_dir = traffic_screenshots_folder(screenshot_period)
    images = []
    dates = []
    for date, image in tqdm(get_map_function(workers=workers)(
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


def save_load_traffic_by_pixel_data(screenshot_period=15, recalculate=False, nrows2load_traffic_data=None, workers=1,
                                    chunksize=100):
    processed_filename = f"{config.traffic_dir}/Traffic_by_PixelDate.csv"
    if recalculate or not os.path.exists(processed_filename):
        with timeit("Extracting traffic pixels from images: "):
            traffic_by_pixel = process_traffic_images(screenshot_period=screenshot_period, workers=workers)
            # 2Gb and slower to save than to load images and calculate.
        with timeit("Saving extracted traffic pixels: "):
            traffic_by_pixel.to_csv(processed_filename)
    else:
        def str2tuple(s):
            x, y = s.split(",")
            return int(x[1:]), int(y[1:-1])

        with timeit(f"Loading traffic pixels data: {processed_filename}"):
            if chunksize is None:
                traffic_by_pixel = pd.read_csv(processed_filename, nrows=nrows2load_traffic_data,
                                               low_memory=True, index_col=0)
                traffic_by_pixel.index = pd.to_datetime(traffic_by_pixel.index)
                traffic_by_pixel.columns = traffic_by_pixel.columns.map(str2tuple)
            else:
                df_list = []  # list to hold the batch dataframe
                for df_chunk in tqdm(get_map_function(workers)
                                         (lambda x: x, pd.read_csv(processed_filename,
                                                                   nrows=nrows2load_traffic_data,
                                                                   chunksize=chunksize, low_memory=True,
                                                                   index_col=0))):
                    df_chunk.index = pd.to_datetime(df_chunk.index)
                    df_chunk.columns = df_chunk.columns.map(str2tuple)
                    # Alternatively, append the chunk to list and merge all
                    df_list.append(df_chunk)
                # Merge all dataframes into one dataframe
                traffic_by_pixel = pd.concat(df_list)

    return traffic_by_pixel


# =============== =============== =============== #
#                Pixels to lat-long
# =============== =============== =============== #
def project(lat, long, tile_size):
    """Mercator projection: Takes lat, long and returns world coordinates
    """
    siny = np.sin(lat * np.pi / 180)
    siny = min(max(siny, -0.9999), 0.9999)
    return tile_size * (0.5 + long / 360), tile_size * (0.5 - np.log((1 + siny) / (1 - siny)) / (4 * np.pi))


# User settings for screenshot
IMG_SETTINGS = {'lat': 48.864716,
                'long': 2.349014,
                'zoom_level': 15,
                'tile_size': 256,
                'scale': 2 ** 15,
                'shape': (5000, 10000)
                }


def inv_project(wcx, wcy, tile_size):
    """Inverse of project: Takes world coordinates and returns lat-long
    """
    long = 360 * (wcx / tile_size - 0.5)
    b = np.exp(4 * np.pi * (0.5 - wcy / tile_size))
    siny = (b - 1) / (b + 1)
    return np.arcsin(siny) * 180 / np.pi, long


def img_to_latlong(img=IMG_SETTINGS):
    """Maps each pixel to its corresponding lat-long
    """
    nx, ny = img['shape'][0], img['shape'][1]

    # Google Maps World Coordinates
    wcx, wcy = project(img['lat'], img['long'], img['tile_size'])

    # Google Maps Pixel Coordinates
    px, py = int(np.floor(wcx * img['scale'])), int(np.floor(wcy * img['scale']))

    # Matrix of Pixel Coordinates
    I, J = np.indices(img['shape'])
    PX = (px - ny // 2) * np.ones(img['shape']) + J
    PY = (py - nx // 2) * np.ones(img['shape']) + I

    # Matrix of World Coordinates
    WCX = PX / img['scale']
    WCY = PY / img['scale']

    # Matrix of Lat-Long
    LAT, LONG = inv_project(WCX, WCY, img['tile_size'])

    return LAT, LONG


def get_traffic_pixel_coords(screenshot_period, traffic_by_pixel):
    background = load_background(screenshot_period)
    px_x, px_y, _ = np.shape(background)
    _, center, zoom, _ = get_info_from_name(
        glob.glob(f"{traffic_screenshots_folder(screenshot_period)}/Screenshot*.png")[0])
    latitudes, longitudes = img_to_latlong(img={
        'lat': center.latitude,
        'long': center.longitude,
        'zoom_level': zoom,
        'tile_size': 256,
        'scale': 2 ** zoom,
        'shape': (px_x, px_y)
    })

    traffic_pixels_coords = pd.DataFrame(
        [latitudes[tuple(zip(*traffic_by_pixel.columns))], longitudes[tuple(zip(*traffic_by_pixel.columns))]],
        index=["lat", "long"],
        columns=traffic_by_pixel.columns)
    return latitudes, longitudes, traffic_pixels_coords
