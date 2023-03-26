import os
import urllib
from datetime import datetime

import geopandas
import numpy as np

import pandas as pd

import src.config as config
from src.performance_utils import timeit


def download_pollution_files(year, filename):
    if year == 2023:
        code = "977a0df593384000991de41669679303"
    elif year == 2022:
        code = "bfeac8949e5c4d75b281a26a36aee3f4"
    else:
        raise Exception("Only years 2022 and 2023 are available")
    with timeit(f"Downloading NO2 pollution file of {year}"):
        urllib.request.urlretrieve(f"https://www.arcgis.com/sharing/rest/content/items/{code}/data", filename)


def load_pollution_file(year):
    filename = f"{config.observations_dir}/{year}_NO2.csv"
    if not os.path.exists(filename):
        download_pollution_files(year, filename)

    with timeit(f"Loading NO2 pollution file of {year}"):
        pollution = pd.read_csv(filename, index_col=0, header=2)[3:]
    pollution.index = pd.to_datetime(pollution.index).tz_localize(None)
    pollution = pollution.astype(float)
    return pollution


def get_pollution(date_start: datetime, date_end: datetime):
    pollution = pd.concat([load_pollution_file(year=year) for year in range(date_start.year, date_end.year + 1)])
    return pollution.loc[(pollution.index >= date_start) & (pollution.index <= date_end)]


def get_stations_lat_long():
    filename = f"{config.observations_dir}/Station_Airparif.csv"
    if os.path.exists(filename):
        # Location (in Lambert 93) of the stations in the inner city of Paris (Vivien Mallet's sctip)
        stations = pd.read_csv(filename, encoding="ISO8859",
                               skiprows=[0], usecols=[2, 4, 5, 6], names=["name", "z", "x", "y"])
        # Remove spaces before station names
        stations.name = stations.name.apply(lambda s: s.strip())
        # Station "BPEST" has another name in Airparif data.
        stations.loc[stations.name == "BPEST", "name"] = "BP_EST"
        stations = stations.set_index("name")

        dfstations = pd.DataFrame({
            'lambert_x': stations["x"],
            'lambert_y': stations["y"],
            'values': 2 * np.ones(len(stations))})
        gdfstations = geopandas.GeoDataFrame(
            dfstations,
            geometry=geopandas.points_from_xy(dfstations.lambert_x, dfstations.lambert_y),
            crs="EPSG:2154")

        gdfstations = gdfstations.to_crs(epsg=3857)  # Change to webmercator projection
        stations_latlong = pd.DataFrame([s.coords[0] for s in gdfstations.to_crs('epsg:4326').geometry],
                                        index=stations.index, columns=["long", "lat"]).T
    else:
        stations_latlong = pd.DataFrame([
            [48.86867978171355, 48.85719963458387, 48.84951967972958, 48.85621636967545, 48.870360489330714,
             48.87329987270092, 48.85943820267798, 48.828608029768915, 48.85261565506112, 48.8916687478872,
             48.82770896097115, 48.830548878814895, 48.83720120744435, 48.83860845472763, 48.83802336734008],
            [2.311806320972728, 2.293299113256733, 2.253411499806541, 2.3343433611339512, 2.3322591802886805,
             2.330387763170951, 2.351109707540651, 2.360279635559221, 2.3601153218686632, 2.346671039638325,
             2.3267480081515686, 2.2696778063113494, 2.393899257058465, 2.4127801276675642, 2.408116146897623]
        ],
            index=["lat", "long"],
            columns=['ELYS', 'PA07', 'AUT', 'BONAP', 'OPERA', 'HAUS', 'PA04C', 'PA13',
             'CELES', 'PA18', 'BASCH', 'PA15L', 'PA12', 'BP_EST', 'SOULT'])
    return stations_latlong


def filter_pollution_dates(pollution, station_coordinates, traffic_by_pixel, traffic_pixels_coords,
                           minimal_proportion_of_available_data=0.2):
    # ----- filter pollution data by traffic dates ------ #
    pollution = pollution.loc[pollution.index.intersection(traffic_by_pixel.index)]  # filter the useful rows
    known_stations = pollution.columns.intersection(station_coordinates.columns)
    pollution = pollution[known_stations]  # filter the known stations
    # filter station with no data more than 20% of the relevant period
    stations_nan_mask = ~(pollution.isna().mean() > minimal_proportion_of_available_data)
    station_coordinates = station_coordinates[known_stations]
    station_coordinates = station_coordinates.loc[:, stations_nan_mask]
    pollution = pollution.loc[:, stations_nan_mask]
    # filter the stations inside the map
    max_coords = traffic_pixels_coords.max(axis=1)
    min_coords = traffic_pixels_coords.min(axis=1)
    pollution = pollution.loc[:,
                ((station_coordinates.T <= max_coords) & (station_coordinates.T >= min_coords)).all(axis=1)]
    pollution.sort_index(inplace=True)
    station_coordinates = station_coordinates[pollution.columns]
    print(f"Remaining {pollution.shape[1]} stations with enough data in studied period and selected region: "
          f"{pollution.columns}")
    return pollution, station_coordinates
