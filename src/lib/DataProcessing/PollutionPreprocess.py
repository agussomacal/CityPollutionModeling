from datetime import datetime

import geopandas
import numpy as np

import pandas as pd

import src.config as config


def load_pollution_file(year):
    pollution = pd.read_csv(f"{config.observations_dir}/{year}_NO2.csv", index_col=0, header=2)[3:]
    pollution.index = pd.to_datetime(pollution.index).tz_localize(None)
    pollution = pollution.astype(float)
    return pollution


def get_pollution(date_start: datetime, date_end: datetime):
    pollution = pd.concat([load_pollution_file(year=year) for year in range(date_start.year, date_end.year + 1)])
    return pollution.loc[(pollution.index >= date_start) & (pollution.index <= date_end)]


def get_stations_lat_long():
    # Location (in Lambert 93) of the stations in the inner city of Paris (Vivien Mallet's sctip)
    stations = pd.read_csv(f"{config.observations_dir}/Station_Airparif.csv", encoding="ISO8859",
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
                                    index=stations.index, columns=["long", "lat"])
    return stations_latlong


def filter_pollution_dates(pollution, station_coordinates, traffic_by_pixel, traffic_pixels_coords, minimal_proportion_of_available_data=0.2):
    # ----- filter pollution data by traffic dates ------ #
    pollution = pollution.loc[pollution.index.intersection(traffic_by_pixel.index)]  # filter the useful rows
    known_stations = pollution.columns.intersection(station_coordinates.index)
    pollution = pollution[known_stations]  # filter the known stations
    # filter station with no data more than 20% of the relevant period
    stations_nan_mask = ~(pollution.isna().mean() > minimal_proportion_of_available_data)
    station_coordinates = station_coordinates.loc[known_stations, :]
    station_coordinates = station_coordinates.loc[stations_nan_mask, :]
    pollution = pollution.loc[:, stations_nan_mask]
    # filter the stations inside the map
    max_coords = traffic_pixels_coords.max(axis=1)
    min_coords = traffic_pixels_coords.min(axis=1)
    pollution = pollution.loc[:, ((station_coordinates <= max_coords) & (station_coordinates >= min_coords)).all(axis=1)]
    pollution.sort_index(inplace=True)
    station_coordinates = station_coordinates.loc[pollution.columns, :]
    print(f"Remaining {pollution.shape[1]} stations with enough data in studied period and selected region: "
          f"{pollution.columns}")
    return pollution, station_coordinates
