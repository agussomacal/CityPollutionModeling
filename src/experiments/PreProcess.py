import os.path
import os.path
import time
from itertools import chain
from pathlib import Path
from typing import List, Type

import joblib
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.pipeline import Pipeline

from PerplexityLab.DataManager import DataManager
from src.config import results_dir, city_dir
from src.experiments.config_experiments import screenshot_period, recalculate_traffic_by_pixel, nrows2load_traffic_data, \
    proportion_of_past_times, shuffle, server, chunksize, stations2test, simulation, max_num_stations, seed, \
    filter_graph
from src.lib.DataProcessing.PollutionPreprocess import get_pollution, get_stations_lat_long, filter_pollution_dates
from src.lib.DataProcessing.Prepare4Experiments import get_traffic_pollution_data_per_hour
from src.lib.DataProcessing.TrafficGraphConstruction import osm_graph, project_pixels2edges, project_traffic_to_edges
from src.lib.DataProcessing.TrafficProcessing import save_load_traffic_by_pixel_data, get_traffic_pixel_coords, \
    load_background
from src.lib.Models.BaseModel import BaseModel, split_by_station, Bounds, ModelsAggregator
from PerplexityLab.miscellaneous import timeit, if_true_str, filter_dict
from PerplexityLab.visualization import save_fig


def plot_stations(station_coordinates, lat, long):
    x = [np.argmin((l - long[0, :]) ** 2) for l in station_coordinates.T.long]
    y = [np.argmin((l - lat[:, 0]) ** 2) for l in station_coordinates.T.lat]
    plt.scatter(x, y, s=25, c="r", marker="x", edgecolors="k")
    for pos_x, pos_y, station_name in zip(x, y, station_coordinates.columns):
        plt.text(pos_x + 25, pos_y + 25, station_name, {'size': 7, "color": "red"})


def plot_stations_in_map(background, station_coordinates, lat, long, alpha=1.0):
    plt.imshow(background, alpha=alpha)
    plot_stations(station_coordinates, lat, long)


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
    path=results_dir,
    name=experiment_name,
    country_alpha_code="FR",
    trackCO2=True
)

with data_manager.track_emissions("PreprocessesTrafficPollution"):
    path2models = Path(
        f"{data_manager.path}/Rows{nrows2load_traffic_data}{if_true_str(simulation, '_Sim')}{if_true_str(shuffle, '_Shuffle')}_models")
    path2models.mkdir(parents=True, exist_ok=True)

    preprocessing_data_path = f"{data_manager.path}/preprocessing_data{nrows2load_traffic_data}Shuffle{shuffle}.compressed"
    if os.path.exists(preprocessing_data_path):
        with timeit("Time loading pre-processed data:"):
            traffic_past, pollution_past, traffic_future, pollution_future, station_coordinates = joblib.load(
                preprocessing_data_path)
    else:
        station_coordinates = get_stations_lat_long()
        traffic_by_pixel = save_load_traffic_by_pixel_data(
            screenshot_period=screenshot_period, recalculate=recalculate_traffic_by_pixel,
            nrows2load_traffic_data=nrows2load_traffic_data, workers=1, chunksize=chunksize)
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

    distance_between_stations_pixels = pd.DataFrame(cdist(station_coordinates.loc[["long", "lat"], :].values.T,
                                                          traffic_pixels_coords.loc[["long", "lat"], :].values.T),
                                                    columns=traffic_pixels_coords.columns,
                                                    index=station_coordinates.columns)

with data_manager.track_emissions("PreprocessGraph"):
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        graph = osm_graph(path=city_dir, filename="ParisGraph", south=lat_bounds.lower, north=lat_bounds.upper,
                          west=long_bounds.lower, east=long_bounds.upper)
        edges_pixels = project_pixels2edges(path=city_dir, filename="EdgesPixels", graph=graph,
                                            traffic_pixels_coords=traffic_pixels_coords)
        traffic_by_edge = project_traffic_to_edges(path=city_dir, filename=f"TrafficGraph",
                                                   traffic_by_pixel=pd.concat((traffic_past, traffic_future),
                                                                              axis=0).sort_index(),
                                                   edges_pixels=edges_pixels)
        if filter_graph:
            # filter by nodes with neighboaring edges having traffic and keep the biggest commponent.
            graph = nx.subgraph(graph, set(chain(*traffic_by_edge.keys())))
            print(f"nodes after filtering: {graph.number_of_nodes()}\n"
                  f"edges after filtering: {graph.number_of_edges()}")
            graph = nx.subgraph(graph, max(nx.connected_components(graph.to_undirected()), key=len))
            print(f"nodes after keeping biggest component: {graph.number_of_nodes()}\n"
                  f"edges after  keeping biggest component: {graph.number_of_edges()}")
            edges = set([(e[0], e[1]) for e in graph.to_undirected().edges])
            edges_pixels = filter_dict(edges, edges_pixels)
            traffic_by_edge = filter_dict(edges, traffic_by_edge)

if simulation:
    with data_manager.track_emissions("Simulations"):
        from src.experiments.config_experiments import simulated_model

        with timeit("Simulating dataset"):
            np.random.seed(seed)
            target_positions = traffic_pixels_coords.iloc[:,
                               np.random.choice(traffic_pixels_coords.shape[1], size=max_num_stations)]
            stations2test = list(target_positions.columns)
            station_coordinates = traffic_pixels_coords.loc[["long", "lat"], stations2test]
            distance_between_stations_pixels = pd.DataFrame(
                cdist(station_coordinates.values.T,
                      traffic_pixels_coords.loc[["long", "lat"], :].values.T),
                columns=traffic_pixels_coords.columns,
                index=stations2test)
            pollution_past = pd.DataFrame(simulated_model.state_estimation(
                observed_stations=station_coordinates,
                observed_pollution=pollution_past.iloc[:10],
                traffic=traffic_past.iloc[:10],
                target_positions=target_positions,
                traffic_coords=traffic_pixels_coords,
                distance_between_stations_pixels=distance_between_stations_pixels),
                index=traffic_past.index[:10],
                columns=target_positions.columns)

            pollution_future = pd.DataFrame(simulated_model.state_estimation(
                observed_stations=station_coordinates,
                observed_pollution=pollution_future.iloc[:10],
                traffic=traffic_future.iloc[:10],
                target_positions=target_positions,
                traffic_coords=traffic_pixels_coords,
                distance_between_stations_pixels=distance_between_stations_pixels),
                index=traffic_future.index[:10],
                columns=target_positions.columns)
            traffic_past = traffic_past.loc[pollution_past.index]
            traffic_future = traffic_future.loc[pollution_future.index]

            stations2test = stations2test[:2]
else:
    stations2test = stations2test


# ----- Defining Experiment ----- #
def train_test_model(model: BaseModel):
    def decorated_func(station):
        print(model)
        # in train time use the past
        data_known, _ = split_by_station(unknown_station=station, observed_stations=station_coordinates,
                                         observed_pollution=pollution_past, traffic=traffic_past)
        data_known.pop("target_positions")  # this are reserved only when testing.
        t0 = time.time()
        model.calibrate(**data_known, traffic_coords=traffic_pixels_coords,
                        distance_between_stations_pixels=distance_between_stations_pixels,
                        stations2test=stations2test,
                        graph=graph, traffic_by_edge=traffic_by_edge)
        t_to_fit = time.time() - t0

        path2model = Path(f"{path2models}/{station}")
        path2model.mkdir(parents=True, exist_ok=True)
        joblib.dump((model, t_to_fit), filename=f"{path2model}/{model}.compressed")

        return {}

    decorated_func.__name__ = str(model)
    return decorated_func


# ----- Defining Experiment ----- #
def train_test_averagers(models: List[BaseModel], aggregator: Pipeline):
    def decorated_func(station):
        path2model = Path(f"{path2models}/{station}")
        path2model.mkdir(parents=True, exist_ok=True)
        loaded_models, fitting_time = tuple(
            list(zip(*[joblib.load(filename=f"{path2model}/{model}.compressed") for model in models])))
        model = ModelsAggregator(models=loaded_models, aggregator=aggregator)

        # in train time use the past
        data_known, _ = split_by_station(unknown_station=station, observed_stations=station_coordinates,
                                         observed_pollution=pollution_past, traffic=traffic_past)
        data_known.pop("target_positions")  # this are reserved only when testing.
        t0 = time.time()
        model.calibrate(**data_known, traffic_coords=traffic_pixels_coords,
                        distance_between_stations_pixels=distance_between_stations_pixels,
                        stations2test=stations2test, graph=graph, traffic_by_edge=traffic_by_edge)
        t_to_fit = time.time() - t0 + np.sum(fitting_time)
        # in test time use the future
        data_known, data_unknown = split_by_station(unknown_station=station, observed_stations=station_coordinates,
                                                    observed_pollution=pollution_future, traffic=traffic_future)

        t0 = time.time()
        estimation = model.state_estimation(**data_known, traffic_coords=traffic_pixels_coords,
                                            distance_between_stations_pixels=distance_between_stations_pixels,
                                            graph=graph, traffic_by_edge=traffic_by_edge)
        t_to_estimate = time.time() - t0

        return {
            "losses": model.losses,
            "model_params": model.params,
            "estimation": estimation.ravel(),
            "error": ((estimation.ravel() - data_unknown.values.ravel()) ** 2).ravel(),
            "time_to_fit": t_to_fit,
            "time_to_estimate": t_to_estimate,
        }

    modelnames = ','.join([''.join(filter(lambda c: c.isupper(), str(model))) for model in models])
    name = f"{aggregator}_{modelnames}"
    decorated_func.__name__ = str(models[0]) if len(models) == 1 else name
    return decorated_func


print(f"CO2 {data_manager.CO2kg}kg")
print(f"Electricity consumption {data_manager.electricity_consumption_kWh}kWh")

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

    with save_fig(data_manager.path, "traffic_num_pixels.png"):
        num_pixels = (pd.concat((traffic_past, traffic_future)) > 0).sum(axis=1)
        num_pixels.hist(bins="sqrt")
        median = np.median(num_pixels)
        delta = np.max(num_pixels) - median
        plt.vlines([median - delta, median, median + delta], color="k", ymin=0, ymax=400)

    with save_fig(data_manager.path, "traffic_num_pixels_diff.png"):
        diff_pixels = num_pixels.diff()
        diff_pixels.hist(bins="sqrt")
        median = np.median(diff_pixels)
        delta = np.max(diff_pixels) - median
        plt.vlines([median - delta, median, median + delta], color="k", ymin=0, ymax=400)

    with save_fig(data_manager.path, "traffic_num_pixels_timeseries.png"):
        num_pixels.iloc[-400:].plot()
        diff_pixels.iloc[-400:].plot()

    with save_fig(data_manager.path, "Pixels2Edges.png"):
        img = load_background(screenshot_period)
        mask = np.zeros(np.shape(img)[:2])
        for i, pixels in zip(np.random.choice(len(edges_pixels), size=len(edges_pixels), replace=False),
                             edges_pixels.values()):
            mask[np.array(pixels)[:, 0], np.array(pixels)[:, 1]] = float(i) / len(edges_pixels)
        plt.imshow(mask, cmap="jet")
        plt.imshow(img, alpha=0.5)
        plt.title(f"Graph compression percentage: \n"
                  f"filtered graph: {len(edges_pixels) / np.shape(traffic_future)[1] * 100:.2f}% and "
                  f"full graph: {len(graph.edges()) / np.shape(traffic_future)[1] * 100:.2f}%")

    k_neighbours = 10
    with save_fig(data_manager.path, f"StationsGraphNeighbourhood{k_neighbours}.png"):
        img = load_background(screenshot_period)
        mask = np.zeros(np.shape(img)[:2])

        node_positions = np.array([(graph.nodes[n]["x"], graph.nodes[n]["y"]) for n in graph.nodes])
        position2node_index = [int(np.argmin(cdist(node_positions, np.array([tp])), axis=0)) for tp in
                               station_coordinates.values.T]
        for n, ix in enumerate(position2node_index):
            for edge in nx.bfs_tree(graph, source=list(graph.nodes)[ix], depth_limit=k_neighbours).edges():
                if edge in edges_pixels:
                    mask[np.array(edges_pixels[edge])[:, 0], np.array(edges_pixels[edge])[:, 1]] = 1

        plot_stations(station_coordinates, latitudes, longitudes)
        plt.imshow(img, alpha=0.25)
        plt.imshow(mask, alpha=mask, cmap="Blues")
        plt.title(f"Graph compression percentage: \n"
                  f"filtered graph: {len(edges_pixels) / np.shape(traffic_future)[1] * 100:.2f}% and "
                  f"full graph: {len(graph.edges()) / np.shape(traffic_future)[1] * 100:.2f}%")
