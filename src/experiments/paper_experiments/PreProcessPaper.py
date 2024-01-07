import os.path
import time
from datetime import datetime
from itertools import chain
from pathlib import Path
from typing import List, Union, Tuple

import joblib
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pylab as plt
import matplotlib as mpl
from matplotlib import colors
from scipy.interpolate import griddata
from scipy.spatial.distance import cdist
from sklearn.pipeline import Pipeline

from PerplexityLab.DataManager import DataManager
from PerplexityLab.miscellaneous import timeit, if_true_str, filter_dict, if_exist_load_else_do
from PerplexityLab.visualization import save_fig, perplex_plot, one_line_iterator
from src.config import results_dir, city_dir
from src.experiments.paper_experiments.params4runs import screenshot_period, recalculate_traffic_by_pixel, \
    nrows2load_traffic_data, proportion_of_past_times, shuffle, chunksize, stations2test, simulation, max_num_stations, \
    seed, filter_graph, runsinfo, path2latex_figures
from src.lib.DataProcessing.OtherVariablesPreprocess import process_windGuru_data
from src.lib.DataProcessing.PollutionPreprocess import get_pollution, get_stations_lat_long, filter_pollution_dates
from src.lib.DataProcessing.Prepare4Experiments import get_traffic_pollution_data_per_hour
from src.lib.DataProcessing.SeleniumScreenshots import traffic_screenshots_folder, center_of_paris, \
    get_filename_from_date
from src.lib.DataProcessing.TrafficGraphConstruction import osm_graph, project_pixels2edges, project_traffic_to_edges
from src.lib.DataProcessing.TrafficProcessing import save_load_traffic_by_pixel_data, get_traffic_pixel_coords, \
    load_background, load_image, filter_image_by_colors, TRAFFIC_VALUES, TRAFFIC_COLORS
from src.lib.FeatureExtractors.GraphFeatureExtractors import get_graph_node_positions
from src.lib.Models.BaseModel import BaseModel, split_by_station, ModelsAggregator
from src.lib.Modules import Bounds


def plot_stations(station_coordinates, lat, long, color="red", marker="x", size=7, label=True):
    x = [np.argmin((l - long[0, :]) ** 2) for l in station_coordinates.T.long]
    y = [np.argmin((l - lat[:, 0]) ** 2) for l in station_coordinates.T.lat]
    plt.scatter(x, y, s=25, c=color, marker=marker, edgecolors="k")
    if label:
        for pos_x, pos_y, station_name in zip(x, y, station_coordinates.columns):
            plt.text(pos_x + 25, pos_y + 25, station_name, {'size': size, "color": color})


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

    # ------ Other data ------ #
    # TODO check version of osmnx: AttributeError: module 'osmnx' has no attribute 'nearest_edges'
    wind = process_windGuru_data("WindVelocity")
    temperature = process_windGuru_data("Temperature")
    # filtering times where some station has NaN
    available_times = pollution_past.index[~pollution_past.isna().T.any()].union(
        pollution_future.index[~pollution_future.isna().T.any()])
    # filter times where wind and temperature also exist.
    available_times = temperature[~temperature.isna()].index.intersection(
        wind[~wind.isna().values].index.intersection(available_times))
    print("Available times after filtering Temperature and Wind NaNs", len(available_times))

    pollution_past = pollution_past.loc[pollution_past.index.intersection(available_times), :]
    pollution_future = pollution_future.loc[pollution_future.index.intersection(available_times), :]
    traffic_past = pollution_past.loc[traffic_past.index.intersection(available_times), :]
    traffic_future = pollution_future.loc[traffic_future.index.intersection(available_times), :]
    times_past = pollution_past.index
    times_future = pollution_future.index
    times_all = times_past.tolist() + times_future.tolist()
    runsinfo.append_info(
        numstations=int(len(station_coordinates.columns)),
        numberoftesttimes=len(times_future),
        numberoftraintimes=len(times_past),
        numberoftimes=len(times_all),
    )

with data_manager.track_emissions("PreprocessGraph"):
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        graph = osm_graph(path=city_dir, filename="ParisGraph", south=lat_bounds.lower, north=lat_bounds.upper,
                          west=long_bounds.lower, east=long_bounds.upper)
        runsinfo.append_info(
            numnodes=int(graph.number_of_nodes()),
            numedges=int(graph.number_of_edges())
        )
        # TODO check version of osmnx: AttributeError: module 'osmnx' has no attribute 'nearest_edges'
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

        runsinfo.append_info(
            numnodesfilter=int(graph.number_of_nodes()),
            numedgesfilter=int(graph.number_of_edges())
        )

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

def train_test(model, station):
    # in train time use the past
    data_known, data_unknown = split_by_station(unknown_station=station, observed_stations=station_coordinates,
                                                observed_pollution=pollution_past, traffic=traffic_past)
    target_position = data_known.pop("target_positions")  # this are reserved only when testing.
    t0 = time.time()
    model.calibrate(**data_known, traffic_coords=traffic_pixels_coords,
                    distance_between_stations_pixels=distance_between_stations_pixels,
                    stations2test=stations2test, graph=graph, traffic_by_edge=traffic_by_edge,
                    temperature=temperature, wind=wind, longitudes=longitudes, latitudes=latitudes,
                    target_position=target_position, target_observation=data_unknown)
    t_to_fit = time.time() - t0

    # in test time use the future
    data_known, data_unknown = split_by_station(unknown_station=station, observed_stations=station_coordinates,
                                                observed_pollution=pollution_future, traffic=traffic_future)

    t0 = time.time()
    estimation = model.state_estimation(**data_known, traffic_coords=traffic_pixels_coords,
                                        distance_between_stations_pixels=distance_between_stations_pixels,
                                        graph=graph, traffic_by_edge=traffic_by_edge,
                                        temperature=temperature, wind=wind, longitudes=longitudes,
                                        latitudes=latitudes)
    t_to_estimate = time.time() - t0

    # path2model = Path(f"{path2models}/{station}")
    # path2model.mkdir(parents=True, exist_ok=True)
    # joblib.dump((model, t_to_fit), filename=f"{path2model}/{model}.compressed")

    return {
        "losses": model.losses,
        "model_params": model.params,
        "estimation": estimation.ravel(),
        "ground_truth": data_unknown.values.ravel(),
        "error": ((estimation.ravel() - data_unknown.values.ravel()) ** 2).ravel(),
        "time_to_fit": t_to_fit,
        "time_to_estimate": t_to_estimate,
        "trained_model": model,
        "model_name": str(model)
    }


def train_test_model(model: BaseModel):
    def decorated_func(station):
        if isinstance(model, BaseModel):
            m = model
        else:
            aggregator, models = model
            path2model = Path(f"{path2models}/{station}")
            path2model.mkdir(parents=True, exist_ok=True)
            loaded_models, fitting_time = tuple(
                list(zip(*[joblib.load(filename=f"{path2model}/{m}.compressed") for m in models])))
            m = ModelsAggregator(models=loaded_models, aggregator=aggregator)

        print(station, m)
        res = train_test(m, station)

        if isinstance(model, BaseModel):
            path2model = Path(f"{path2models}/{station}")
            path2model.mkdir(parents=True, exist_ok=True)
            joblib.dump((m, res["time_to_fit"]), filename=f"{path2model}/{m}.compressed")

        return res

    decorated_func.__name__ = str(model) if isinstance(model, BaseModel) else "_".join(list(map(str, model[1:])))
    return decorated_func


def estimate_pollution_map(time, station, trained_model, num_points):
    data_known, data_unknown = split_by_station(unknown_station=station, observed_stations=station_coordinates,
                                                observed_pollution=pollution_future.loc[[time], :],
                                                traffic=traffic_future.loc[[time], :])
    long, lat = np.meshgrid(np.linspace(*long_bounds, num=num_points), np.linspace(*lat_bounds, num=num_points))

    data_known["target_positions"] = pd.DataFrame([long.ravel(), lat.ravel()], index=["long", "lat"])
    estimation = trained_model.state_estimation(**data_known, traffic_coords=traffic_pixels_coords,
                                                distance_between_stations_pixels=distance_between_stations_pixels,
                                                graph=graph, traffic_by_edge=traffic_by_edge,
                                                temperature=temperature, wind=wind, longitudes=longitudes,
                                                latitudes=latitudes)

    # def par_func(arg):
    #     dk = arg[0]
    #     dk["target_positions"] = arg[1]
    #     return trained_model.state_estimation(**dk, traffic_coords=traffic_pixels_coords,
    #                                           distance_between_stations_pixels=distance_between_stations_pixels,
    #                                           graph=graph, traffic_by_edge=traffic_by_edge,
    #                                           temperature=temperature, wind=wind, longitudes=longitudes,
    #                                           latitudes=latitudes)
    #
    # ncores = 10
    # step = num_points ** 2 // ncores
    # estimation = np.hstack(list(Pool(ncores).map(par_func, [(data_known.copy(),
    #                                                          pd.DataFrame([long.ravel()[i * step:(i + 1) * step],
    #                                                                        lat.ravel()[i * step:(i + 1) * step]],
    #                                                                       index=["long", "lat"])) for i in
    #                                                         range(ncores)])))

    return long, lat, estimation.reshape((num_points, num_points))


@perplex_plot
def plot_pollution_map(fig, ax, station, trained_model, time, cmap='RdGy', num_points=20):
    station = station[0]
    trained_model = trained_model[0]
    img = load_background(screenshot_period)
    ax.imshow(img, extent=[0, 1, 0, 1], alpha=1)

    long, lat, estimation = estimate_pollution_map(time, station, trained_model, num_points=num_points)

    # long = (long - long_bounds.lower) / (long_bounds.upper - long_bounds.lower)
    # lat = (lat - lat_bounds.lower) / (lat_bounds.upper - lat_bounds.lower)
    # contours = ax.contour(long, lat, estimation, 3, colors='black', alpha=0.5)
    # ax.clabel(contours, inline=True, fontsize=8)
    ax.imshow(estimation, extent=[0, 1, 0, 1], origin='lower', cmap=cmap, alpha=0.5)
    # ax.colorbar()


@if_exist_load_else_do(file_format="csv", loader=pd.read_csv, saver=pd.DataFrame.to_csv)
def estimate_pollution_map_in_graph(time, station, trained_model, nodes_indexes):
    data_known, data_unknown = split_by_station(unknown_station=station, observed_stations=station_coordinates,
                                                observed_pollution=pollution_future.loc[[time], :],
                                                traffic=traffic_future.loc[[time], :])
    node_positions = get_graph_node_positions(graph)[nodes_indexes]
    data_known["target_positions"] = pd.DataFrame(node_positions, columns=["long", "lat"]).T
    estimation = trained_model.state_estimation(**data_known, traffic_coords=traffic_pixels_coords,
                                                distance_between_stations_pixels=distance_between_stations_pixels,
                                                graph=graph, traffic_by_edge=traffic_by_edge,
                                                temperature=temperature, wind=wind, longitudes=longitudes,
                                                latitudes=latitudes)
    return pd.concat(
        (pd.DataFrame(node_positions, columns=["long", "lat"]), pd.DataFrame(estimation, columns=["pollution"])),
        axis=1)


@perplex_plot(legend=False)
@one_line_iterator
def plot_pollution_map_in_graph(fig, ax, station, trained_model, diffusion_method=None, time=None, nodes_indexes=None,
                                cmap='RdGy', zoom=13, center_of_city=center_of_paris, s=20, alpha=0.5, bar=False,
                                max_val=None, plot_nodes=False, levels=0):
    img = load_image(
        f"{traffic_screenshots_folder(screenshot_period)}/{get_filename_from_date(zoom, *center_of_city, time.utctimetuple())}.png")

    estimation = estimate_pollution_map_in_graph(path=data_manager.path,
                                                 filename=f"PollutionEstimation_{station}_{time}_{trained_model}",
                                                 time=time, station=station, trained_model=trained_model,
                                                 nodes_indexes=nodes_indexes)

    # smoothing
    pollution = np.ravel(diffusion_method(estimation["pollution"].values.reshape((-1, 1))))
    x = (estimation["long"] - long_bounds.lower) / (long_bounds.upper - long_bounds.lower)
    y = (estimation["lat"] - lat_bounds.lower) / (lat_bounds.upper - lat_bounds.lower)
    print(min(pollution), max(pollution))
    if plot_nodes:
        sc = ax.scatter(x=x,
                        y=y,
                        c=pollution, cmap=cmap,
                        norm=colors.Normalize(vmin=0, vmax=max_val) if max_val is not None else None,
                        s=s, alpha=alpha)
    else:
        grid_x, grid_y = np.meshgrid(np.linspace(np.min(estimation["long"]), np.max(estimation["long"])),
                                     np.linspace(np.min(estimation["lat"]), np.max(estimation["lat"])))
        grid_z2 = griddata(np.transpose([x, y]), pollution, (grid_x, grid_y), method='nearest')
        if levels > 0:
            sc = ax.contourf(grid_z2, levels=levels, alpha=alpha, cmap=cmap,
                             norm=colors.Normalize(vmin=0, vmax=max_val) if max_val is not None else None)
        else:
            sc = ax.imshow(grid_z2.T, extent=(0, 1, 0, 1), origin='lower', alpha=alpha, cmap=cmap,
                           # interpolation="bicubic",
                           # norm=colors.Normalize(vmin=0, vmax=max_val) if max_val is not None else None
                           )

    ax.imshow(img, extent=[0, 1, 0, 1], alpha=1 - alpha)

    if bar:
        plt.colorbar(sc, ax=ax)


print(f"CO2 {data_manager.CO2kg}kg")
print(f"Electricity consumption {data_manager.electricity_consumption_kWh}kWh")

if __name__ == "__main__":
    # import contextily
    # import geopandas
    import osmnx
    import shutil
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    # ----- Available stations -----
    node_positions = get_graph_node_positions(graph)
    vertex_stations = pd.DataFrame(np.squeeze([node_positions[np.argmin(cdist(node_positions, np.array([tp])), axis=0)]
                                               for tp in map(tuple, station_coordinates.values.T)]),
                                   columns=["long", "lat"],
                                   index=station_coordinates.columns).T
    # Linear approximation of distance: distance between Arc du Triumph and Vincennes 9,84km or
    # 0.1266096365565058 lat long
    ratio = 9840 / cdist([[48.87551413370949, 2.2944611276838867]], [[48.835641353886075, 2.414628350744604]])[0][0]

    runsinfo.append_info(
        MinDistNodeStation=int(np.max(np.sqrt(np.sum((station_coordinates - vertex_stations) ** 2, axis=0))) * ratio),
    )
    with save_fig([data_manager.path, path2latex_figures], "AvailableStations_InPeriod.pdf", dpi=300):
        plt.imshow(load_background(screenshot_period), alpha=1.0)
        plot_stations(vertex_stations, latitudes, longitudes, color="red", marker="o", size=7)
        plot_stations(station_coordinates, latitudes, longitudes, color="blue", marker="x", size=7, label=False)

    # ---------- Graph and cropping ---------- #
    original_graph = osm_graph(path=city_dir, filename="ParisGraph", south=lat_bounds.lower, north=lat_bounds.upper,
                               west=long_bounds.lower, east=long_bounds.upper)
    # to_remove = random.sample(list(graph.edges()), k=int(0.2 * graph.number_of_edges()))
    # graph.remove_edges_from(to_remove)
    # runsinfo.append_info(
    #     numnodesfilter=int(graph.number_of_nodes()),
    #     numedgesfilter=int(graph.number_of_edges())
    # )
    # plot deleted edges in red
    ec = ['y' if e in graph.edges else 'r' for e in original_graph.edges(keys=True, data=False)]
    with save_fig([data_manager.path, path2latex_figures], "GoogleMapsAndOpenStreetMap.pdf", dpi=300):
        figsize_x = 10
        fig, ax = plt.subplots(1, 1, figsize=(figsize_x, 10))
        dimensions = np.max(node_positions, axis=0) - np.min(node_positions, axis=0)
        fig, ax = osmnx.plot_graph(
            original_graph, ax=ax,
            bgcolor='white',
            node_color="k", node_size=5.,
            edge_color=ec, edge_linewidth=2.,
            show=False, close=False, figsize=(figsize_x, dimensions[1] / dimensions[0] * figsize_x)
        )  # Graph
        # contextily.add_basemap(ax, zoom=14, source=contextily.providers.OpenStreetMap.France.url)
        # ax.imshow(load_background(screenshot_period), alpha=0.5)
        fig.set_facecolor("white")

    # ---------- correlation distance ---------- #
    distance_between_stations = pd.DataFrame(
        cdist(station_coordinates.values.T * ratio, station_coordinates.values.T * ratio),
        columns=station_coordinates.columns, index=station_coordinates.columns)
    runsinfo.append_info(operahausdistance=f'{distance_between_stations.loc["OPERA", "HAUS"]:.0f}')
    pollution = get_pollution(date_start=datetime(2022, 1, 1, 0), date_end=datetime.now())
    pollution = pollution[station_coordinates.columns]
    triu = np.triu_indices(len(distance_between_stations), 1)
    with save_fig([data_manager.path, path2latex_figures], "StationsCorrelationVSDistance.pdf", dpi=300):
        plt.scatter(distance_between_stations.values[triu], pollution.corr().values[triu])
        plt.xlabel("Distance (m)")
        plt.ylabel("Correlation")
        plt.title("Correlation and distance between stations")
        plt.axvline(runsinfo["MinDistNodeStation"], linestyle="--", c="r")

    with save_fig([data_manager.path], "StationsCorrelationVSDistancelog.pdf", dpi=300):
        plt.scatter(distance_between_stations.values[triu], pollution.corr().values[triu])
        plt.xlabel("Distance (m)")
        plt.ylabel("Correlation")
        plt.title("Correlation and distance between stations")
        plt.axvline(runsinfo["MinDistNodeStation"], linestyle="--", c="r")
        plt.xscale("log")
        # plt.yscale("log")

    # ---------- correlation distance ---------- #
    with save_fig([data_manager.path], "StationsCorrelationVSDistanceWithoutSnaphotmean.pdf", dpi=300):
        plt.scatter(distance_between_stations.values[triu],
                    (pollution.T - pollution.mean(axis=1)).T.corr().values[triu])
        plt.xlabel("Distance (m)")
        plt.ylabel("Correlation")
        plt.title("Correlation and distance between stations")
        plt.axvline(runsinfo["MinDistNodeStation"], linestyle="--", c="r")

    # ---------- correlation and resistors distance ---------- #
    graph = nx.Graph(graph).to_undirected()
    nx.set_edge_attributes(graph,
                           {(u, v): data["length"] for u, v, data in graph.edges.data()},  # /data["lanes"]
                           'weight')
    print(min(nx.get_edge_attributes(graph, "weight").values()))
    with save_fig([data_manager.path], "ResistorDistanceHistogram.pdf", dpi=300):
        sns.histplot(nx.get_edge_attributes(graph, "weight").values())
        plt.xlabel("Weight distance")
        plt.ylabel("Counts")
        plt.title("Weight histogram")
        # plt.axvline(runsinfo["MinDistNodeStation"], linestyle="--", c="r")

    vertex_stations_index = pd.Series(np.squeeze([np.argmin(cdist(get_graph_node_positions(graph),
                                                                  np.array([tp])), axis=0)
                                                  for tp in map(tuple, station_coordinates.values.T)]),
                                      index=station_coordinates.columns).T
    distance = pd.DataFrame(0, columns=vertex_stations_index.index, index=vertex_stations_index.index)
    for i in range(len(vertex_stations_index)):
        for j in range(i + 1, len(vertex_stations_index)):
            distance.iloc[i, j] = distance.iloc[j, i] = (
                nx.resistance_distance(graph, list(graph.nodes)[vertex_stations_index.values[i]],
                                       list(graph.nodes)[vertex_stations_index.values[j]],
                                       weight="weight", invert_weight=True))
    triu = np.triu_indices(len(distance_between_stations), 1)
    with save_fig([data_manager.path, path2latex_figures], "StationsCorrelationVSResistorDistance.pdf", dpi=300):
        plt.scatter(distance.values[triu], pollution.corr().values[triu])
        plt.xlabel("Resistor distance")
        plt.ylabel("Correlation")
        plt.title("Correlation and resistor distance between stations")

    with save_fig([data_manager.path], "StationsCorrelationVSResistorDistancelog.pdf", dpi=300):
        plt.scatter(distance.values[triu], pollution.corr().values[triu])
        plt.xlabel("Resistor distance")
        plt.ylabel("Correlation")
        plt.title("Correlation and resistor distance between stations")
        plt.xscale("log")
        plt.yscale("log")

    with save_fig([data_manager.path], "ResistorDistanceVSStationsDistance.pdf", dpi=300):
        plt.scatter(distance_between_stations.loc[distance.index, distance.columns].values[triu], distance.values[triu])
        plt.ylabel("Resistor distance")
        plt.xlabel("Euclidean distance")
        plt.title("Resistor distance vs Euclidean")
        # plt.axvline(runsinfo["MinDistNodeStation"], linestyle="--", c="r")

    with save_fig([data_manager.path], "StationsCorrelationVSResistorDistanceWithoutSnaphotmean.pdf", dpi=300):
        plt.scatter(distance.values[triu], (pollution.T - pollution.mean(axis=1)).T.corr().values[triu])
        plt.xlabel("Resistor distance")
        plt.ylabel("Correlation")
        plt.title("Correlation and resistor distance between stations")

    # ---------- Histogram of degree ---------- #
    with save_fig([data_manager.path, path2latex_figures], "Ghistdegree.pdf"):
        degree_sequence = sorted((d for n, d in graph.degree()), reverse=True)

        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        # fig = plt.figure("Degree of a random graph", figsize=(8, 8))
        # # Create a gridspec for adding subplots of different sizes
        # axgrid = fig.add_gridspec(5, 4)
        # ax2 = fig.add_subplot(axgrid[3:, 2:])
        ax.bar(*np.unique(degree_sequence, return_counts=True))
        ax.set_title("Degree histogram")
        ax.set_xlabel("Degree")
        ax.set_ylabel("# of Nodes")

    # ---------- Example screenshot ---------- #
    image_path = f"{traffic_screenshots_folder(screenshot_period)}/Screenshot_48.8580073_2.3342828_13_2022_12_8_10_45.png"
    shutil.copyfile(
        image_path,
        f"{path2latex_figures}/TrafficScreenshot.png"
    )

    # ---------- Projection of colors in graph ---------- #
    runsinfo.append_info(
        green=TRAFFIC_COLORS["green"],
        yellow=TRAFFIC_COLORS["yellow"],
        red=TRAFFIC_COLORS["red"],
        darkred=TRAFFIC_COLORS["dark_red"]
    )
    image = load_image(image_path)
    image = filter_image_by_colors(image, TRAFFIC_VALUES, color_width=1)
    for color, value in TRAFFIC_VALUES.items():
        with save_fig([data_manager.path, path2latex_figures], f"{color}.pdf", dpi=300):
            plt.imshow(load_background(screenshot_period), alpha=0.45)
            cmap = mpl.colors.LinearSegmentedColormap.from_list(
                'binary_transparent', [(0, 0, 0, 0), np.array(TRAFFIC_COLORS[color]) / 255], 256)
            plt.imshow(image == value, alpha=1, interpolation='none', cmap=cmap)
