import numpy as np
import seaborn as sns

from PerplexityLab.DataManager import DataManager
from PerplexityLab.LabPipeline import LabPipeline
from PerplexityLab.miscellaneous import NamedPartial
from PerplexityLab.visualization import generic_plot
from src import config
from src.experiments.paper_experiments.PreProcessPaper import traffic_by_edge, graph, times_all, station_coordinates, \
    pollution_past, traffic_past, pollution_future
from src.experiments.paper_experiments.params4runs import stations2test
from src.lib.DataProcessing.TrafficProcessing import TRAFFIC_VALUES
from src.lib.FeatureExtractors.GraphFeatureExtractors import get_graph_node_positions
from src.lib.Models.TrueStateEstimationModels.PhysicsModel import get_traffic_by_node_conv, get_nearest_node_mapping, \
    get_space_indexes, get_time_indexes, distance_between_nodes

data_manager = DataManager(
    path=config.paper_experiments_dir,
    emissions_path=config.results_dir,
    name="TrafficConv",
    country_alpha_code="FR",
    trackCO2=True
)


def gaussian(d, sigma):
    return np.exp(-d ** 2 / sigma ** 2 / 2)


def exponential(d, sigma):
    return np.exp(-d / sigma)


def rational(d, sigma):
    return 1.0 / (d + sigma)


weight_functions = {
    exponential.__name__: exponential,
    gaussian.__name__: gaussian,
    rational.__name__: rational
}

experiment_name = "PollutionTraffic_statistics"
if __name__ == "__main__":
    with data_manager.track_emissions("Emissions for pollution traffic statistics."):
        path4preprocess = data_manager.path
        redo_preprocessing = False
        observed_stations = station_coordinates
        observed_pollution = pollution_past.loc[:, station_coordinates.columns]  # same columns order as locations
        observed_pollution_future = pollution_future.loc[:,
                                    station_coordinates.columns]  # same columns order as locations

        traffic_by_node = get_traffic_by_node_conv(
            path=path4preprocess, times=times_all,
            traffic_by_edge=traffic_by_edge,
            graph=graph, recalculate=redo_preprocessing, lnei=1)
        # [#times, #nodes, #traffic colors]

        node_positions = get_graph_node_positions(graph)
        position2node_index = get_nearest_node_mapping(path=path4preprocess, filename=f"pos2node",
                                                       target_positions=observed_stations,
                                                       graph=graph, recalculate=redo_preprocessing)
        times = get_time_indexes(times_all, observed_pollution.index)
        nodes = get_space_indexes(position2node_index, observed_stations)

        distances_betw_nodes = distance_between_nodes(path=path4preprocess, recalculate=redo_preprocessing,
                                                      graph=graph)

    plotting_stations = stations2test


    def correlation(station, traffic_by_node, color):
        c = TRAFFIC_VALUES[color] - 1
        pollution = observed_pollution[station]
        z_pollution = (pollution - np.mean(pollution)) / np.std(pollution)
        z_traffic = (traffic_by_node[:, c] - np.mean(traffic_by_node[:, c], axis=0)) / np.std(
            traffic_by_node[:, c], axis=0)
        return np.nanmean(z_pollution * z_traffic, axis=0)


    def correlation_no_avg(station, traffic_by_node, color):
        c = TRAFFIC_VALUES[color] - 1
        pollution = observed_pollution[station] - np.mean(observed_pollution[plotting_stations].values, axis=1)
        z_pollution = ((pollution - np.mean(pollution)) / np.std(pollution))
        z_traffic = (traffic_by_node[:, c] - np.mean(traffic_by_node[:, c], axis=0)) / np.std(
            traffic_by_node[:, c], axis=0)
        return np.nanmean(z_pollution * z_traffic, axis=0)


    def convolution(station, weight_func, sigma):
        node = get_space_indexes(position2node_index, observed_stations[[station]])
        w = weight_functions[weight_func](distances_betw_nodes[node], sigma).ravel()
        w = w / np.nansum(w)
        traffic_by_node_new = np.einsum("tnc,n->tc", traffic_by_node[times], w)
        res = dict()
        for c in TRAFFIC_VALUES:
            res[f"cor_{c}"] = correlation(station, traffic_by_node_new, c)
            res[f"cor_noavg_{c}"] = correlation_no_avg(station, traffic_by_node_new, c)
        return res


    lab = LabPipeline()
    lab.define_new_block_of_functions(
        "experiment",
        convolution,
        recalculate=False
    )

    sigma = np.round(np.logspace(np.log10(0.016), np.log10(0.16), 10), decimals=3).tolist()
    sigma = sigma + np.round(np.logspace(np.log10(0.001), np.log10(0.16), 10), decimals=3).tolist()
    lab.execute(
        data_manager,
        num_cores=15,
        forget=False,
        save_on_iteration=15,
        station=stations2test,
        weight_func=list(weight_functions.keys()),
        sigma=sigma
    )

    def plot(**kwargs):
        sns.lineplot(**kwargs, marker="o")
        kwargs["ax"].axvline(0.005, linestyle="--", color="r")
        kwargs["ax"].axhline(0, linestyle="--", color="gray")

    for c in TRAFFIC_VALUES:
        for avg in ["", "_noavg"]:
            generic_plot(
                data_manager,
                x="sigma",
                y=f"cor{avg}_{c}",
                axes_by="weight_func",
                label="station",
                log="x",
                plot_func=plot,
                xticks=sigma
            )
