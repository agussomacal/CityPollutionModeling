import numpy as np
import seaborn as sns
from scipy.spatial.distance import cdist

from PerplexityLab.DataManager import DataManager
from PerplexityLab.LabPipeline import LabPipeline
from PerplexityLab.miscellaneous import NamedPartial
from PerplexityLab.visualization import generic_plot, LegendOutsidePlot
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


def threshold(d, sigma):
    w = np.zeros(len(d))
    w[d <= sigma] = 1
    return w


weight_functions = {
    exponential.__name__: exponential,
    gaussian.__name__: gaussian,
    rational.__name__: rational,
    threshold.__name__: threshold,
}

experiment_name = "PollutionTraffic_statistics"
if __name__ == "__main__":
    with data_manager.track_emissions("Emissions for pollution traffic statistics."):
        path4preprocess = data_manager.path
        redo_preprocessing = True
        observed_stations = station_coordinates
        observed_pollution = pollution_past.loc[:, station_coordinates.columns]  # same columns order as locations
        observed_pollution_future = pollution_future.loc[:,
                                    station_coordinates.columns]  # same columns order as locations

        traffic_by_node = get_traffic_by_node_conv(
            path=path4preprocess, times=times_all,
            traffic_by_edge=traffic_by_edge,
            graph=graph, recalculate=False, lnei=1)
        # [#times, #nodes, #traffic colors]

        node_positions = get_graph_node_positions(graph)
        position2node_index = get_nearest_node_mapping(path=path4preprocess, filename=f"pos2node",
                                                       target_positions=observed_stations,
                                                       graph=graph, recalculate=redo_preprocessing)
        times = get_time_indexes(times_all, observed_pollution.index)
        nodes = get_space_indexes(position2node_index, observed_stations)

        distances_betw_nodes = distance_between_nodes(path=path4preprocess, recalculate=False,
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
        w = weight_functions[weight_func](distances_betw_nodes[node].ravel(), sigma).ravel()
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

    # Linear approximation of distance: distance between Arc du Triumph and Vincennes 9,84km or
    # 0.1266096365565058 lat long
    ratio = 9840 / cdist([[48.87551413370949, 2.2944611276838867]], [[48.835641353886075, 2.414628350744604]])[0][0]

    def plot(**kwargs):
        sns.lineplot(**kwargs, marker="o")
        kwargs["ax"].axvline(0.005*ratio, linestyle="--", color="r")
        kwargs["ax"].axhline(0, linestyle="--", color="gray")



    for c in TRAFFIC_VALUES:
        for avg in ["", "_noavg"]:
            generic_plot(
                data_manager,
                x="distance",
                distance=lambda sigma: sigma * ratio,
                y=f"cor{avg}_{c}",
                plot_by="weight_func",
                label="station",
                log="x",
                plot_func=plot,
                # xticks=sigma,
                dpi=300,
                axes_xy_proportions=(10, 8),
                axis_font_dict={'color': 'black', 'weight': 'normal', 'size': 16},
                labels_font_dict={'color': 'black', 'weight': 'normal', 'size': 18},
                legend_font_dict={'weight': 'normal', "size": 18, 'stretch': 'normal'},
                font_family="amssymb",
                uselatex=True,
                xlabel=r"Distance (m)",
                ylabel=r"Correlation",
                xticks=sigma[::2],
                legend_outside_plot=LegendOutsidePlot(
                    loc='center right',
                    extra_y_top=0.01, extra_y_bottom=0.1,
                    extra_x_left=0.125, extra_x_right=0.2),
            )
