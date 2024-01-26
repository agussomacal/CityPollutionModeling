import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from tqdm import tqdm

from PerplexityLab.miscellaneous import check_create_path
from PerplexityLab.visualization import save_fig
from src import config
from src.experiments.paper_experiments.PreProcessPaper import traffic_by_edge, graph, times_all, station_coordinates, \
    pollution_past, traffic_past, pollution_future
from src.experiments.paper_experiments.SourceModels import data_manager
from src.experiments.paper_experiments.params4runs import stations2test
from src.lib.DataProcessing.TrafficProcessing import TRAFFIC_VALUES, TRAFFIC_COLORS
from src.lib.FeatureExtractors.GraphFeatureExtractors import get_graph_node_positions
from src.lib.Models.TrueStateEstimationModels.PhysicsModel import get_traffic_by_node_conv, get_nearest_node_mapping, \
    get_space_indexes, get_time_indexes, get_traffic_by_node_by_dist_conv, distance_between_nodes, \
    get_traffic_by_node_by_dist_weight_conv
from src.lib.visualization_tools import plot_correlation_clustermap

experiment_name = "PollutionTraffic_statistics"
if __name__ == "__main__":
    with data_manager.track_emissions("Emissions for pollution traffic statistics."):
        path4preprocess = data_manager.path
        redo_preprocessing = False
        observed_stations = station_coordinates
        observed_pollution = pollution_past.loc[:, station_coordinates.columns]  # same columns order as locations
        observed_pollution_future = pollution_future.loc[:,
                                    station_coordinates.columns]  # same columns order as locations
        traffic = traffic_past

        # source_dist = get_traffic_by_node_by_dist_weight_conv(
        #     path=path4preprocess, times=times_all,
        #     traffic_by_edge=traffic_by_edge,
        #     graph=graph, recalculate=redo_preprocessing,
        #     distances_betw_nodes=distance_between_nodes(path=path4preprocess, recalculate=redo_preprocessing,
        #                                                 graph=graph),
        #     weight_func=lambda d, sigma: np.exp(-d ** 2 / sigma ** 2 / 2), wf_params={"sigma": 0.01})

        source_dist = get_traffic_by_node_by_dist_weight_conv(
            path=path4preprocess, times=times_all, filename="get_traffic_by_node_by_dist_weight_conv005",
            traffic_by_edge=traffic_by_edge,
            graph=graph, recalculate=redo_preprocessing,
            distances_betw_nodes=distance_between_nodes(path=path4preprocess, recalculate=redo_preprocessing,
                                                        graph=graph),
            weight_func=lambda d, sigma: np.exp(-d ** 2 / sigma ** 2 / 2), wf_params={"sigma": 0.005})

        # source_dist = get_traffic_by_node_by_dist_conv(
        #     path=path4preprocess, times=times_all,
        #     traffic_by_edge=traffic_by_edge,
        #     graph=graph, recalculate=redo_preprocessing,
        #     distances_betw_nodes=distance_between_nodes(path=path4preprocess, recalculate=redo_preprocessing,
        #                                                 graph=graph),
        #     distance_threshold=0.2)
        # [#times, #nodes, #traffic colors]

        # traffic_by_node = get_traffic_by_node_conv(
        #     path=path4preprocess, times=times_all,
        #     traffic_by_edge=traffic_by_edge,
        #     graph=graph, recalculate=redo_preprocessing, lnei=1)
        # # [#times, #nodes, #traffic colors]

        node_positions = get_graph_node_positions(graph)
        position2node_index = get_nearest_node_mapping(path=path4preprocess, filename=f"pos2node",
                                                       target_positions=observed_stations,
                                                       graph=graph, recalculate=redo_preprocessing)
        times = get_time_indexes(times_all, observed_pollution.index)
        nodes = get_space_indexes(position2node_index, observed_stations)

        if False:
            plotting_stations = stations2test
            # plotting_stations = observed_pollution.columns
            # plotting_stations = stations2test
            cor = dict()
            cor_no_avg = dict()
            d = dict()
            for station in tqdm(plotting_stations):
                pollution = observed_pollution[station]
                d[station] = cdist(node_positions, [station_coordinates[station].values]).ravel()
                order = np.argsort(d[station])
                d[station] = d[station][order]

                cor[station] = {
                    traf: np.nanmean(((pollution - np.mean(pollution)) / np.std(pollution))[:, np.newaxis] *
                                     (traffic_by_node[times][:, order, i] - np.mean(traffic_by_node[times][:, order, i],
                                                                                    axis=0,
                                                                                    keepdims=True)) / np.std(
                        traffic_by_node[times][:, order, i], axis=0, keepdims=True), axis=0)
                    for i, traf in enumerate(TRAFFIC_COLORS)}
                pollution = pollution - np.mean(observed_pollution[plotting_stations].values, axis=1)
                cor_no_avg[station] = {
                    traf: np.nanmean(((pollution - np.mean(pollution)) / np.std(pollution))[:, np.newaxis] *
                                     (traffic_by_node[times][:, order, i] - np.mean(traffic_by_node[times][:, order, i],
                                                                                    axis=0,
                                                                                    keepdims=True)) / np.std(
                        traffic_by_node[times][:, order, i], axis=0, keepdims=True), axis=0)
                    for i, traf in enumerate(TRAFFIC_COLORS)}

                # cor[station] = {
                #     traf: np.corrcoef(pollution, traffic_by_node[times][:, order, i], rowvar=False)[0, 1:]
                #     for i, traf in enumerate(TRAFFIC_COLORS)}
                # pollution = pollution - np.mean(observed_pollution.values, axis=1)
                # cor_no_avg[station] = {
                #     traf: np.corrcoef(pollution,
                #                       traffic_by_node[times][:, order, i], rowvar=False)[0, 1:]
                #     for i, traf in enumerate(TRAFFIC_COLORS)}

                # cor[station] = {
                #     traf: np.array([np.corrcoef(pollution, traffic_by_node[times][:, o, i], rowvar=False)[0, 1] for o in order])
                #     for i, traf in enumerate(TRAFFIC_COLORS)}
                # pollution = pollution - np.mean(observed_pollution.values, axis=1)
                # cor_no_avg[station] = {
                #     traf: np.array([np.corrcoef(pollution,
                #                                 traffic_by_node[times][:, o, i], rowvar=False)[0, 1] for o in order])
                #     for i, traf in enumerate(TRAFFIC_COLORS)}

            with sns.color_palette("tab10", n_colors=10):
                for traf in TRAFFIC_COLORS:
                    # with save_fig(paths=check_create_path(config.results_dir, experiment_name),
                    #               filename=f"Correlation_stations_nodetraffic_distance_{traf}"):
                    #     fig, ax = plt.subplots(figsize=(10, 8))
                    #     for station in plotting_stations:
                    #         ax.plot(d[station], cor[station][traf], label=station)
                    #     ax.legend()

                    with save_fig(paths=check_create_path(config.results_dir, experiment_name),
                                  filename=f"Correlation_stations_nodetraffic_distance_cumsum_{traf}"):
                        fig, ax = plt.subplots(figsize=(10, 8))
                        for station in plotting_stations:
                            c = cor[station][traf]
                            ix = ~np.isnan(c)
                            c = c[ix]
                            c = np.cumsum(c)
                            ax.plot(d[station][ix], c / (1 + np.arange(len(c))), label=station)
                        ax.legend(loc="right")

                    with save_fig(paths=check_create_path(config.results_dir, experiment_name),
                                  filename=f"Correlation_stations_nodetraffic_distance_cumsum_avg_{traf}"):
                        fig, ax = plt.subplots(figsize=(10, 8))
                        for station in plotting_stations:
                            c = cor_no_avg[station][traf]
                            ix = ~np.isnan(c)
                            c = c[ix]
                            c = np.cumsum(c)
                            ax.plot(d[station][ix], c / (1 + np.arange(len(c))), label=station)
                        ax.legend(loc="right")

        source = source_dist.copy()
        # source = traffic_by_node.copy()
        source[np.isnan(source)] = 0

        nodes = get_space_indexes(position2node_index, observed_stations)
        for i, trf in enumerate(TRAFFIC_VALUES):
            with save_fig(check_create_path(config.results_dir, experiment_name),
                          f"CorrelationClustermap_traffic_{trf}",
                          dpi=300):
                corr = pd.DataFrame(source[times][:, nodes, i], columns=observed_stations.columns).corr()
                plot_correlation_clustermap(corr, cmap="autumn", linewidths=0.5, annot=True, annot_size=10,
                                            figsize=(10, 10))
        with save_fig(check_create_path(config.results_dir, experiment_name), f"CorrelationClustermap_pollution",
                      dpi=300):
            plot_correlation_clustermap(pollution_past.corr(), cmap="autumn", linewidths=0.5, annot=True, annot_size=10,
                                        figsize=(10, 10))
        with save_fig(check_create_path(config.results_dir, experiment_name), f"CorrelationClustermap_pollution_noavg",
                      dpi=300):
            plot_correlation_clustermap((pollution_past - np.mean(pollution_past.values, axis=1, keepdims=True)).corr(),
                                        cmap="autumn", linewidths=0.5, annot=True, annot_size=10,
                                        figsize=(10, 10))

        sdkubv
        # ========== Testing predictions ========== #
        # deg = np.array(list(zip(*nx.degree(graph.to_undirected())))[1])
        # stations = observed_pollution.columns
        stations = ['OPERA', 'BASCH', 'BONAP', 'CELES', 'ELYS', 'PA07', 'PA12', 'PA13', 'PA18', 'HAUS', 'PA15L']
        times = get_time_indexes(times_all, observed_pollution.index)
        times_f = get_time_indexes(times_all, observed_pollution_future.index)

        # for station in stations:
        #     cols = [c for c in stations if c != station]
        #     for c in cols:
        #         LinearRegression(fit_intercept=False).fit(pollution_past[], pollution_past[c])

        mse = []
        for station in stations:
            cols = [c for c in stations if c != station]
            nodes = get_space_indexes(position2node_index, observed_stations[cols])
            # source_model = MLPRegressor(hidden_layer_sizes=(20, 20),
            #                             activation="relu",  # 'relu',
            #                             learning_rate_init=0.1,
            #                             learning_rate='adaptive',
            #                             early_stopping=True,
            #                             solver="adam",
            #                             max_iter=1000)

            # ----- fit ------
            source_model = Pipeline([("PF", PolynomialFeatures(degree=2)),
                                     ("LR", LassoCV(selection="cyclic", positive=False,
                                                    # cv=len(stations)
                                                    ))])
            # source_model = RandomForestRegressor(n_estimators=25, max_depth=3)
            # source_model = LassoCV(selection="random", positive=False)
            avg = np.mean(observed_pollution[cols].values, axis=1, keepdims=True)
            avg_traffic = np.nanmean(source[times][:, :, :], axis=1, keepdims=False)
            avg_nodes_traffic = np.nanmean(source[times][:, nodes, :], axis=1, keepdims=False)
            # known = source[times][:, nodes, :].reshape((-1, 4), order="F")
            # known = np.concatenate([
            #     (source[times][:, nodes, :]).reshape((-1, 4), order="F"),
            #     np.transpose([avg.ravel()] * len(nodes)).reshape((-1, 1), order="F")],
            #     axis=-1)
            known = np.concatenate([
                (source[times][:, nodes, :]).reshape((-1, 4), order="F"),
                np.transpose([avg_traffic] * len(nodes), (1, 0, 2)).reshape((-1, 4), order="F"),
                np.transpose([avg_nodes_traffic] * len(nodes), (1, 0, 2)).reshape((-1, 4), order="F"),
                np.transpose([avg.ravel()] * len(nodes), (1, 0)).reshape((-1, 1), order="F")],
                axis=-1)
            # target = (observed_pollution[cols].values).reshape((-1, 1), order="F")
            target = (observed_pollution[cols].values - avg).reshape((-1, 1), order="F")
            source_model.fit(known, target)

            # savg = np.mean(observed_pollution[cols].values, axis=0, keepdims=True)
            # sstd = np.std(observed_pollution[cols].values, axis=0, keepdims=True)

            # ----- predict -----
            # print(source_model.coef_)
            avg = np.mean(observed_pollution_future[cols].values, axis=1, keepdims=True)
            node = get_space_indexes(position2node_index, observed_stations[[station]])
            avg_traffic = np.nanmean(source[times_f][:, :, :], axis=1, keepdims=False)
            avg_nodes_traffic = np.nanmean(source[times_f][:, nodes, :], axis=1, keepdims=False)
            # known = source[times_f][:, node, :].reshape((-1, 4), order="F")
            known = np.concatenate([
                source[times_f][:, node, :].reshape((-1, 4), order="F"),
                np.transpose([avg_traffic] * len(node), (1, 0, 2)).reshape((-1, 4), order="F"),
                np.transpose([avg_nodes_traffic] * len(node), (1, 0, 2)).reshape((-1, 4), order="F"),
                np.transpose([avg.ravel()] * len(node), (1, 0)).reshape((-1, 1), order="F")],
                axis=-1)
            predictions = source_model.predict(known)

            # np.quantile(observed_pollution[cols].values)
            # np.median((observed_pollution_future[[station]].values - savg)/sstd, axis=1)
            mse.append([
                np.mean((predictions - (observed_pollution_future[[station]].values - avg).ravel(order="F")) ** 2),
                # np.mean((predictions - observed_pollution_future[[station]].values.ravel(order="F")) ** 2),
                np.mean((observed_pollution_future[[station]].values - avg).ravel(order="F") ** 2),
                # np.mean(().ravel(order="F") ** 2)
            ])

        error = pd.DataFrame(np.sqrt(mse), index=stations, columns=["With traffic", "Spatial avg"])
        error["difference"] = np.round(error["With traffic"] - error["Spatial avg"], decimals=2)
        print(error.sort_values(by="difference"))

        # plt.scatter(source[times_f][:, node, 0].reshape((-1, 1), order="F"),
        #             (observed_pollution_future[[station]].values - avg).ravel(order="F"))
        # plt.show()

        # Poly 2 0.05 weight adding avg and traffic avgs in input
        # OPERA      9.052433    12.441966       -3.39
        # PA15L      9.653306    12.187942       -2.53
        # HAUS       8.720527    10.653934       -1.93
        # PA13       8.294594     9.260490       -0.97
        # PA07       7.655115     8.311913       -0.66
        # BONAP      5.164147     5.648023       -0.48
        # BASCH     11.239144    11.363024       -0.12
        # CELES      7.120992     7.241304       -0.12
        # ELYS       4.925601     4.952627       -0.03
        # PA18       8.674873     6.662902        2.01
        # PA12       9.364176     5.730246        3.63

        # Poly 2 0.05 weight adding avg in input
        # With traffic  Spatial avg  difference
        # OPERA     59.317524   131.770897  -72.453373
        # HAUS      58.411830    97.571465  -39.159635
        # PA13      79.559209   105.702521  -26.143312
        # PA07      72.745427    88.799543  -16.054117
        # ELYS      21.882838    30.083210   -8.200372
        # BONAP     27.101770    27.592710   -0.490941
        # CELES     59.673992    50.666131    9.007861
        # BASCH    131.923096   113.406323   18.516773
        # PA12      57.779721    35.552042   22.227679
        # PA18      88.091514    57.057654   31.033860

        # Poly 2 0.05
        #        With traffic  Spatial avg  difference
        # OPERA     58.785107   131.770897  -72.985790
        # HAUS      59.769997    97.571465  -37.801468
        # PA13      77.373725   105.702521  -28.328795
        # PA07      70.630632    88.799543  -18.168911
        # ELYS      22.457441    30.083210   -7.625769
        # BONAP     28.203666    27.592710    0.610956
        # PA12      44.076077    35.552042    8.524035
        # CELES     59.443178    50.666131    8.777048
        # BASCH    138.836670   113.406323   25.430346
        # PA18      85.820430    57.057654   28.762777

        # Poly 2
        #        With traffic  Spatial avg  difference
        # OPERA     81.061860   131.770897  -50.709038
        # HAUS      63.387149    97.571465  -34.184316
        # PA13      75.905676   105.702521  -29.796845
        # PA07      81.122598    88.799543   -7.676945
        # ELYS      24.756391    30.083210   -5.326819
        # BONAP     27.523956    27.592710   -0.068754

        # CELES     54.695408    50.666131    4.029278
        # PA18      76.168723    57.057654   19.111069
        # BASCH    176.167469   113.406323   62.761145
        # PA12     100.329446    35.552042   64.777404

        # Poly 1
        #        With traffic  Spatial avg  difference
        # PA13      96.723699   105.702521   -8.978822
        # BONAP     26.239244    27.592710   -1.353466
        # CELES     49.790547    50.666131   -0.875584
        # ELYS      29.786035    30.083210   -0.297174

        # OPERA    131.770897   131.770897    0.000000
        # HAUS      97.571465    97.571465    0.000000

        # PA07      88.820763    88.799543    0.021220
        # BASCH    119.836696   113.406323    6.430372
        # PA18      69.870135    57.057654   12.812481
        # PA12     151.283303    35.552042  115.731261
