from functools import partial

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from spiderplot import spiderplot

# from spektral.layers import GATConv

import src.config as config
from PerplexityLab.DataManager import DataManager, dmfilter, apply
from PerplexityLab.LabPipeline import LabPipeline
from src.experiments.PreProcess import longer_distance, train_test_model, station_coordinates, \
    distance_between_stations_pixels, train_test_averagers, simulation, stations2test
from src.experiments.config_experiments import num_cores, shuffle, filter_graph
from src.lib.DataProcessing.TrafficProcessing import load_background
from src.lib.Models.BaseModel import Bounds, mse, UNIFORM, ModelsSequenciator, \
    ModelsAverager, LOGUNIFORM, medianse, GRAD, Optim, CMA, NONE_OPTIM_METHOD
from src.lib.Models.TrueStateEstimationModels.AverageModels import SnapshotMeanModel, GlobalMeanModel
from src.lib.Models.TrueStateEstimationModels.GCNN import GraphCNN
from src.lib.Models.TrueStateEstimationModels.GraphModels import HEqStaticModel, GraphEmissionsModel, \
    GraphEmissionsNeigEdgeModel
from src.lib.Models.TrueStateEstimationModels.TrafficConvolution import TrafficMeanModel, TrafficConvolutionModel, \
    gaussker
from PerplexityLab.miscellaneous import NamedPartial, if_true_str, partial_filter
from PerplexityLab.visualization import generic_plot, save_fig

if __name__ == "__main__":
    niter = 100
    experiment_name = f"TrafficGraphModelComparisonAvgRFSeqLoopFit{if_true_str(shuffle, '_Shuffled')}" \
                      f"{if_true_str(simulation, '_Sim')}{if_true_str(filter_graph, '_Gfiltered')}"

    data_manager = DataManager(
        path=config.results_dir,
        name=experiment_name,
        trackCO2=True
    )

    base_models = [
        SnapshotMeanModel(summary_statistic="mean"),
        GlobalMeanModel()
    ]
    # 621.5069384089682 = [2.87121906 0.16877082 1.04179242 1.23798909 3.42959526 3.56328527]
    models = [
        # GraphEmissionsModel(
        #     name="t1g1",
        #     tau=1,
        #     gamma=1,
        #     # tau=Optim(1, 0.01, 1),
        #     # gamma=Optim(1, 0, 1),
        #     k_neighbours=3,
        #     # green=Optim(1.04179242, None, None), yellow=Optim(1.23798909, None, None),
        #     # red=Optim(3.42959526, None, None), dark_red=Optim(3.56328527, None, None),
        #     niter=100, verbose=True, fit_intercept=True,
        #     optim_method=NONE_OPTIM_METHOD,
        #     loss=medianse),
        # ModelsSequenciator(models=[
        #     SnapshotMeanModel(summary_statistic="mean"),
        #     GraphCNN(
        #         name="GCNN",
        #         spektral_layer=partial_filter(GATConv, attn_heads=1, concat_heads=True, dropout_rate=0.5,
        #                                       return_attn_coef=False, add_self_loops=True, activation="relu",
        #                                       use_bias=True),
        #         hidden_layers=(10,),
        #         loss=medianse,
        #         epochs_to_stop=5000,
        #         experiment_dir=data_manager.path,
        #         verbose=False,
        #         niter=2,
        #     )
        # ]),
        # ModelsSequenciator(models=[
        #     SnapshotMeanModel(summary_statistic="mean"),
        #     GraphEmissionsModel(
        #         name="tau082g0",
        #         # 'tau': 0.8200031737805101, 'gamma': 0.0
        #         tau=0.82,
        #         gamma=0,
        #         # tau=Optim(1, 0.01, 1),
        #         # gamma=Optim(1, 0, 1),
        #         k_neighbours=3,
        #         model=LinearRegression(),
        #         # green=Optim(1.04179242, None, None), yellow=Optim(1.23798909, None, None),
        #         # red=Optim(3.42959526, None, None), dark_red=Optim(3.56328527, None, None),
        #         niter=2, verbose=True,
        #         optim_method=NONE_OPTIM_METHOD,
        #         loss=medianse)
        # ]),
        # ModelsSequenciator(models=[
        #     SnapshotMeanModel(summary_statistic="mean"),
        #     GraphEmissionsModel(
        #         name="tau082g0",
        #         # 'tau': 0.8200031737805101, 'gamma': 0.0
        #         tau=0.82,
        #         gamma=0,
        #         # tau=Optim(1, 0.01, 1),
        #         # gamma=Optim(1, 0, 1),
        #         k_neighbours=3,
        #         model=RandomForestRegressor(n_estimators=10, max_depth=4),
        #         # green=Optim(1.04179242, None, None), yellow=Optim(1.23798909, None, None),
        #         # red=Optim(3.42959526, None, None), dark_red=Optim(3.56328527, None, None),
        #         niter=2, verbose=True,
        #         optim_method=NONE_OPTIM_METHOD,
        #         loss=medianse)
        # ]),
        ModelsSequenciator(models=[
            SnapshotMeanModel(summary_statistic="mean"),
            GraphEmissionsNeigEdgeModel(
                k_neighbours=5,
                model=Pipeline(steps=[("LR", LinearRegression())]),
                niter=2, verbose=True,
                optim_method=NONE_OPTIM_METHOD,
                loss=medianse)
        ]),
        # ModelsSequenciator(models=[
        #     SnapshotMeanModel(summary_statistic="mean"),
        #     GraphEmissionsNeigEdgeModel(
        #         k_neighbours=5,
        #         model=Pipeline(steps=[("RF", RandomForestRegressor(n_estimators=10, max_depth=5))]),
        #         niter=2, verbose=True,
        #         optim_method=NONE_OPTIM_METHOD,
        #         loss=medianse)
        # ]),
        # ModelsSequenciator(models=[
        #     SnapshotMeanModel(summary_statistic="mean"),
        #     GraphEmissionsNeigEdgeModel(
        #         k_neighbours=5,
        #         model=Pipeline(steps=[("Poly2", PolynomialFeatures(degree=2)), ("LR", LinearRegression())]),
        #         niter=2, verbose=True,
        #         optim_method=NONE_OPTIM_METHOD,
        #         loss=medianse)
        # ]),
        # ModelsSequenciator(models=[
        #     SnapshotMeanModel(summary_statistic="mean"),
        #     GraphEmissionsNeigEdgeModel(
        #         k_neighbours=5,
        #         model=Pipeline(steps=[("Poly2", PolynomialFeatures(degree=2)),
        #                               ("RF", RandomForestRegressor(n_estimators=10, max_depth=5))]),
        #         niter=2, verbose=True,
        #         optim_method=NONE_OPTIM_METHOD,
        #         loss=medianse)
        # ]),

        # ModelsSequenciator(models=[
        #     SnapshotMeanModel(summary_statistic="mean"),
        #     GraphEmissionsModel(
        #         name="t1g1",
        #         tau=1, gamma=1, k_neighbours=1,
        #         # green=Optim(1.04179242, None, None), yellow=Optim(1.23798909, None, None),
        #         # red=Optim(3.42959526, None, None), dark_red=Optim(3.56328527, None, None),
        #         loss=medianse)
        #     # HEqStaticModel(
        #     #     absorption=0,
        #     #     diffusion=0,
        #     #     green=Optim(1.04179242, None, None), yellow=Optim(1.23798909, None, None),
        #     #     red=Optim(3.42959526, None, None), dark_red=Optim(3.56328527, None, None),
        #     #
        #     #     # absorption=2.87121906,
        #     #     # diffusion=0.16877082,
        #     #     # green=1.04179242, yellow=1.23798909,
        #     #     # red=3.42959526, dark_red=3.56328527,
        #     #
        #     #     # absorption=Optim(start=2.87121906, lower=0, upper=np.inf),
        #     #     # diffusion=Optim(start=0.16877082, lower=0, upper=np.inf),
        #     #     # green=Optim(1.04179242, None, None), yellow=Optim(1.23798909, None, None),
        #     #     # red=Optim(3.42959526, None, None), dark_red=Optim(3.56328527, None, None),
        #     #     name="NormalizedEmissions", loss=medianse, optim_method=CMA, verbose=True,
        #     #     niter=niter, sigma0=1e-2,
        #     #     k_neighbours=None)
        # ]),
    ]

    lab = LabPipeline()
    lab.define_new_block_of_functions("train_individual_models", *list(map(train_test_model,
                                                                           # base_models +
                                                                           models
                                                                           )))
    lab.define_new_block_of_functions("model",
                                      *list(map(partial(train_test_averagers, positive=False, fit_intercept=True),
                                                [[model] for model in models + base_models] +
                                                [models + base_models]
                                                )))

    lab.execute(
        data_manager,
        num_cores=15,
        forget=False,
        recalculate=True,
        save_on_iteration=1,
        station=stations2test  # station_coordinates.columns.to_list()[:2]
    )

    # ----- Plotting results ----- #
    generic_plot(data_manager, x="station", y="mse", label="model", plot_func=sns.barplot,
                 sort_by=["mse"],
                 mse=lambda error: np.sqrt(error.mean()),
                 ylim=(0, 60))

    generic_plot(data_manager, x="l1_error", y="model", plot_func=NamedPartial(sns.boxenplot, orient="horizontal"),
                 sort_by=["mse"],
                 l1_error=lambda error: np.abs(error).ravel(),
                 mse=lambda error: np.sqrt(error.mean()),
                 # xlim=(0, 20)
                 )

    generic_plot(data_manager, x="mse", y="model",
                 plot_func=NamedPartial(sns.violinplot, orient="horizontal", inner="stick"),
                 sort_by=["mse"],
                 mse=lambda error: np.sqrt(np.nanmean(error)),
                 xlim=(0, 100),
                 # model=["SnapshotMeanModelmean", "A+SMM,GMM,SMMTCMN", "GlobalMeanModelmean"]
                 )

    generic_plot(data_manager, x="model", y="time_to_fit", plot_func=sns.boxenplot)
    generic_plot(data_manager, x="model", y="time_to_estimate", plot_func=sns.boxenplot)

    generic_plot(data_manager, x="station", y="mse", label="model", plot_func=spiderplot,
                 mse=lambda error: np.sqrt(error.mean()))

    generic_plot(data_manager, x="station", y="l1_error", label="model", plot_func=sns.boxenplot,
                 sort_by=["mse"],
                 l1_error=lambda error: np.abs(error).ravel(),
                 mse=lambda error: np.sqrt(error.mean()),
                 ylim=(0, 100))
