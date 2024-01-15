import subprocess
from collections import OrderedDict
from datetime import datetime
from functools import partial

import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from spiderplot import spiderplot

import src.config as config
from PerplexityLab.mlutils.scikit_keras import SkKerasRegressor
from PerplexityLab.DataManager import DataManager
from PerplexityLab.LabPipeline import LabPipeline
from PerplexityLab.miscellaneous import NamedPartial, copy_main_script_version
from PerplexityLab.visualization import generic_plot, perplex_plot
from src.experiments.paper_experiments.PreProcessPaper import train_test_model, stations2test, \
    plot_pollution_map_in_graph, times_future, graph, pollution_past, station_coordinates, times_all, traffic_by_edge
from src.experiments.paper_experiments.params4runs import path2latex_figures, runsinfo
from src.lib.FeatureExtractors.GraphFeatureExtractors import label_prop, diffusion_eq
from src.lib.Models.BaseModel import ModelsSequenciator, \
    medianse, NONE_OPTIM_METHOD, mse, GRAD, CMA
from src.lib.Models.SensorDependentModels.BLUEFamily import BLUEModel
from src.lib.Models.TrueStateEstimationModels.AverageModels import SnapshotMeanModel, GlobalMeanModel
from src.lib.Models.TrueStateEstimationModels.GraphModels import GraphEmissionsNeigEdgeModel, HEqStaticModel
from src.lib.Models.TrueStateEstimationModels.KernelModels import GaussianKernelModel, ExponentialKernelModel
from src.lib.Models.TrueStateEstimationModels.PhysicsModel import PhysicsModel
from src.lib.Modules import Optim

from sklearn.base import BaseEstimator, TransformerMixin

from src.lib.tools import IdentityTransformer

if __name__ == "__main__":
    k_neighbours = 10
    hidden_layer_sizes = (20, 20,)
    activation = "logistic"
    learning_rate_init = 0.1
    learning_rate = "adaptive"
    early_stopping = True
    solver = "Adam"
    max_iter = 10000
    runsinfo.append_info(
        kneighbours=k_neighbours,
        hiddenlayers=len(hidden_layer_sizes),
        neurons=hidden_layer_sizes[0] if len(set(hidden_layer_sizes)) == 1 else hidden_layer_sizes,
        activation=activation,
        solver=solver
    )
    # experiment_name = f"MapExtraRegressors{if_true_str(shuffle, '_Shuffled')}" \
    #                   f"{if_true_str(simulation, '_Sim')}{if_true_str(filter_graph, '_Gfiltered')}"

    data_manager = DataManager(
        path=config.paper_experiments_dir,
        emissions_path=config.results_dir,
        name="Experimentation",
        country_alpha_code="FR",
        trackCO2=True
    )
    copy_main_script_version(__file__, data_manager.path)

    models = {
        "Spatial Avg":
            SnapshotMeanModel(summary_statistic="mean"),
        "Kernel":
            ExponentialKernelModel(alpha=Optim(start=None, lower=0.001, upper=0.5),
                                   beta=Optim(start=np.log(1), lower=np.log(0.01), upper=np.log(2)),
                                   distrust=0,
                                   name="", loss=mse, optim_method=GRAD, niter=1000, verbose=False),
        "BLUE":
            BLUEModel(name="BLUE", loss=mse, optim_method=NONE_OPTIM_METHOD, niter=1000, verbose=False),
        # "BLUE distrust":
        #     BLUEModel(name="BLUEIU", sensor_distrust=Optim(start=0, lower=0, upper=1),
        #               loss=mse, optim_method=GRAD, niter=1000, verbose=False),
        # "BLUE individual distrust":
        #     BLUEModel(name="BLUEI",
        #               sensor_distrust={c: Optim(start=0, lower=0, upper=1) for c in
        #                                pollution_past.columns},
        #               loss=mse, optim_method=GRAD, niter=1000, verbose=False),
        "Physics":
            PhysicsModel(path4preprocess=data_manager.path, graph=graph,
                         spacial_locations=station_coordinates, times=times_all,
                         traffic_by_edge=traffic_by_edge,
                         lnei=1,
                         source_model=Pipeline([("Poly", PolynomialFeatures(degree=2)),
                                                ("Lss", LassoCV(selection="random", positive=True))]),
                         substract_mean=False,
                         extra_regressors=[],
                         # extra_regressors=["temperature", "wind"],
                         k_max=10,
                         # k=Optim(start=10.0, lower=1, upper=10),
                         k=10,
                         redo_preprocessing=False,
                         name="", loss=mse, optim_method=GRAD,
                         verbose=True, niter=10, sigma0=1,
                         # absorption=Optim(start=0.004837509240306621, lower=0, upper=None),
                         alpha=Optim(1, 0, 1),
                         absorption=0.005,
                         diffusion=1000,
                         # alpha=0.0001,
                         ),
        "Physics-avg":
            PhysicsModel(path4preprocess=data_manager.path, graph=graph,
                         spacial_locations=station_coordinates, times=times_all,
                         traffic_by_edge=traffic_by_edge,
                         source_model=Pipeline(
                             steps=[("Poly", PolynomialFeatures(degree=2)),
                                    ("Lasso", LassoCV(selection="random"))]),
                         substract_mean=True,
                         lnei=1,
                         extra_regressors=[],
                         # extra_regressors=["temperature", "wind"],
                         k_max=10,
                         # k=Optim(start=10.0, lower=1, upper=10),
                         k=10,
                         redo_preprocessing=False,
                         name="zlasso", loss=mse, optim_method=GRAD,
                         verbose=True, niter=500, sigma0=1,
                         # absorption=Optim(start=30, lower=0, upper=None),
                         # diffusion=Optim(start=1000, lower=0, upper=None),
                         absorption=30,
                         diffusion=1000,
                         alpha=Optim(1, 0, 1),
                         ),
        "Physics-seq":
            ModelsSequenciator(
                name="LR",
                models=[
                    SnapshotMeanModel(summary_statistic="mean"),
                    PhysicsModel(path4preprocess=data_manager.path, graph=graph,
                                 spacial_locations=station_coordinates, times=times_all,
                                 traffic_by_edge=traffic_by_edge,
                                 lnei=2,
                                 source_model=LassoCV(selection="random", positive=True),
                                 substract_mean=False,
                                 extra_regressors=[],
                                 # extra_regressors=["temperature", "wind"],
                                 k_max=10,
                                 # k=Optim(start=10.0, lower=1, upper=10),
                                 k=10,
                                 redo_preprocessing=False,
                                 name="", loss=mse, optim_method=GRAD,
                                 verbose=True, niter=10, sigma0=1,
                                 absorption=Optim(0.78, 0, None),
                                 diffusion=Optim(4000, 0, None),
                                 alpha=Optim(start=0.003, lower=0, upper=1),
                                 ),
                ],
                transition_model=[
                    Pipeline([("LR", LinearRegression())]),
                    Pipeline([("Lss", LassoCV(selection="cyclic"))])
                ]
            ),
        "Physics-zNN":
            PhysicsModel(path4preprocess=data_manager.path, graph=graph,
                         spacial_locations=station_coordinates, times=times_all,
                         traffic_by_edge=traffic_by_edge,
                         source_model=Pipeline(
                             steps=[("zscore", StandardScaler()),
                                    ("NN",
                                     MLPRegressor(hidden_layer_sizes=hidden_layer_sizes,
                                                  activation=activation,  # 'relu',
                                                  learning_rate_init=learning_rate_init,
                                                  learning_rate=learning_rate,
                                                  early_stopping=early_stopping,
                                                  solver=solver.lower(),
                                                  max_iter=max_iter))]),
                         extra_regressors=[],
                         # extra_regressors=["temperature", "wind"],
                         k_max=10,
                         # k=Optim(start=10.0, lower=1, upper=10),
                         k=10,
                         redo_preprocessing=False,
                         name="zNN", loss=mse, optim_method=GRAD,
                         verbose=True, niter=500, sigma0=1,
                         absorption=Optim(start=6.240205, lower=0, upper=None),
                         # diffusion=Optim(start=-0.5981040000835097, lower=None, upper=None),
                         diffusion=Optim(start=0.598104, lower=0, upper=None),
                         # alpha=Optim(start=1.9263212966429901, lower=0, upper=2),
                         # alpha=Optim(start=0.000658, lower=0, upper=1),
                         alpha=Optim(start=1, lower=0, upper=1),
                         # alpha=1,
                         # green=Optim(start=2.58609, lower=None, upper=None),
                         # yellow=Optim(start=8.420218, lower=None, upper=None),
                         # red=Optim(start=5.643795, lower=None, upper=None),
                         # dark_red=Optim(start=6.794929, lower=None, upper=None)
                         ),
        # "Pysics + Kernel":
        #     ModelsSequenciator(
        #         name="PhysicsKriging",
        #         models=[
        #             PhysicsModel(path4preprocess=data_manager.path, graph=graph,
        #                          spacial_locations=station_coordinates, times=times_all,
        #                          traffic_by_edge=traffic_by_edge,
        #                          extra_regressors=[],
        #                          # extra_regressors=["temperature", "wind"],
        #                          k_max=10,
        #                          k=10,
        #                          redo_preprocessing=False,
        #                          name="", loss=mse, optim_method=GRAD,
        #                          verbose=True, niter=500, sigma0=1,
        #                          absorption=Optim(start=6.240205, lower=0, upper=None),
        #                          diffusion=Optim(start=0.598104, lower=0, upper=None),
        #                          alpha=Optim(start=0.99, lower=0, upper=1),
        #                          ),
        #             ExponentialKernelModel(alpha=Optim(start=None, lower=0.001, upper=0.5),  # 0.01266096365565058
        #                                    beta=Optim(start=np.log(1), lower=np.log(0.01), upper=np.log(2)),
        #                                    distrust=0,
        #                                    name="", loss=mse, optim_method=GRAD, niter=1000, verbose=False)
        #         ],
        #         transition_model=[
        #             Pipeline([("Id", IdentityTransformer())]),
        #             Pipeline([("Lss", LassoCV(selection="cyclic"))])
        #         ]
        #     ),
        #
        # "Pysics 50":
        #     PhysicsModel(path4preprocess=data_manager.path,
        #                  # name="K",
        #                  graph=graph,
        #                  spacial_locations=station_coordinates, times=times_all,
        #                  traffic_by_edge=traffic_by_edge,
        #                  extra_regressors=["temperature", "wind"],
        #                  k_max=50,
        #                  k=50,
        #                  redo_preprocessing=False,
        #                  loss=mse, optim_method=GRAD,
        #                  verbose=True, niter=500, sigma0=1,
        #                  absorption=Optim(start=6.240205, lower=0, upper=None),
        #                  # diffusion=Optim(start=-0.5981040000835097, lower=None, upper=None),
        #                  diffusion=Optim(start=0.598104, lower=0, upper=None),
        #                  # alpha=Optim(start=1.9263212966429901, lower=0, upper=2),
        #                  # alpha=Optim(start=0.000658, lower=0, upper=1),
        #                  alpha=Optim(start=0.99, lower=0, upper=1),
        #                  # alpha=1,
        #                  # green=Optim(start=2.58609, lower=None, upper=None),
        #                  # yellow=Optim(start=8.420218, lower=None, upper=None),
        #                  # red=Optim(start=5.643795, lower=None, upper=None),
        #                  # dark_red=Optim(start=6.794929, lower=None, upper=None)
        #                  ),
        # "Kernel GlobalMean":
        #     ModelsSequenciator(
        #         name="AvgKrigging",
        #         models=[
        #             GlobalMeanModel(summary_statistic="mean"),
        #             ExponentialKernelModel(alpha=Optim(start=None, lower=0.001, upper=0.5),  # 0.01266096365565058
        #                                    beta=Optim(start=np.log(1), lower=np.log(0.01), upper=np.log(2)),
        #                                    distrust=0,
        #                                    name="", loss=mse, optim_method=GRAD, niter=1000, verbose=False)
        #         ],
        #         transition_model=[
        #             Pipeline([("Id", IdentityTransformer())]),
        #             Pipeline([("Id", IdentityTransformer())])
        #         ]
        #     ),
        "Graph Linear":
            ModelsSequenciator(
                name="LR",
                models=[
                    SnapshotMeanModel(summary_statistic="mean"),
                    GraphEmissionsNeigEdgeModel(
                        path2trafficbynode=data_manager.path,
                        extra_regressors=[],
                        k_neighbours=k_neighbours,
                        # model=Pipeline(steps=[("zscore", StandardScaler()), ("LR", LinearRegression())]),
                        model=Pipeline(steps=[("zscore", StandardScaler()), ("Lasso", LassoCV(selection="cyclic"))]),
                        niter=2, verbose=True,
                        optim_method=NONE_OPTIM_METHOD,
                        loss=medianse)
                ],
                transition_model=[
                    Pipeline([("LR", LinearRegression())]),
                    Pipeline([("Lss", LassoCV(selection="cyclic"))])
                ]
            ),
        "Graph Linear + T W":
            ModelsSequenciator(
                name="LR_Extra",
                models=[
                    SnapshotMeanModel(summary_statistic="mean"),
                    GraphEmissionsNeigEdgeModel(
                        path2trafficbynode=data_manager.path,
                        extra_regressors=["temperature", "wind"],
                        k_neighbours=k_neighbours,
                        # model=Pipeline(steps=[("zscore", StandardScaler()), ("LR", LinearRegression())]),
                        model=Pipeline(steps=[("zscore", StandardScaler()), ("Lasso", LassoCV(selection="cyclic"))]),
                        niter=2, verbose=True,
                        optim_method=NONE_OPTIM_METHOD,
                        loss=medianse)
                ],
                transition_model=[
                    Pipeline([("LR", LinearRegression())]),
                    Pipeline([("Lss", LassoCV(selection="cyclic"))])
                ]
            ),
        "Graph NN + T W":
            ModelsSequenciator(
                name="NN_Extra",
                models=[
                    SnapshotMeanModel(summary_statistic="mean"),
                    GraphEmissionsNeigEdgeModel(
                        path2trafficbynode=data_manager.path,
                        extra_regressors=["temperature", "wind"],
                        k_neighbours=k_neighbours,
                        # model=Pipeline(steps=[("zscore", StandardScaler()), ("LR", LinearRegression())]),
                        # model=Pipeline(steps=[("zscore", StandardScaler()), ("Lasso", LassoCV(selection="cyclic"))]),
                        model=Pipeline(
                            steps=[("zscore", StandardScaler()),
                                   ("NN",
                                    # SkKerasRegressor(hidden_layer_sizes=hidden_layer_sizes,
                                    #                        epochs=max_iter, activation=activation, validation_size=0.1,
                                    #                        restarts=1,
                                    #                        batch_size=0.1, criterion="mse",
                                    #                        # optimizer="Adam",
                                    #                        lr=None, lr_lower_limit=1e-12,
                                    #                        lr_upper_limit=1, n_epochs_without_improvement=100,
                                    #                        train_noise=1e-5)
                                    MLPRegressor(hidden_layer_sizes=hidden_layer_sizes,
                                                 activation=activation,  # 'relu',
                                                 learning_rate_init=learning_rate_init,
                                                 learning_rate=learning_rate,
                                                 early_stopping=early_stopping,
                                                 solver=solver.lower(),
                                                 max_iter=max_iter)
                                    )]),
                        niter=2, verbose=True,
                        optim_method=NONE_OPTIM_METHOD,
                        loss=medianse)
                ],
                transition_model=[
                    Pipeline([("LR", LinearRegression())]),
                    Pipeline([("Lss", LassoCV(selection="cyclic"))])
                ]
            ),
    }

    lab = LabPipeline()
    lab.define_new_block_of_functions(
        "individual_models",
        train_test_model(("Spatial Avg", models["Spatial Avg"])),
        train_test_model(("BLUE", models["BLUE"])),
        train_test_model(("Kernel", models["Kernel"])),
        train_test_model(("Physics", models["Physics"])),
        train_test_model(("Physics-avg", models["Physics-avg"])),
        # train_test_model(("Physics-avg-z", models["Physics-avg-z"])),
        # train_test_model(("Physics-seq", models["Physics-seq"])),
        # train_test_model(("Graph NN + T W", models["Graph NN + T W"])),
        # train_test_model(("Graph Linear + T W", models["Graph Linear + T W"])),
        # train_test_model(("Physics-avg-z", models["Physics-avg-z"])),
        # train_test_model(("Physics-zNN", models["Physics-zNN"])),
        recalculate=True
    )

    lab.execute(
        data_manager,
        num_cores=15,
        forget=False,
        save_on_iteration=None,
        # station=["BONAP"],  # stations2test
        station=stations2test
    )

    # (Pipeline([("LR", LassoCV(selection="random"))]), base_models[:2] + models),
    # (Pipeline([("LR", LassoCV(selection="random"))]), base_models[:1] + models),
    # (Pipeline([("LR", LassoCV(selection="random"))]), [models[-1]]),
    # (Pipeline([("LR", LassoCV(selection="random"))]), [models[-2]]),
    # (Pipeline([("LR", LassoCV(selection="random"))]), [models[0]]),
    # (Pipeline([("LR", LassoCV(selection="random"))]), [models[0]]),
    # (Pipeline([("LR", LassoCV(selection="random"))]), [models[1]]),
    # (Pipeline([("LR", LassoCV(selection="random"))]), models[:2]),

    # (Pipeline([("Id", IdentityTransformer())]), [models[0]]),
    # (Pipeline([("LR", LassoCV(selection="random"))]), [models[0], base_models[1]])
    # (Pipeline([("LR", LassoCV(selection="random"))]), base_models[:1] + models[-2:]),
    # (Pipeline([("LR", LassoCV(selection="random"))]), [models[-1]]),

    lab.define_new_block_of_functions(
        "individual_models",
        # train_test_model(("SL Spatial Avg", (Pipeline([("LR", LassoCV(selection="random"))]), ["Spatial Avg"]))),
        train_test_model(("SL Physics", (Pipeline([("LR", LassoCV(selection="random"))]), ["Spatial Avg", "Physics"]))),
        train_test_model(
            ("SL Physics-avg", (Pipeline([("LR", LassoCV(selection="random"))]), ["Spatial Avg", "Physics-avg"]))),

        # train_test_model(("SL Physics-seq", (Pipeline([("LR", LassoCV(selection="random"))]), ["Spatial Avg", "Physics-seq"]))),
        # train_test_model(("SL Physics + Kernel",
        #                   (Pipeline([("LR", LassoCV(selection="random"))]), ["Spatial Avg", "Physics", "Kernel"]))),
        # train_test_model(("SL Graph Linear",
        #                   (Pipeline([("LR", LassoCV(selection="random"))]), ["Spatial Avg", "Graph Linear + T W"]))),
        # train_test_model(
        #     ("SL Graph NN", (Pipeline([("LR", LassoCV(selection="random"))]), ["Spatial Avg", "Graph NN + T W"]))),
        # train_test_model(("Esemble", (Pipeline([("LR", LassoCV(selection="random"))]),
        #                               ["Spatial Avg", "Graph Linear + T W", "Graph NN + T W", "Physics", "Kernel"]))),
        # train_test_model(
        #     ("SL Physics-avg-z", (Pipeline([("LR", LassoCV(selection="random"))]), ["Spatial Avg", "Physics-avg-z"]))),

        # train_test_model(("SL Physics-z", (Pipeline([("LR", LassoCV(selection="random"))]), ["Physics-z"]))),
        # train_test_model(("SL-z Physics", (
        #     Pipeline(steps=[("zscore", StandardScaler()), ("Lasso", LassoCV(selection="random"))]),
        #     ["Physics"]))),
        # train_test_model(("SL-z Physics-z", (
        #     Pipeline(steps=[("zscore", StandardScaler()), ("Lasso", LassoCV(selection="random"))]),
        #     ["Physics-z"]))),
        # train_test_model(("SL-z Physics-zNN", (
        #     Pipeline(steps=[("zscore", StandardScaler()), ("Lasso", LassoCV(selection="random"))]),
        #     ["Physics-zNN"]))),
        recalculate=True
    )
    lab.execute(
        data_manager,
        num_cores=15,
        forget=False,
        save_on_iteration=None,
        # station=["BONAP"],  # stations2test
        station=stations2test
    )

    import DoPlots
    # subprocess.call("./DoPlots.py", shell=True)

    # # ----- Plotting results ----- #
    #
    # model_names = OrderedDict([
    #     ("ExponentialKernelModel", "Krigging"),
    #     ("AvgKrigging", "Global Average \n Krigging"),
    #     ("SnapshotMeanModelmean", "Average in space"),
    #     ("LR", "Graph \n Linear Regression"),
    #     ("LR_Extra", "Graph Temp Wind \n Linear Regression"),
    #     ("NN_Extra", "Graph Temp Wind \n NeuralNetwork"),
    # ])
    #
    # models_order = list(model_names.values()) + ["Ensemble"]
    # # models_order.remove("Krigging")
    # models2plot = set(data_manager["model"])
    # # models2plot.remove("ExponentialKernelModel")
    # models2plot = list(models2plot)
    #
    # runsinfo.append_info(
    #     average=model_names["SnapshotMeanModelmean"].replace(" \n", ""),
    #     krigging=model_names["ExponentialKernelModel"].replace(" \n", ""),
    #     avgkrigging=model_names["AvgKrigging"].replace(" \n", ""),
    #     lm=model_names["LR"].replace(" \n", ""),
    #     lmextra=model_names["LR_Extra"].replace(" \n", ""),
    #     nn=model_names["NN_Extra"].replace(" \n", ""),
    #     ensemble="Ensemble",
    # )
    #
    #
    # def name_models(model):
    #     if "Pipeline" in model:
    #         return "Ensemble"
    #     else:
    #         return model_names[model]
    #
    #
    # # plot_pollution_map_in_graph(
    # #     data_manager=data_manager,
    # #     folder=path2latex_figures,
    # #     time=times_future[4], station="OPERA",
    # #     plot_by=["model", "station"],
    # #     num_cores=10, models=name_models, model=["LR", "LR_Extra", "NN_Extra", "Ensemble"], s=10,
    # #     nodes_indexes=np.arange(len(graph)),
    # #     # nodes_indexes=np.random.choice(len(graph), size=2000, replace=False),
    # #     cmap=sns.color_palette("coolwarm", as_cmap=True), alpha=0.7, dpi=300,
    # #     format=".pdf")
    #
    # generic_plot(
    #     data_manager=data_manager,
    #     # folder=path2latex_figures,
    #     # x="models",
    #     x="model_name",
    #     y="mse",
    #     plot_func=NamedPartial(sns.barplot, orient="vertical",
    #                            # order=models_order
    #                            # , errorbar=("ci", 0)
    #                            ),
    #     # plot_func=sns.barplot,
    #     sort_by=["models"],
    #     mse=lambda error: np.NAN if error is None else np.sqrt(error.mean()),
    #     # models=name_models,
    #     # model=models2plot,
    #     dpi=300,
    #     axes_xy_proportions=(12, 8),
    #     # format=".pdf"
    # )
    #
    # stations_order = ["BONAP", "HAUS", "CELES"]
    # generic_plot(
    #     name="BadStations",
    #     data_manager=data_manager,
    #     folder=path2latex_figures,
    #     x="station", y="mse", label="models",
    #     plot_func=NamedPartial(sns.barplot, orient="vertical", hue_order=models_order,
    #                            order=stations_order),
    #     sort_by=["models"],
    #     mse=lambda error: np.NAN if error is None else np.sqrt(error.mean()),
    #     models=name_models,
    #     model=models2plot,
    #     station=stations_order,
    #     dpi=300,
    #     format=".pdf"
    # )
    #
    # stations_order = ["OPERA", "BASCH", "PA13", "PA07", "PA18", "ELYS", "PA12", ]
    # generic_plot(
    #     name="GoodStations",
    #     data_manager=data_manager,
    #     folder=path2latex_figures,
    #     x="station", y="mse", label="models",
    #     plot_func=NamedPartial(sns.barplot, orient="vertical", hue_order=models_order,
    #                            order=stations_order),
    #     sort_by=["models"],
    #     mse=lambda error: np.NAN if error is None else np.sqrt(error.mean()),
    #     models=name_models,
    #     model=models2plot,
    #     station=stations_order,
    #     dpi=300,
    #     format=".pdf"
    # )
    #
    # generic_plot(data_manager, x="l1_error", y="model",
    #              plot_func=NamedPartial(sns.boxenplot, orient="horizontal", order=stations_order),
    #              sort_by=["mse"],
    #              l1_error=lambda error: np.abs(error).ravel(),
    #              mse=lambda error: np.sqrt(error.mean()),
    #              model=models2plot
    #              # xlim=(0, 20)
    #              )
    #
    # generic_plot(data_manager, x="mse", y="model",
    #              plot_func=NamedPartial(sns.violinplot, orient="horizontal", inner="stick", order=stations_order),
    #              sort_by=["mse"],
    #              mse=lambda error: np.sqrt(np.nanmean(error)),
    #              models=name_models,
    #              model=models2plot
    #              # xlim=(0, 100),
    #              # model=["SnapshotMeanModelmean", "A+SMM,GMM,SMMTCMN", "GlobalMeanModelmean"]
    #              )
    #
    # generic_plot(data_manager, x="model", y="time_to_fit", plot_func=sns.boxenplot, model=models2plot)
    # generic_plot(data_manager, x="model", y="time_to_estimate", plot_func=sns.boxenplot, model=models2plot)
    #
    # generic_plot(data_manager, x="station", y="mse", label="models",
    #              plot_func=NamedPartial(spiderplot, hue=models_order),
    #              mse=lambda error: np.sqrt(error.mean()),
    #              models=name_models,
    #              model=models2plot
    #              )
    #
    # generic_plot(data_manager, x="station", y="l1_error", label="model", plot_func=sns.boxenplot,
    #              sort_by=["mse"],
    #              l1_error=lambda error: np.abs(error).ravel(),
    #              mse=lambda error: np.sqrt(error.mean()),
    #              model=models2plot
    #              )
    #
    # plot_pollution_map_in_graph(
    #     data_manager=data_manager,
    #     folder=path2latex_figures,
    #     time=times_future[4],
    #     # diffusion_method=partial(diffusion_eq, path=data_manager.path, graph=graph, diffusion_coef=1,
    #     #                          absorption_coef=0.01, recalculate=False),
    #     # diffusion_method=partial(label_prop, graph=graph,
    #     #                          edge_function=lambda data: 1.0 / data["length"],
    #     #                          lamb=1, iter=10, p=0.5),
    #     # diffusion_method=lambda f: label_prop(f=f + np.random.uniform(size=np.shape(f)), graph=graph,
    #     #                                       edge_function=lambda data: 1.0 / data["length"],
    #     #                                       lamb=1, iter=10, p=0.1),
    #     diffusion_method=lambda f: f,
    #     # time=times_future[4], Screenshot_48.8580073_2.3342828_13_2022_12_8_13_15
    #     # time=pollution_past.index[11],
    #     station="OPERA",
    #     plot_by=["model", "station"],
    #     num_cores=1, models=name_models, model=[
    #         "SnapshotMeanModelmean", "ExponentialKernelModel",
    #         "AvgKrigging", "LR_Extra", "NN_Extra"
    #     ],
    #     nodes_indexes=np.arange(len(graph)), s=10,
    #     cmap=sns.color_palette("plasma", as_cmap=True), alpha=0.6, dpi=300,
    #     format=".pdf")
