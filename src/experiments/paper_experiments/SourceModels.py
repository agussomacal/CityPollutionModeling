import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

import src.config as config
from PerplexityLab.DataManager import DataManager
from PerplexityLab.LabPipeline import LabPipeline
from PerplexityLab.miscellaneous import copy_main_script_version
from src.experiments.paper_experiments.PreProcessPaper import train_test_model, stations2test, station_coordinates, \
    graph, times_all, traffic_by_edge, pollution_past
from src.experiments.paper_experiments.params4runs import runsinfo
from src.lib.Models.BaseModel import GRAD, mse, BAYES, NONE_OPTIM_METHOD
from src.lib.Models.SensorDependentModels.BLUEFamily import BLUEModel
from src.lib.Models.TrueStateEstimationModels.AverageModels import SnapshotMeanModel
from src.lib.Models.TrueStateEstimationModels.KernelModels import ExponentialKernelModel, GaussianKernelModel, \
    DistanceModel, rational_kernel
from src.lib.Models.TrueStateEstimationModels.PhysicsModel import PCASourceModel, LaplacianSourceModel, PhysicsModel, \
    ModelsAggregator, NodeSourceModel, SoftDiffusion, ProjectionAfterSourceModel, \
    ProjectionFullSourceModel, ModelsAggregatorNoCV
from src.lib.Modules import Optim

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

    train_with_relative_error = False
    data_manager = DataManager(
        path=config.paper_experiments_dir,
        emissions_path=config.results_dir,
        name="SourceModelsRelErr" if train_with_relative_error else "SourceModels",
        country_alpha_code="FR",
        trackCO2=True
    )
    copy_main_script_version(__file__, data_manager.path)

    models = {
        "Spatial Avg":
            SnapshotMeanModel(summary_statistic="mean"),
        "Gaussian": GaussianKernelModel(
            sigma=Optim(start=None, lower=0.001, upper=1.0),
            sensor_distrust=Optim(start=None, lower=0.0, upper=1),
            name="", loss=mse, optim_method=GRAD, niter=100, verbose=True),
        # "ExponentialD":
        #     ExponentialKernelModel(
        #         alpha=None,
        #         # sensor_distrust=Optim(start=None, lower=0.0, upper=0.15),
        #         sensor_distrust={'PA12': 0.032, 'CELES': 0.089, 'BONAP': 0.015, 'OPERA': 0.013, 'PA13': 0.013,
        #                          'PA07': 0.001, 'ELYS': 0.044, 'PA18': 0.059, 'BASCH': 0.007, 'HAUS': 0.034},
        #         name="", loss=mse, optim_method=BAYES, niter=100, verbose=True),
        "Exponential":
            ExponentialKernelModel(
                alpha=Optim(start=None, lower=0.001, upper=10.0),
                sensor_distrust=Optim(start=None, lower=0.0, upper=1),
                name="", loss=mse, optim_method=GRAD, niter=100, verbose=True),
        "ExponentialFit":
            ExponentialKernelModel(
                alpha=Optim(start=None, lower=0.001, upper=10.0),
                sensor_distrust=Optim(start=None, lower=0.0, upper=1),
                name="", loss=mse, optim_method=NONE_OPTIM_METHOD, niter=100, verbose=True),
        "ExponentialOld":
            ExponentialKernelModel(
                alpha=Optim(start=None, lower=0.001, upper=10.0),
                sensor_distrust=Optim(start=None, lower=0.0, upper=1),
                name="", loss=mse, optim_method=NONE_OPTIM_METHOD, niter=100, verbose=True, old_method=True),
        "BLUE":
            BLUEModel(name="BLUE", loss=mse, optim_method=NONE_OPTIM_METHOD, niter=1000, verbose=True),
        "BLUE_DU":
            BLUEModel(name="BLUE", loss=mse, optim_method=BAYES, niter=100, verbose=True,
                      sensor_distrust=Optim(0.0, 0.0, 1.0)),
        "BLUE_DI":
            BLUEModel(name="BLUEI",
                      sensor_distrust={c: Optim(start=0.0, lower=0.0, upper=1.0) for c in pollution_past.columns},
                      loss=mse, optim_method=BAYES, niter=100, verbose=True),
        # "SourceModel_Poly1Lasso_avg":
        #     NodeSourceModel(path4preprocess=data_manager.path, graph=graph,
        #                     spacial_locations=station_coordinates, times=times_all,
        #                     traffic_by_edge=traffic_by_edge,
        #                     redo_preprocessing=False,
        #                     name="", loss=mse, optim_method=GRAD,
        #                     verbose=True, niter=10, sigma0=1,
        #                     lnei=1,
        #                     source_model=LassoCV(selection="random", positive=False),
        #                     substract_mean=True,
        #                     # extra_regressors=["temperature", "wind"],
        #                     ),
        # "SourceModel_Poly1Lasso_avg_TWHW":
        #     NodeSourceModel(path4preprocess=data_manager.path, graph=graph,
        #                     spacial_locations=station_coordinates, times=times_all,
        #                     traffic_by_edge=traffic_by_edge,
        #                     redo_preprocessing=False,
        #                     name="", loss=mse, optim_method=GRAD,
        #                     verbose=True, niter=10, sigma0=1,
        #                     lnei=1,
        #                     source_model=LassoCV(selection="random", positive=False),
        #                     substract_mean=True,
        #                     extra_regressors=["temperature", "wind", "hours", "week"],
        #                     ),
        # "SourceModel_NN_avg_TWHW":
        #     NodeSourceModel(path4preprocess=data_manager.path, graph=graph,
        #                     spacial_locations=station_coordinates, times=times_all,
        #                     traffic_by_edge=traffic_by_edge,
        #                     redo_preprocessing=False,
        #                     name="", loss=mse, optim_method=GRAD,
        #                     verbose=True, niter=10, sigma0=1,
        #                     lnei=1,
        #                     source_model=MLPRegressor(hidden_layer_sizes=hidden_layer_sizes,
        #                                               activation=activation,  # 'relu',
        #                                               learning_rate_init=learning_rate_init,
        #                                               learning_rate=learning_rate,
        #                                               early_stopping=early_stopping,
        #                                               solver=solver.lower(),
        #                                               max_iter=max_iter),
        #                     substract_mean=True,
        #                     extra_regressors=["temperature", "wind", "hours", "week"],
        #                     ),
        # "SourceModel_RF_avg_TWHW":
        #     NodeSourceModel(
        #         path4preprocess=data_manager.path, graph=graph,
        #         spacial_locations=station_coordinates, times=times_all,
        #         traffic_by_edge=traffic_by_edge,
        #         redo_preprocessing=False,
        #         name="", loss=mse, optim_method=GRAD,
        #         verbose=True, niter=10, sigma0=1,
        #         lnei=1,
        #         source_model=RandomForestRegressor(n_estimators=25, max_depth=3),
        #         substract_mean=True,
        #         extra_regressors=["temperature", "wind", "hours", "week"],
        #     ),
        "PCA_ProjectionFullSourceModel_LM_TWHW": ProjectionFullSourceModel(
            path4preprocess=data_manager.path, graph=graph,
            spacial_locations=station_coordinates,
            times=times_all,
            traffic_by_edge=traffic_by_edge,
            redo_preprocessing=False,
            name="", loss=mse, optim_method=NONE_OPTIM_METHOD,
            verbose=True, niter=25, sigma0=1,
            lnei=1, k_max=10,
            # source_model=LassoCV(selection="cyclic", positive=False, cv=12),
            # source_model=LassoCV(selection="random", positive=False),
            # source_model=Pipeline([("Poly", PolynomialFeatures(degree=2)),
            #                        ("LR", LassoCV(selection="cyclic", positive=False, cv=12))]),
            source_model=MLPRegressor(hidden_layer_sizes=hidden_layer_sizes,
                                      activation=activation,  # 'relu',
                                      learning_rate_init=learning_rate_init,
                                      learning_rate=learning_rate,
                                      early_stopping=early_stopping,
                                      solver=solver.lower(),
                                      max_iter=max_iter),
            # source_model=RandomForestRegressor(n_estimators=25, max_depth=3)
            substract_mean=True, cv_in_space=False,
            extra_regressors=["temperature", "wind", "hours", "week"],
            # basis="both",
            basis="geometrical",
            # kv=Optim(5, None, None), ky=Optim(5, None, None), kr=Optim(5, None, None), kd=Optim(5, None, None),
            kv=5, ky=5, kr=5, kd=5,
            # kv=10, ky=3, kr=3, kd=1,
            # D0=Optim(1e4, 1e-4, 1e4), A0=Optim(1e-4, 1e-4, 1e4),
            D0=0.0, A0=0.0,
            D1=0.0, A1=0.0,
            D2=0.0, A2=0.0,
            D3=0.0, A3=0.0,
            # forward_weight0=0.0, source_weight0=Optim(0.999, 0.0, 1.0),
            # forward_weight1=0.0, source_weight1=Optim(0.999, 0.0, 1.0),
            # forward_weight2=0.0, source_weight2=Optim(0.999, 0.0, 1.0),
            # forward_weight3=0.0, source_weight3=Optim(0.999, 0.0, 1.0),
            forward_weight0=0.0, source_weight0=1,
            forward_weight1=0.0, source_weight1=1,
            forward_weight2=0.0, source_weight2=1,
            forward_weight3=0.0, source_weight3=1,
        ),
        # "PCAAfterSourceModel_LM_TWHW":
        #     ProjectionAfterSourceModel(
        #         name="", loss=mse, optim_method=BAYES,
        #         verbose=True, niter=11, sigma0=1,
        #         source_model=NodeSourceModel(
        #             path4preprocess=data_manager.path, graph=graph,
        #             spacial_locations=station_coordinates,
        #             times=times_all,
        #             traffic_by_edge=traffic_by_edge,
        #             redo_preprocessing=False,
        #             name="", loss=mse, optim_method=GRAD,
        #             verbose=True, niter=10, sigma0=1,
        #             lnei=1,
        #             source_model=LassoCV(selection="random", positive=False),
        #             substract_mean=True,
        #             extra_regressors=["temperature", "wind", "hours", "week"],
        #         ),
        #         basis="pca",
        #         # k=Optim(10, 1, 10)
        #         k=10
        #     ),
        # "LapAfterSourceModel_LM_TWHW":
        #     ProjectionAfterSourceModel(
        #         name="", loss=mse, optim_method=BAYES,
        #         verbose=True, niter=11, sigma0=1,
        #         source_model=NodeSourceModel(
        #             path4preprocess=data_manager.path, graph=graph,
        #             spacial_locations=station_coordinates,
        #             times=times_all,
        #             traffic_by_edge=traffic_by_edge,
        #             redo_preprocessing=False,
        #             name="", loss=mse, optim_method=GRAD,
        #             verbose=True, niter=10, sigma0=1,
        #             lnei=1,
        #             source_model=LassoCV(selection="random", positive=False),
        #             substract_mean=True,
        #             extra_regressors=["temperature", "wind", "hours", "week"],
        #         ),
        #         basis="graph_laplacian",
        #         k=Optim(10, 1, 10)
        #     ),
        # # # "SourceModel_Poly1NN_avg_TWHW":
        # # #     NodeSourceModel(path4preprocess=data_manager.path, graph=graph,
        # # #                     spacial_locations=station_coordinates, times=times_all,
        # # #                     traffic_by_edge=traffic_by_edge,
        # # #                     redo_preprocessing=False,
        # # #                     name="", loss=mse, optim_method=GRAD,
        # # #                     verbose=True, niter=10, sigma0=1,
        # # #                     lnei=1,
        # # #                     source_model=MLPRegressor(hidden_layer_sizes=hidden_layer_sizes,
        # # #                                               activation=activation,  # 'relu',
        # # #                                               learning_rate_init=learning_rate_init,
        # # #                                               learning_rate=learning_rate,
        # # #                                               early_stopping=early_stopping,
        # # #                                               solver=solver.lower(),
        # # #                                               max_iter=max_iter),
        # # #                     substract_mean=True,
        # # #                     extra_regressors=["temperature", "wind", "hours", "week"],
        # # #                     ),
        # #
        # # # "PCASourceModel_Poly1Lasso_avg_TW":
        # # #     PCASourceModel(path4preprocess=data_manager.path, graph=graph,
        # # #                    spacial_locations=station_coordinates, times=times_all,
        # # #                    traffic_by_edge=traffic_by_edge, mean_normalize=True, std_normalize=False,
        # # #                    redo_preprocessing=False,
        # # #                    name="", loss=mse, optim_method=GRAD,
        # # #                    verbose=True, niter=10, sigma0=1,
        # # #                    lnei=1, k_max=10,  # k=5,
        # # #                    source_model=LassoCV(selection="random", positive=False),
        # # #                    substract_mean=True,
        # # #                    extra_regressors=["temperature", "wind"],
        # # #                    ),
        # #
        # # "PCASourceModel_Poly1Lasso_avg_TWHW":
        # #     PCASourceModel(path4preprocess=data_manager.path, graph=graph,
        # #                    spacial_locations=station_coordinates, times=times_all,
        # #                    traffic_by_edge=traffic_by_edge, mean_normalize=True, std_normalize=False,
        # #                    redo_preprocessing=False,
        # #                    name="", loss=mse, optim_method=GRAD,
        # #                    verbose=True, niter=10, sigma0=1,
        # #                    lnei=1, k_max=10, k=5,
        # #                    source_model=LassoCV(selection="random", positive=False),
        # #                    substract_mean=True,
        # #                    extra_regressors=["temperature", "wind", "hours", "week"],
        # #                    ),
        # # "PhysicsModel":
        # #     PhysicsModel(path4preprocess=data_manager.path, graph=graph,
        # #                  spacial_locations=station_coordinates, times=times_all,
        # #                  traffic_by_edge=traffic_by_edge, mean_normalize=True, std_normalize=False,
        # #                  redo_preprocessing=False,
        # #                  name="",
        # #                  verbose=True,
        # #                  niter=50, sigma0=1,
        # #                  lnei=1, k_max=10, k=5,
        # #                  rb_k_max=10,
        # #                  source_model=LassoCV(selection="random", positive=False),
        # #                  substract_mean=True,
        # #                  extra_regressors=["temperature", "wind", "hours", "week"],
        # #                  loss=mse, optim_method=GRAD,
        # #                  cv_in_space=True,
        # #                  # rb_k=Optim(start=9, lower=1, upper=10),
        # #                  rb_k=9,
        # #                  # basis="graph_laplacian",
        # #                  basis="pca_source",
        # #                  absorption=0.1,
        # #                  diffusion=0.1,
        # #                  # absorption=Optim(start=0.1, lower=1e-4, upper=1e3),
        # #                  # diffusion=Optim(start=7137, lower=1e-4, upper=1e5),
        # #                  # alpha=Optim(start=0.0, lower=0.0, upper=1.0),
        # #                  # delta=Optim(start=0.013, lower=0.0, upper=1.0),
        # #                  alpha=0,
        # #                  delta=0.013,
        # #                  ),
        # # "PCASourceModel_Poly1NN_avg_TWHW":
        # #     PCASourceModel(path4preprocess=data_manager.path, graph=graph,
        # #                    spacial_locations=station_coordinates, times=times_all,
        # #                    traffic_by_edge=traffic_by_edge, mean_normalize=True, std_normalize=False,
        # #                    redo_preprocessing=False,
        # #                    name="", loss=mse, optim_method=GRAD,
        # #                    verbose=True, niter=10, sigma0=1,
        # #                    lnei=1, k_max=10, k=5,
        # #                    source_model=MLPRegressor(hidden_layer_sizes=hidden_layer_sizes,
        # #                                              activation=activation,  # 'relu',
        # #                                              learning_rate_init=learning_rate_init,
        # #                                              learning_rate=learning_rate,
        # #                                              early_stopping=early_stopping,
        # #                                              solver=solver.lower(),
        # #                                              max_iter=max_iter),
        # #                    # source_model=BayesSearchCV(
        # #                    #     RandomForestRegressor(n_estimators=25),
        # #                    #     {
        # #                    #         "max_depth": Integer(1, 5),
        # #                    #         "min_samples_split": Integer(2, 10),
        # #                    #         "min_samples_leaf": Integer(1, 10),
        # #                    #         # "min_weight_fraction_leaf" = 0.0,
        # #                    #         "max_features": Integer(1, 10),
        # #                    #         # "max_leaf_nodes" = None,
        # #                    #         # "min_impurity_decrease" = 0.0,
        # #                    #         # ccp_alpha = 0.0,
        # #                    #         # max_samples = None,
        # #                    #     },
        # #                    #     cv=5,
        # #                    #     n_iter=32,
        # #                    #     random_state=0
        # #                    # ),
        # #                    substract_mean=True,
        # #                    extra_regressors=["temperature", "wind", "hours", "week"],
        # #                    ),
        # # "PCASourceModel_Poly1RF_avg_TWHW":
        # #     PCASourceModel(path4preprocess=data_manager.path, graph=graph,
        # #                    spacial_locations=station_coordinates, times=times_all,
        # #                    traffic_by_edge=traffic_by_edge, mean_normalize=True, std_normalize=False,
        # #                    redo_preprocessing=False,
        # #                    name="", loss=mse, optim_method=GRAD,
        # #                    verbose=True, niter=10, sigma0=1,
        # #                    lnei=1, k_max=10, k=5,
        # #                    source_model=RandomForestRegressor(n_estimators=25, max_depth=3),
        # #                    # source_model=BayesSearchCV(
        # #                    #     RandomForestRegressor(n_estimators=25),
        # #                    #     {
        # #                    #         "max_depth": Integer(1, 5),
        # #                    #         "min_samples_split": Integer(2, 10),
        # #                    #         "min_samples_leaf": Integer(1, 10),
        # #                    #         # "min_weight_fraction_leaf" = 0.0,
        # #                    #         "max_features": Integer(1, 10),
        # #                    #         # "max_leaf_nodes" = None,
        # #                    #         # "min_impurity_decrease" = 0.0,
        # #                    #         # ccp_alpha = 0.0,
        # #                    #         # max_samples = None,
        # #                    #     },
        # #                    #     cv=5,
        # #                    #     n_iter=32,
        # #                    #     random_state=0
        # #                    # ),
        # #                    substract_mean=True,
        # #                    extra_regressors=["temperature", "wind", "hours", "week"],
        # #                    ),
        # # "LaplacianSourceModel_Poly1Lasso_avg_TWHW":
        # #     LaplacianSourceModel(path4preprocess=data_manager.path, graph=graph,
        # #                          spacial_locations=station_coordinates, times=times_all,
        # #                          traffic_by_edge=traffic_by_edge, mean_normalize=True, std_normalize=False,
        # #                          redo_preprocessing=False,
        # #                          name="", loss=mse, optim_method=GRAD,
        # #                          verbose=True, niter=10, sigma0=1,
        # #                          lnei=1, k_max=10, k=5,
        # #                          source_model=LassoCV(selection="random", positive=False),
        # #                          substract_mean=True,
        # #                          extra_regressors=["temperature", "wind", "hours", "week"],
        # #                          ),
        # "V2LaplacianSourceModel_Poly1Lasso_avg_TWHW":
        #     LaplacianSourceModel(path4preprocess=data_manager.path, graph=graph,
        #                          spacial_locations=station_coordinates, times=times_all,
        #                          traffic_by_edge=traffic_by_edge, mean_normalize=True, std_normalize=False,
        #                          redo_preprocessing=False,
        #                          name="", loss=mse, optim_method=NONE_OPTIM_METHOD,
        #                          verbose=True, niter=10, sigma0=1,
        #                          lnei=1, k_max=10, k=None,
        #                          source_model=LassoCV(selection="random", positive=False),
        #                          substract_mean=True, cv_in_space=False,
        #                          extra_regressors=["temperature", "wind", "hours", "week"],
        #                          ),
        # "V2LaplacianSourceModel_NN_avg_TWHW":
        #     LaplacianSourceModel(path4preprocess=data_manager.path, graph=graph,
        #                          spacial_locations=station_coordinates, times=times_all,
        #                          traffic_by_edge=traffic_by_edge, mean_normalize=True, std_normalize=False,
        #                          redo_preprocessing=False,
        #                          name="", loss=mse, optim_method=NONE_OPTIM_METHOD,
        #                          verbose=True, niter=10, sigma0=1,
        #                          lnei=1, k_max=10, k=None,
        #                          source_model=MLPRegressor(hidden_layer_sizes=hidden_layer_sizes,
        #                                                    activation=activation,  # 'relu',
        #                                                    learning_rate_init=learning_rate_init,
        #                                                    learning_rate=learning_rate,
        #                                                    early_stopping=early_stopping,
        #                                                    solver=solver.lower(),
        #                                                    max_iter=max_iter),
        #                          substract_mean=True, cv_in_space=False,
        #                          extra_regressors=["temperature", "wind", "hours", "week"],
        #                          ),
        # "V2LaplacianSourceModel_RF_avg_TWHW":
        #     LaplacianSourceModel(path4preprocess=data_manager.path, graph=graph,
        #                          spacial_locations=station_coordinates, times=times_all,
        #                          traffic_by_edge=traffic_by_edge, mean_normalize=True, std_normalize=False,
        #                          redo_preprocessing=False,
        #                          name="", loss=mse, optim_method=NONE_OPTIM_METHOD,
        #                          verbose=True, niter=10, sigma0=1,
        #                          lnei=1, k_max=10, k=None,
        #                          source_model=RandomForestRegressor(n_estimators=25, max_depth=3),
        #                          substract_mean=True, cv_in_space=False,
        #                          extra_regressors=["temperature", "wind", "hours", "week"],
        #                          ),

    }

    models2 = dict()

    # models2["Ensemble"] = ModelsAggregator(models=[
    #     models["PCASourceModel_Poly1Lasso_avg_TWHW"],
    #     models["LaplacianSourceModel_Poly1Lasso_avg_TWHW"],
    #     models["PCASourceModel_Poly1RF_avg_TWHW"],
    #     models["PCASourceModel_Poly1NN_avg_TWHW"],
    #     models["SourceModel_Poly1Lasso_avg_TWHW"],
    #     models["PhysicsModel"],
    #     models["Spatial Avg"]],
    #     aggregator=Pipeline([("lasso", LassoCV(selection="random", positive=False))]))
    #
    # models2["Ensemble2"] = ModelsAggregator(models=[
    #     models["PCASourceModel_Poly1Lasso_avg_TWHW"],
    #     # models["LaplacianSourceModel_Poly1Lasso_avg_TWHW"],
    #     # models["PCASourceModel_Poly1RF_avg_TWHW"],
    #     # models["PCASourceModel_Poly1NN_avg_TWHW"],
    #     models["SourceModel_Poly1Lasso_avg_TWHW"],
    #     models["PhysicsModel"],
    #     models["Spatial Avg"]],
    #     aggregator=Pipeline([("lasso", LassoCV(selection="random", positive=False))]))

    # models2["EnsembleAvg"] = ModelsAggregator(models=[
    #     models["PCASourceModel_Poly1Lasso_avg_TWHW"],
    #     models["LaplacianSourceModel_Poly1Lasso_avg_TWHW"],
    #     models["PCASourceModel_Poly1RF_avg_TWHW"],
    #     models["PCASourceModel_Poly1NN_avg_TWHW"],
    #     # models["SourceModel_Poly1Lasso_avg_TWHW"],
    #     # models["PhysicsModel"],
    #     # models["Kernel"],
    #     models["Spatial Avg"]],
    #     fitting_strategy="average",
    #     aggregator=Pipeline([("lasso", LassoCV(selection="random", positive=False))]))

    # models2["EnsembleKernelBLUE"] = ModelsAggregator(models=[
    #     models["PCASourceModel_Poly1Lasso_avg_TWHW"],
    #     models["LaplacianSourceModel_Poly1Lasso_avg_TWHW"],
    #     models["PCASourceModel_Poly1RF_avg_TWHW"],
    #     models["PCASourceModel_Poly1NN_avg_TWHW"],
    #     # models["SourceModel_Poly1Lasso_avg_TWHW"],
    #     # models["PhysicsModel"],
    #     models["Kernel"],
    #     models["Spatial Avg"]],
    #     fitting_strategy="BLUE_weighting",
    #     aggregator=Pipeline([("lasso", LassoCV(selection="random", positive=False))]))
    # models2["EnsembleBLUE"] = ModelsAggregator(models=[
    #     models["PCASourceModel_Poly1Lasso_avg_TWHW"],
    #     models["LaplacianSourceModel_Poly1Lasso_avg_TWHW"],
    #     models["PCASourceModel_Poly1RF_avg_TWHW"],
    #     models["PCASourceModel_Poly1NN_avg_TWHW"],
    #     # models["SourceModel_Poly1Lasso_avg_TWHW"],
    #     # models["PhysicsModel"],
    #     # models["Kernel"],
    #     models["Spatial Avg"]],
    #     fitting_strategy="BLUE_weighting",
    #     aggregator=Pipeline([("lasso", LassoCV(selection="random", positive=False))]))
    #
    # models2["EnsembleKernelMSE"] = ModelsAggregator(models=[
    #     models["PCASourceModel_Poly1Lasso_avg_TWHW"],
    #     models["LaplacianSourceModel_Poly1Lasso_avg_TWHW"],
    #     models["PCASourceModel_Poly1RF_avg_TWHW"],
    #     models["PCASourceModel_Poly1NN_avg_TWHW"],
    #     # models["SourceModel_Poly1Lasso_avg_TWHW"],
    #     # models["PhysicsModel"],
    #     models["Kernel"],
    #     models["Spatial Avg"]],
    #     fitting_strategy="mse_weighting",
    #     aggregator=Pipeline([("lasso", LassoCV(selection="random", positive=False))]))
    # models2["SoftDiffusion"] = SoftDiffusion(path4preprocess=data_manager.path, graph=graph,
    #                                          spacial_locations=station_coordinates, times=times_all,
    #                                          traffic_by_edge=traffic_by_edge,
    #                                          redo_preprocessing=False, cv_in_space=False,
    #                                          name="", loss=mse, optim_method=GRAD,
    #                                          verbose=True, niter=50, sigma0=1,
    #                                          source_model=models["PCASourceModel_Poly1Lasso_avg_TWHW"],
    #                                          substract_mean=True,
    #                                          alpha=Optim(1.0, None, None),
    #                                          beta=Optim(0.0, None, None),
    #                                          delta=Optim(0.0, None, None),
    #                                          )
    # models2["PCASourceModel_Poly1NN_avg_TWHW"] = models["PCASourceModel_Poly1NN_avg_TWHW"]
    # models2["Kernel"] = models["Kernel"]
    # models2["BLUE"] = models["BLUE"]
    # models2["BLUE_DU"] = models["BLUE_DU"]
    # models2["BLUE_DI"] = models["BLUE_DI"]
    # models2["ExponentialD"] = models["ExponentialD"]
    # models2["Exponential"] = models["Exponential"]
    # models2["ExponentialFit"] = models["ExponentialFit"]
    # models2["Gaussian"] = models["Gaussian"]
    # models2["ExponentialOld"] = models["ExponentialOld"]
    # models2["V2LaplacianSourceModel_Poly1Lasso_avg_TWHW"] = models["V2LaplacianSourceModel_Poly1Lasso_avg_TWHW"]
    # models2["V2LaplacianSourceModel_NN_avg_TWHW"] = models["V2LaplacianSourceModel_NN_avg_TWHW"]
    # models2["V2LaplacianSourceModel_RF_avg_TWHW"] = models["V2LaplacianSourceModel_RF_avg_TWHW"]

    # models2["SourceModel_Poly1Lasso_avg_TWHW"] = models["SourceModel_Poly1Lasso_avg_TWHW"]
    # models2["SourceModel_NN_avg_TWHW"] = models["SourceModel_NN_avg_TWHW"]
    # models2["SourceModel_RF_avg_TWHW"] = models["SourceModel_RF_avg_TWHW"]

    # models2["PCAAfterSourceModel_LM_TWHW"] = models["PCAAfterSourceModel_LM_TWHW"]
    # models2["LapAfterSourceModel_LM_TWHW"] = models["LapAfterSourceModel_LM_TWHW"]

    # models2["DistanceRational"] = DistanceModel(
    #     kernel_function=rational_kernel,
    #     alpha=1.0, beta=Optim(0.01, 0.01, 10),
    #     name="", loss=mse, optim_method=GRAD, niter=50, verbose=True)
    for source_model_name, source_model in [("linear", LassoCV(selection="random", positive=False)),
                                            # ("nn", MLPRegressor(hidden_layer_sizes=hidden_layer_sizes,
                                            #                     activation=activation,  # 'relu',
                                            #                     learning_rate_init=learning_rate_init,
                                            #                     learning_rate=learning_rate,
                                            #                     early_stopping=early_stopping,
                                            #                     solver=solver.lower(),
                                            #                     max_iter=max_iter)),
                                            ("RF", RandomForestRegressor(n_estimators=25, max_depth=3))
                                            ]:
        models2[f"node_{source_model_name}_TWHW"] = NodeSourceModel(
            train_with_relative_error=train_with_relative_error,
            path4preprocess=data_manager.path, graph=graph,
            spacial_locations=station_coordinates,
            times=times_all,
            traffic_by_edge=traffic_by_edge,
            redo_preprocessing=False,
            name="", loss=mse, optim_method=NONE_OPTIM_METHOD,
            verbose=True, niter=1, sigma0=1,
            lnei=1,
            source_model=source_model,
            substract_mean=True,
            extra_regressors=["temperature", "wind", "hours", "week"],
        )
        for basis in ["geometrical", "pca"]:
            models2[f"{basis}_{source_model_name}_TWHW"] = ProjectionFullSourceModel(
                train_with_relative_error=train_with_relative_error,
                path4preprocess=data_manager.path, graph=graph,
                spacial_locations=station_coordinates,
                times=times_all,
                traffic_by_edge=traffic_by_edge,
                redo_preprocessing=False,
                name="", loss=mse, optim_method=NONE_OPTIM_METHOD,
                verbose=True, niter=25, sigma0=1,
                lnei=1, k_max=10,
                source_model=source_model,
                substract_mean=True, cv_in_space=False,
                extra_regressors=["temperature", "wind", "hours", "week"],
                basis=basis,
                # kv=Optim(5, None, None), ky=Optim(5, None, None), kr=Optim(5, None, None), kd=Optim(5, None, None),
                kv=5, ky=5, kr=5, kd=5,
                # kv=10, ky=3, kr=3, kd=1,
                # D0=Optim(1e4, 1e-4, 1e4), A0=Optim(1e-4, 1e-4, 1e4),
                D0=0.0, A0=0.0,
                D1=0.0, A1=0.0,
                D2=0.0, A2=0.0,
                D3=0.0, A3=0.0,
                # forward_weight0=0.0, source_weight0=Optim(0.999, 0.0, 1.0),
                # forward_weight1=0.0, source_weight1=Optim(0.999, 0.0, 1.0),
                # forward_weight2=0.0, source_weight2=Optim(0.999, 0.0, 1.0),
                # forward_weight3=0.0, source_weight3=Optim(0.999, 0.0, 1.0),
                forward_weight0=0.0, source_weight0=1,
                forward_weight1=0.0, source_weight1=1,
                forward_weight2=0.0, source_weight2=1,
                forward_weight3=0.0, source_weight3=1,
            )
    models2 = {
        "EnsembleAvg": ModelsAggregator(
            models=[models2["geometrical_RF_TWHW"], models2["node_linear_TWHW"], models2["pca_linear_TWHW"]],
            weighting="average",
            train_on_llo=True,
            aggregator=Pipeline([("lasso", LassoCV(selection="random", positive=False))])
        )
    }

    # models2 = {
    #     # "EnsembleAvgNoCV_average": ModelsAggregatorNoCV(
    #     #     models=list(models2.values()),
    #     #     aggregator="average",
    #     #     extra_regressors=[]
    #     # ),
    #     # "EnsembleAvgNoCV_std": ModelsAggregatorNoCV(
    #     #     models=list(models2.values()),
    #     #     aggregator="std",
    #     #     extra_regressors=[]
    #     # ),
    #     # "EnsembleAvgNoCV_cv": ModelsAggregatorNoCV(
    #     #     models=list(models2.values()),
    #     #     aggregator="cv",
    #     #     extra_regressors=[]
    #     # ),
    #     # "EnsembleAvgNoCV_weighted_average": ModelsAggregatorNoCV(
    #     #     models=list(models2.values()),
    #     #     aggregator="weighted_average",
    #     #     extra_regressors=[]
    #     # ),
    # }


    # models2 = {
    #     "EnsembleAvgNoCV_cv": ModelsAggregatorNoCV(
    #         models=[models2["geometrical_RF_TWHW"], models2["node_linear_TWHW"], models2["pca_linear_TWHW"],
    #                 # models["ExponentialFit"], models["Spatial Avg"]
    #                 ],
    #         aggregator="cv",
    #         extra_regressors=[]
    #     )
    # }

    # models2 = {
    #     "EnsembleAvgNoCV_Lasso": ModelsAggregatorNoCV(
    #         models=[models2["geometrical_RF_TWHW"], models2["node_linear_TWHW"], models2["pca_linear_TWHW"],
    #                 # models["ExponentialFit"], models["Spatial Avg"]
    #                 ],
    #         aggregator=Pipeline([("lasso", LassoCV(selection="cyclic", positive=True))]),
    #         extra_regressors=[]
    #     )
    # }

    # models2 = {
    #     "EnsembleAvgNoCV_RF": ModelsAggregatorNoCV(
    #         models=[models2["geometrical_RF_TWHW"], models2["pca_RF_TWHW"], models2["pca_linear_TWHW"]],
    #         aggregator=RandomForestRegressor(n_estimators=25, max_depth=3),
    #         extra_regressors=["temperature", "wind", "hours", "week"]
    #     )
    # }
    # models2 = {
    #     "EnsembleAvgNoCV_nn": ModelsAggregatorNoCV(
    #         models=[models2["geometrical_RF_TWHW"], models2["pca_RF_TWHW"], models2["pca_linear_TWHW"]],
    #         aggregator=MLPRegressor(hidden_layer_sizes=hidden_layer_sizes,
    #                                 activation=activation,  # 'relu',
    #                                 learning_rate_init=learning_rate_init,
    #                                 learning_rate=learning_rate,
    #                                 early_stopping=early_stopping,
    #                                 solver=solver.lower(),
    #                                 max_iter=max_iter),
    #         extra_regressors=["temperature", "wind", "hours", "week"]
    #     )
    # }

    # models2["BLUE"] = models["BLUE"]
    # models2["Spatial Avg"] = models["Spatial Avg"]
    lab = LabPipeline()
    lab.define_new_block_of_functions(
        "individual_models",
        *list(map(train_test_model, models2.items())),
        recalculate=True
    )

    lab.execute(
        data_manager,
        num_cores=15,
        forget=False,
        save_on_iteration=None,
        station=stations2test
    )

    # lab.define_new_block_of_functions(
    #     "individual_models",
    #     # train_test_model(("SL Spatial Avg", (Pipeline([("LR", LassoCV(selection="random"))]), ["Spatial Avg"]))),
    #     train_test_model(("SL Physics", (Pipeline([("LR", LassoCV(selection="random"))]), ["Spatial Avg", "Physics"]))),
    #     train_test_model(
    #         ("SL Physics-avg", (Pipeline([("LR", LassoCV(selection="random"))]), ["Spatial Avg", "Physics-avg"]))),
    #     recalculate=True
    # )
    # lab.execute(
    #     data_manager,
    #     num_cores=15,
    #     forget=False,
    #     save_on_iteration=None,
    #     # station=["BONAP"],  # stations2test
    #     station=stations2test
    # )

    import DoPlotsSourceModels
