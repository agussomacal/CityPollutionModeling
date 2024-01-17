import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline

import src.config as config
from PerplexityLab.DataManager import DataManager
from PerplexityLab.LabPipeline import LabPipeline
from PerplexityLab.miscellaneous import copy_main_script_version
from src.experiments.paper_experiments.PreProcessPaper import train_test_model, stations2test, station_coordinates, \
    graph, times_all, traffic_by_edge
from src.experiments.paper_experiments.params4runs import runsinfo
from src.lib.Models.BaseModel import GRAD, mse, BAYES, NONE_OPTIM_METHOD
from src.lib.Models.SensorDependentModels.BLUEFamily import BLUEModel
from src.lib.Models.TrueStateEstimationModels.AverageModels import SnapshotMeanModel
from src.lib.Models.TrueStateEstimationModels.KernelModels import ExponentialKernelModel
from src.lib.Models.TrueStateEstimationModels.PhysicsModel import PCASourceModel, LaplacianSourceModel, PhysicsModel, \
    ModelsAggregator, NodeSourceModel
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

    data_manager = DataManager(
        path=config.paper_experiments_dir,
        emissions_path=config.results_dir,
        name="SourceModels",
        country_alpha_code="FR",
        trackCO2=False
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
        "SourceModel_Poly1Lasso_avg":
            NodeSourceModel(path4preprocess=data_manager.path, graph=graph,
                            spacial_locations=station_coordinates, times=times_all,
                            traffic_by_edge=traffic_by_edge,
                            redo_preprocessing=False,
                            name="", loss=mse, optim_method=GRAD,
                            verbose=True, niter=10, sigma0=1,
                            lnei=1,
                            source_model=LassoCV(selection="random", positive=False),
                            substract_mean=True,
                            # extra_regressors=["temperature", "wind"],
                            ),
        "SourceModel_Poly1Lasso_avg_TWHW":
            NodeSourceModel(path4preprocess=data_manager.path, graph=graph,
                            spacial_locations=station_coordinates, times=times_all,
                            traffic_by_edge=traffic_by_edge,
                            redo_preprocessing=False,
                            name="", loss=mse, optim_method=GRAD,
                            verbose=True, niter=10, sigma0=1,
                            lnei=1,
                            source_model=LassoCV(selection="random", positive=False),
                            substract_mean=True,
                            extra_regressors=["temperature", "wind"],
                            ),
        # "SourceModel_Poly1NN_avg_TWHW":
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

        # "PCASourceModel_Poly1Lasso_avg_TW":
        #     PCASourceModel(path4preprocess=data_manager.path, graph=graph,
        #                    spacial_locations=station_coordinates, times=times_all,
        #                    traffic_by_edge=traffic_by_edge, mean_normalize=True, std_normalize=False,
        #                    redo_preprocessing=False,
        #                    name="", loss=mse, optim_method=GRAD,
        #                    verbose=True, niter=10, sigma0=1,
        #                    lnei=1, k_max=10,  # k=5,
        #                    source_model=LassoCV(selection="random", positive=False),
        #                    substract_mean=True,
        #                    extra_regressors=["temperature", "wind"],
        #                    ),

        "PCASourceModel_Poly1Lasso_avg_TWHW":
            PCASourceModel(path4preprocess=data_manager.path, graph=graph,
                           spacial_locations=station_coordinates, times=times_all,
                           traffic_by_edge=traffic_by_edge, mean_normalize=True, std_normalize=False,
                           redo_preprocessing=False,
                           name="", loss=mse, optim_method=GRAD,
                           verbose=True, niter=10, sigma0=1,
                           lnei=1, k_max=10,  k=5,
                           source_model=LassoCV(selection="random", positive=False),
                           substract_mean=True,
                           extra_regressors=["temperature", "wind", "hours", "week"],
                           ),
        "PhysicsModel":
            PhysicsModel(path4preprocess=data_manager.path, graph=graph,
                         spacial_locations=station_coordinates, times=times_all,
                         traffic_by_edge=traffic_by_edge, mean_normalize=True, std_normalize=False,
                         redo_preprocessing=False,
                         name="",
                         verbose=True,
                         niter=50, sigma0=1,
                         lnei=1, k_max=10, k=5,
                         rb_k_max=10,
                         source_model=LassoCV(selection="random", positive=False),
                         substract_mean=True,
                         extra_regressors=["temperature", "wind", "hours", "week"],
                         loss=mse, optim_method=GRAD,
                         cv_in_space=True,
                         # rb_k=Optim(start=9, lower=1, upper=10),
                         rb_k=9,
                         # basis="graph_laplacian",
                         basis="pca_source",
                         absorption=0.1,
                         diffusion=0.1,
                         # absorption=Optim(start=0.1, lower=1e-4, upper=1e3),
                         # diffusion=Optim(start=7137, lower=1e-4, upper=1e5),
                         # alpha=Optim(start=0.0, lower=0.0, upper=1.0),
                         # delta=Optim(start=0.013, lower=0.0, upper=1.0),
                         alpha=0,
                         delta=0.013,
                         ),
        # "PCASourceModel_Poly1Lasso_avg_TWHW_nei+":
        #     PCASourceModel(path4preprocess=data_manager.path, graph=graph,
        #                    spacial_locations=station_coordinates, times=times_all,
        #                    traffic_by_edge=traffic_by_edge, mean_normalize=True, std_normalize=False,
        #                    redo_preprocessing=False,
        #                    name="", loss=mse, optim_method=GRAD,
        #                    verbose=True, niter=10, sigma0=1,
        #                    lnei=2, k_max=10, k=2,
        #                    source_model=LassoCV(selection="random", positive=False),
        #                    substract_mean=True,
        #                    extra_regressors=["temperature", "wind", "hours", "week"],
        #                    ),
        "PCASourceModel_Poly1NN_avg_TWHW":
            PCASourceModel(path4preprocess=data_manager.path, graph=graph,
                           spacial_locations=station_coordinates, times=times_all,
                           traffic_by_edge=traffic_by_edge, mean_normalize=True, std_normalize=False,
                           redo_preprocessing=False,
                           name="", loss=mse, optim_method=GRAD,
                           verbose=True, niter=10, sigma0=1,
                           lnei=1, k_max=10, k=5,
                           source_model=MLPRegressor(hidden_layer_sizes=hidden_layer_sizes,
                                                     activation=activation,  # 'relu',
                                                     learning_rate_init=learning_rate_init,
                                                     learning_rate=learning_rate,
                                                     early_stopping=early_stopping,
                                                     solver=solver.lower(),
                                                     max_iter=max_iter),
                           # source_model=BayesSearchCV(
                           #     RandomForestRegressor(n_estimators=25),
                           #     {
                           #         "max_depth": Integer(1, 5),
                           #         "min_samples_split": Integer(2, 10),
                           #         "min_samples_leaf": Integer(1, 10),
                           #         # "min_weight_fraction_leaf" = 0.0,
                           #         "max_features": Integer(1, 10),
                           #         # "max_leaf_nodes" = None,
                           #         # "min_impurity_decrease" = 0.0,
                           #         # ccp_alpha = 0.0,
                           #         # max_samples = None,
                           #     },
                           #     cv=5,
                           #     n_iter=32,
                           #     random_state=0
                           # ),
                           substract_mean=True,
                           extra_regressors=["temperature", "wind", "hours", "week"],
                           ),
        "PCASourceModel_Poly1RF_avg_TWHW":
            PCASourceModel(path4preprocess=data_manager.path, graph=graph,
                           spacial_locations=station_coordinates, times=times_all,
                           traffic_by_edge=traffic_by_edge, mean_normalize=True, std_normalize=False,
                           redo_preprocessing=False,
                           name="", loss=mse, optim_method=GRAD,
                           verbose=True, niter=10, sigma0=1,
                           lnei=1, k_max=10, k=5,
                           source_model=RandomForestRegressor(n_estimators=25, max_depth=3),
                           # source_model=BayesSearchCV(
                           #     RandomForestRegressor(n_estimators=25),
                           #     {
                           #         "max_depth": Integer(1, 5),
                           #         "min_samples_split": Integer(2, 10),
                           #         "min_samples_leaf": Integer(1, 10),
                           #         # "min_weight_fraction_leaf" = 0.0,
                           #         "max_features": Integer(1, 10),
                           #         # "max_leaf_nodes" = None,
                           #         # "min_impurity_decrease" = 0.0,
                           #         # ccp_alpha = 0.0,
                           #         # max_samples = None,
                           #     },
                           #     cv=5,
                           #     n_iter=32,
                           #     random_state=0
                           # ),
                           substract_mean=True,
                           extra_regressors=["temperature", "wind", "hours", "week"],
                           ),
        "LaplacianSourceModel_Poly1Lasso_avg_TWHW":
            LaplacianSourceModel(path4preprocess=data_manager.path, graph=graph,
                                 spacial_locations=station_coordinates, times=times_all,
                                 traffic_by_edge=traffic_by_edge, mean_normalize=True, std_normalize=False,
                                 redo_preprocessing=False,
                                 name="", loss=mse, optim_method=GRAD,
                                 verbose=True, niter=10, sigma0=1,
                                 lnei=1, k_max=10,  k=5,
                                 source_model=LassoCV(selection="random", positive=False),
                                 substract_mean=True,
                                 extra_regressors=["temperature", "wind", "hours", "week"],
                                 ),

    }

    models2 = dict()

    models2["Ensemble"] = ModelsAggregator(models=[
        models["PCASourceModel_Poly1Lasso_avg_TWHW"],
        models["LaplacianSourceModel_Poly1Lasso_avg_TWHW"],
        models["PCASourceModel_Poly1RF_avg_TWHW"],
        models["PCASourceModel_Poly1NN_avg_TWHW"],
        models["SourceModel_Poly1Lasso_avg_TWHW"],
        models["PhysicsModel"],
        models["Spatial Avg"]],
        aggregator=Pipeline([("lasso", LassoCV(selection="random", positive=False))]))


    models2["Ensemble2"] = ModelsAggregator(models=[
        models["PCASourceModel_Poly1Lasso_avg_TWHW"],
        # models["LaplacianSourceModel_Poly1Lasso_avg_TWHW"],
        # models["PCASourceModel_Poly1RF_avg_TWHW"],
        # models["PCASourceModel_Poly1NN_avg_TWHW"],
        models["SourceModel_Poly1Lasso_avg_TWHW"],
        models["PhysicsModel"],
        models["Spatial Avg"]],
        aggregator=Pipeline([("lasso", LassoCV(selection="random", positive=False))]))

    models2["Ensemble"] = ModelsAggregator(models=[
        models["PCASourceModel_Poly1Lasso_avg_TWHW"],
        models["LaplacianSourceModel_Poly1Lasso_avg_TWHW"],
        models["PCASourceModel_Poly1RF_avg_TWHW"],
        models["PCASourceModel_Poly1NN_avg_TWHW"],
        models["SourceModel_Poly1Lasso_avg_TWHW"],
        models["PhysicsModel"],
        models["Spatial Avg"]],
        aggregator=Pipeline([("lasso", LassoCV(selection="random", positive=False))]))

    models2["EnsembleKernel"] = ModelsAggregator(models=[
        models["PCASourceModel_Poly1Lasso_avg_TWHW"],
        models["LaplacianSourceModel_Poly1Lasso_avg_TWHW"],
        models["PCASourceModel_Poly1RF_avg_TWHW"],
        models["PCASourceModel_Poly1NN_avg_TWHW"],
        models["SourceModel_Poly1Lasso_avg_TWHW"],
        models["PhysicsModel"],
        models["Kernel"],
        models["Spatial Avg"]],
        aggregator=Pipeline([("lasso", LassoCV(selection="random", positive=False))]))

    models2["PCASourceModel_Poly1NN_avg_TWHW"] = models["PCASourceModel_Poly1NN_avg_TWHW"]

    lab = LabPipeline()
    lab.define_new_block_of_functions(
        "individual_models",
        *list(map(train_test_model, models2.items())),
        recalculate=False
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
