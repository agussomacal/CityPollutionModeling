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
from src.lib.Models.TrueStateEstimationModels.KernelModels import ExponentialKernelModel, GaussianKernelModel
from src.lib.Models.TrueStateEstimationModels.PhysicsModel import NodeSourceModel, ProjectionFullSourceModel, \
    ModelsAggregatorNoCV, ModelsAggregator
from src.lib.Modules import Optim

k = 5
sources_dist = [0.005]
if sources_dist == [0.005]:
    source_dist4name = "only005"
elif sources_dist == [0]:
    source_dist4name = ""
elif sources_dist == [0, 0.005]:
    source_dist4name = "005"
else:
    raise Exception("Invalid source dist")
# source_dist4name = ("_" if len(sources_dist) > 0 else "") + "_".join(list(map(str, sources_dist))).replace("0.", "")
# "temperature", "wind", "hours", "week", , "avg_traffic", "avg_nodes_traffic", "avg_pollution", "avg_traffic", "avg_nodes_traffic",
# extra_regressors = ["temperature", "wind", ]
extra_regressors = []
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
    # name="SourceModelsRelErr" if train_with_relative_error else "SourceModels",
    # name="Sauron",
    # name="Kenobi",
    # name="Araucaria",# shuffle True all stations
    # name="Lenga",  # shuffle False all stations
    name="Mara",  # shuffle False all stations no convolution for node traffic
    country_alpha_code="FR",
    trackCO2=True
)

if __name__ == "__main__":
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
    }

    # models2 = dict()
    #
    # models2["Spatial Avg"] = models["Spatial Avg"]
    # models2["BLUE"] = models["BLUE"]
    # models2["ExponentialFit"] = models["ExponentialFit"]
    # for source_model_name, source_model in [
    #     ("poly2", Pipeline([("PF", PolynomialFeatures(degree=2)),
    #                         ("LR", LassoCV(selection="cyclic", positive=False, cv=len(stations2test) - 1
    #                                        ))])),
    #     ("poly3", Pipeline([("PF", PolynomialFeatures(degree=3)),
    #                         ("LR", LassoCV(selection="cyclic", positive=False, cv=len(stations2test) - 1
    #                                        ))])),
    #     ("poly2NN", Pipeline([("PF", PolynomialFeatures(degree=2)),
    #                           ("NN", MLPRegressor(hidden_layer_sizes=hidden_layer_sizes,
    #                                               activation=activation,  # 'relu',
    #                                               learning_rate_init=learning_rate_init,
    #                                               learning_rate=learning_rate,
    #                                               early_stopping=early_stopping,
    #                                               solver=solver.lower(),
    #                                               max_iter=max_iter))])),
    #     ("linear", LassoCV(selection="cyclic", positive=False, cv=len(stations2test) - 1)),
    #     ("nn", MLPRegressor(hidden_layer_sizes=hidden_layer_sizes,
    #                         activation=activation,  # 'relu',
    #                         learning_rate_init=learning_rate_init,
    #                         learning_rate=learning_rate,
    #                         early_stopping=early_stopping,
    #                         solver=solver.lower(),
    #                         max_iter=max_iter)),
    #     ("RF", RandomForestRegressor(n_estimators=25, max_depth=3))
    # ]:
    #     models2[
    #         f"node_{source_model_name}_{''.join([e[0].upper() for e in extra_regressors])}{source_dist4name}"] = NodeSourceModel(
    #         train_with_relative_error=train_with_relative_error,
    #         path4preprocess=data_manager.path, graph=graph,
    #         spacial_locations=station_coordinates,
    #         times=times_all,
    #         traffic_by_edge=traffic_by_edge,
    #         redo_preprocessing=False,
    #         name="", loss=mse, optim_method=NONE_OPTIM_METHOD,
    #         verbose=True, niter=1, sigma0=1,
    #         lnei=1,
    #         source_model=source_model,
    #         substract_mean=True, sources_dist=sources_dist,
    #         extra_regressors=extra_regressors,
    #     )
    #     for basis in ["geometrical", "pca", "geometrical_log", "pca_log"]:  # , "both"
    #         models2[
    #             f"{basis}_{source_model_name}_{''.join([e[0].upper() for e in extra_regressors])}{source_dist4name}{'' if k == 5 else '_' + str(k)}"] = ProjectionFullSourceModel(
    #             train_with_relative_error=train_with_relative_error,
    #             path4preprocess=data_manager.path, graph=graph,
    #             spacial_locations=station_coordinates,
    #             times=times_all,
    #             traffic_by_edge=traffic_by_edge,
    #             redo_preprocessing=False,
    #             name="", loss=mse, optim_method=NONE_OPTIM_METHOD,
    #             verbose=True, niter=25, sigma0=1,
    #             lnei=1, k_max=10,
    #             source_model=source_model, sources_dist=sources_dist,
    #             substract_mean=True,  # cv_in_space=False,
    #             extra_regressors=extra_regressors,
    #             basis=basis,
    #             # kv=Optim(5, None, None), ky=Optim(5, None, None), kr=Optim(5, None, None), kd=Optim(5, None, None),
    #             kv=k, ky=k, kr=k, kd=k,
    #             # kv=10, ky=3, kr=3, kd=1,
    #             # D0=Optim(1e4, 1e-4, 1e4), A0=Optim(1e-4, 1e-4, 1e4),
    #             D0=0.0, A0=0.0,
    #             D1=0.0, A1=0.0,
    #             D2=0.0, A2=0.0,
    #             D3=0.0, A3=0.0,
    #             # forward_weight0=0.0, source_weight0=Optim(0.999, 0.0, 1.0),
    #             # forward_weight1=0.0, source_weight1=Optim(0.999, 0.0, 1.0),
    #             # forward_weight2=0.0, source_weight2=Optim(0.999, 0.0, 1.0),
    #             # forward_weight3=0.0, source_weight3=Optim(0.999, 0.0, 1.0),
    #             forward_weight0=0.0, source_weight0=1,
    #             forward_weight1=0.0, source_weight1=1,
    #             forward_weight2=0.0, source_weight2=1,
    #             forward_weight3=0.0, source_weight3=1,
    #         )
    #
    # lab = LabPipeline()
    # lab.define_new_block_of_functions(
    #     "individual_models",
    #     *list(map(train_test_model, models2.items())),
    #     recalculate=False
    # )
    #
    # lab.execute(
    #     data_manager,
    #     num_cores=15,
    #     forget=False,
    #     save_on_iteration=1,
    #     station=stations2test
    # )

    # ------------------------------------------------------- #
    #                       Averager models                   #
    # ------------------------------------------------------- #
    pca_log_poly2_only005_10 = ProjectionFullSourceModel(
        train_with_relative_error=train_with_relative_error,
        path4preprocess=data_manager.path, graph=graph,
        spacial_locations=station_coordinates,
        times=times_all,
        traffic_by_edge=traffic_by_edge,
        redo_preprocessing=False,
        name="", loss=mse, optim_method=NONE_OPTIM_METHOD,
        verbose=True, niter=25, sigma0=1,
        lnei=1, k_max=10,
        source_model=Pipeline([("PF", PolynomialFeatures(degree=2)),
                               ("LR", LassoCV(selection="cyclic", positive=False, cv=len(stations2test) - 1))]),
        sources_dist=[0.005],
        substract_mean=True,
        extra_regressors=[],
        basis="pca_log",
        kv=5, ky=5, kr=5, kd=5,
        D0=0.0, A0=0.0,
        D1=0.0, A1=0.0,
        D2=0.0, A2=0.0,
        D3=0.0, A3=0.0,
        forward_weight0=0.0, source_weight0=1,
        forward_weight1=0.0, source_weight1=1,
        forward_weight2=0.0, source_weight2=1,
        forward_weight3=0.0, source_weight3=1,
    )
    node_linear_TW = NodeSourceModel(
        train_with_relative_error=train_with_relative_error,
        path4preprocess=data_manager.path, graph=graph,
        spacial_locations=station_coordinates,
        times=times_all,
        traffic_by_edge=traffic_by_edge,
        redo_preprocessing=False,
        name="", loss=mse, optim_method=NONE_OPTIM_METHOD,
        verbose=True, niter=1, sigma0=1,
        lnei=1,
        source_model=LassoCV(selection="cyclic", positive=False, cv=len(stations2test) - 1),
        substract_mean=True, sources_dist=[0],
        extra_regressors=["temperature", "wind"],
    )
    geometrical_poly3NN_ = ProjectionFullSourceModel(
        train_with_relative_error=train_with_relative_error,
        path4preprocess=data_manager.path, graph=graph,
        spacial_locations=station_coordinates,
        times=times_all,
        traffic_by_edge=traffic_by_edge,
        redo_preprocessing=False,
        name="", loss=mse, optim_method=NONE_OPTIM_METHOD,
        verbose=True, niter=25, sigma0=1,
        lnei=1, k_max=10,
        source_model=Pipeline([
            ("PF", PolynomialFeatures(degree=3)),
            ("NN", MLPRegressor(hidden_layer_sizes=hidden_layer_sizes,
                                activation=activation,  # 'relu',
                                learning_rate_init=learning_rate_init,
                                learning_rate=learning_rate,
                                early_stopping=early_stopping,
                                solver=solver.lower(),
                                max_iter=max_iter))]),
        sources_dist=[0],
        substract_mean=True,
        extra_regressors=[],
        basis="geometrical",
        kv=5, ky=5, kr=5, kd=5,
        D0=0.0, A0=0.0,
        D1=0.0, A1=0.0,
        D2=0.0, A2=0.0,
        D3=0.0, A3=0.0,
        forward_weight0=0.0, source_weight0=1,
        forward_weight1=0.0, source_weight1=1,
        forward_weight2=0.0, source_weight2=1,
        forward_weight3=0.0, source_weight3=1,
    )
    pca_linear_TW = ProjectionFullSourceModel(
        train_with_relative_error=train_with_relative_error,
        path4preprocess=data_manager.path, graph=graph,
        spacial_locations=station_coordinates,
        times=times_all,
        traffic_by_edge=traffic_by_edge,
        redo_preprocessing=False,
        name="", loss=mse, optim_method=NONE_OPTIM_METHOD,
        verbose=True, niter=25, sigma0=1,
        lnei=1, k_max=10,
        source_model=LassoCV(selection="cyclic", positive=False, cv=len(stations2test) - 1),
        sources_dist=[0],
        substract_mean=True,
        extra_regressors=["temperature", "wind"],
        basis="pca",
        kv=5, ky=5, kr=5, kd=5,
        D0=0.0, A0=0.0,
        D1=0.0, A1=0.0,
        D2=0.0, A2=0.0,
        D3=0.0, A3=0.0,
        forward_weight0=0.0, source_weight0=1,
        forward_weight1=0.0, source_weight1=1,
        forward_weight2=0.0, source_weight2=1,
        forward_weight3=0.0, source_weight3=1,
    )

    models2 = {
        # "node_linear_TW": node_linear_TW,
        # "geometrical_poly2NN_v2": geometrical_poly2NN_,
        # "pca_linear_TW": pca_linear_TW,
        # "pca_log_poly2_only005_10": pca_log_poly2_only005_10,
        # "EnsembleAvgNoCV_Lasso": ModelsAggregatorNoCV(
        #     models=[node_linear_TW, geometrical_poly3NN_, pca_linear_TW, pca_log_poly2_only005_10],
        #     aggregator=Pipeline([("lasso", LassoCV(selection="cyclic", positive=False, cv=len(stations2test) - 1))]),
        #     extra_regressors=[]
        # ),
        # "EnsembleAvgNoCV_Poly2": ModelsAggregatorNoCV(
        #     models=[node_linear_TW, geometrical_poly3NN_, pca_linear_TW, pca_log_poly2_only005_10],
        #     aggregator=Pipeline([("PF", PolynomialFeatures(degree=2)),
        #                          ("lasso", LassoCV(selection="cyclic", positive=False, cv=len(stations2test) - 1))]),
        #     extra_regressors=[]
        # ),
        # "EnsembleAvgNoCV_avg": ModelsAggregatorNoCV(
        #     models=[node_linear_TW, geometrical_poly3NN_, pca_linear_TW, pca_log_poly2_only005_10],
        #     # models=[m for k, m in models2.items() if "geometrical" in k],
        #     # models=[
        #     #     models2["geometrical_nn_"],
        #     #     models2["geometrical_poly2NN_"],
        #     #     models2["geometrical_poly3_"],
        #     #     models2["geometrical_poly2_"],
        #     #     models2["geometrical_nn_"]
        #     # ],
        #     # models=[models2["geometrical_RF_TWHW"], models2["node_linear_TWHW"], models2["pca_linear_TWHW"],
        #     #         # models["ExponentialFit"], models["Spatial Avg"]
        #     #         ],
        #     aggregator="average",
        #     extra_regressors=[]
        # )
        # "Ensemble": ModelsAggregator(
        #     models=[node_linear_TW, geometrical_poly3NN_, pca_linear_TW, pca_log_poly2_only005_10],
        #     aggregator=Pipeline([("lasso", LassoCV(selection="cyclic", positive=False, cv=len(stations2test) - 1))]),
        #     name=None,
        #     weighting="average",
        #     train_on_llo=True),

        "Ensemble2": ModelsAggregator(
            models=[node_linear_TW, geometrical_poly3NN_, ],  # pca_log_poly2_only005_10 pca_linear_TW
            aggregator=Pipeline([("lasso", LassoCV(selection="cyclic", positive=False, cv=len(stations2test) - 1))]),
            name=None,
            weighting="average",
            train_on_llo=True)
    }

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
        save_on_iteration=15,
        station=stations2test
    )
