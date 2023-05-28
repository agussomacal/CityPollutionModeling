from functools import partial
from typing import Dict, List

import numpy as np
import seaborn as sns
from sklearn.base import RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from spiderplot import spiderplot

import src.config as config
from PerplexityLab.DataManager import DataManager
from PerplexityLab.LabPipeline import LabPipeline
from PerplexityLab.miscellaneous import NamedPartial, if_true_str, copy_main_script_version
from PerplexityLab.visualization import generic_plot
from src.experiments.PreProcess import train_test_model, train_test_averagers, simulation, stations2test
from src.experiments.config_experiments import shuffle, filter_graph
from src.lib.Models.BaseModel import ModelsSequenciator, \
    medianse, NONE_OPTIM_METHOD
from src.lib.Models.TrueStateEstimationModels.AverageModels import SnapshotMeanModel
from src.lib.Models.TrueStateEstimationModels.GraphModels import GraphEmissionsNeigEdgeModel


# from spektral.layers import GATConv


class RFCV(RegressorMixin):
    def __init__(
            self,
            parameters_for_cross_validation: Dict[str, List],
            cv=10,
            n_estimators=100,
            criterion="squared_error",
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            min_weight_fraction_leaf=0.0,
            max_features=1.0,
            max_leaf_nodes=None,
            min_impurity_decrease=0.0,
            bootstrap=True,
            oob_score=False,
            n_jobs=None,
            random_state=None,
            verbose=0,
            warm_start=False,
            ccp_alpha=0.0,
            max_samples=None,
    ):
        # https://stackoverflow.com/questions/38151615/specific-cross-validation-with-random-forest
        self.parameters_for_cross_validation = parameters_for_cross_validation
        self.cv = cv
        self.rf = RandomForestRegressor(
            n_estimators=n_estimators,
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            ccp_alpha=ccp_alpha,
            max_samples=max_samples,
        )

    def fit(self, X, y, sample_weight=None):
        grid_clf = GridSearchCV(self.rf, self.parameters_for_cross_validation, cv=self.cv)
        grid_clf.fit(X, y.ravel())
        self.rf = grid_clf.best_estimator_
        return self

    def predict(self, X):
        return self.rf.predict(X)


if __name__ == "__main__":
    k_neighbours = 10
    experiment_name = f"MapExtraRegressors{if_true_str(shuffle, '_Shuffled')}" \
                      f"{if_true_str(simulation, '_Sim')}{if_true_str(filter_graph, '_Gfiltered')}"

    data_manager = DataManager(
        path=config.results_dir,
        name=experiment_name,
        country_alpha_code="FR",
        trackCO2=True
    )
    copy_main_script_version(__file__, data_manager.path)

    base_models = [
        SnapshotMeanModel(summary_statistic="mean"),
        # SnapshotBLUEModel(sensor_distrust=0),
        # SnapshotPCAModel(n_components=1, summary_statistic="mean"),
        # GlobalMeanModel()
    ]
    # 621.5069384089682 = [2.87121906 0.16877082 1.04179242 1.23798909 3.42959526 3.56328527]
    models = [
        ModelsSequenciator(
            name="LR",
            models=[
                SnapshotMeanModel(summary_statistic="mean"),
                GraphEmissionsNeigEdgeModel(
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
        ModelsSequenciator(
            name="LR_Extra",
            models=[
                SnapshotMeanModel(summary_statistic="mean"),
                GraphEmissionsNeigEdgeModel(
                    extra_regressors=["temperature", "wind", "water", "green"],
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
        ModelsSequenciator(
            name="NN_Extra",
            models=[
                SnapshotMeanModel(summary_statistic="mean"),
                GraphEmissionsNeigEdgeModel(
                    extra_regressors=["temperature", "wind", "water", "green"],
                    k_neighbours=k_neighbours,
                    # model=Pipeline(steps=[("zscore", StandardScaler()), ("LR", LinearRegression())]),
                    # model=Pipeline(steps=[("zscore", StandardScaler()), ("Lasso", LassoCV(selection="cyclic"))]),
                    model=Pipeline(
                        steps=[("zscore", StandardScaler()), ("NN", MLPRegressor(hidden_layer_sizes=(20, 20,),
                                                                                 activation="logistic",
                                                                                 learning_rate_init=0.1,
                                                                                 max_iter=1000))]),
                    niter=2, verbose=True,
                    optim_method=NONE_OPTIM_METHOD,
                    loss=medianse)
            ],
            transition_model=[
                Pipeline([("LR", LinearRegression())]),
                Pipeline([("Lss", LassoCV(selection="cyclic"))])
            ]
        ),
        # ModelsSequenciator(
        #     name="NN_TW",
        #     models=[
        #         SnapshotMeanModel(summary_statistic="mean"),
        #         GraphEmissionsNeigEdgeModel(
        #             extra_regressors=["temperature", "wind"],
        #             k_neighbours=k_neighbours,
        #             # model=Pipeline(steps=[("zscore", StandardScaler()), ("LR", LinearRegression())]),
        #             model=Pipeline(steps=[("NN", MLPRegressor(hidden_layer_sizes=(20, 20,),
        #                                                       activation="logistic",
        #                                                       learning_rate_init=0.1,
        #                                                       max_iter=1000))]),
        #             niter=2, verbose=True,
        #             optim_method=NONE_OPTIM_METHOD,
        #             loss=medianse)
        #     ],
        #     transition_model=[
        #         Pipeline([("LR", LinearRegression())]),
        #         Pipeline([("LR", LinearRegression())])
        #     ]
        # ),
        # ModelsSequenciator(
        #     name="NN",
        #     models=[
        #         SnapshotMeanModel(summary_statistic="mean"),
        #         GraphEmissionsNeigEdgeModel(
        #             k_neighbours=k_neighbours,
        #             model=Pipeline(steps=[("zscore", StandardScaler()), ("NN",
        #                                                                  MLPRegressor(hidden_layer_sizes=(20, 20,),
        #                                                                               activation="logistic",
        #                                                                               learning_rate_init=0.1,
        #                                                                               max_iter=1000))]),
        #             niter=2, verbose=True,
        #             optim_method=NONE_OPTIM_METHOD,
        #             loss=medianse)
        #     ],
        #     transition_model=[
        #         Pipeline([("LR", LinearRegression())]),
        #         Pipeline([("Lss", LassoCV(selection="cyclic"))])
        #     ]
        # ),
    ]

    lab = LabPipeline()
    lab.define_new_block_of_functions("train_individual_models", *list(map(train_test_model,
                                                                           base_models +
                                                                           models
                                                                           )))
    lab.define_new_block_of_functions("model",
                                      *list(map(partial(train_test_averagers,
                                                        aggregator=Pipeline([(
                                                            # "NN",
                                                            # MLPRegressor(hidden_layer_sizes=(20, 20,),
                                                            #              activation="logistic",
                                                            #              learning_rate_init=0.1,
                                                            #              max_iter=1000))
                                                            "LR",
                                                            LassoCV(selection="random"))
                                                        ])),
                                                [[model] for model in models + base_models]
                                                + [models + base_models]
                                                )))

    lab.execute(
        data_manager,
        num_cores=14,
        forget=False,
        recalculate=False,
        save_on_iteration=1,
        # station=["BONAP"],  # stations2test
        station=stations2test
    )


    # ----- Plotting results ----- #
    # plot_pollution_map(data_manager, time=times_future[0], station="OPERA", plot_by=["model", "station"], num_cores=10,
    #                    num_points=100, cmap="cividis")

    def name_models(model):
        if model == "LR":
            return "Graph \n Linear Regression"
        elif model == "LR_Extra":
            return "Graph Temp Wind \n Linear Regression"
        elif model == "NN_Extra":
            return "Graph Temp Wind \n NeuralNetwork"
        elif model == "SnapshotMeanModelmean":
            return "Average in space"
        elif "Pipeline" in model:
            return "Ensamble"


    models_order = ["Average in space", "Graph \n Linear Regression", "Graph Temp Wind \n Linear Regression",
                    "Graph Temp Wind \n NeuralNetwork", "Ensamble"]
    generic_plot(data_manager,
                 x="models", y="mse",
                 plot_func=NamedPartial(sns.barplot, orient="vertical", order=models_order, errorbar=("ci", 0)),
                 # plot_func=sns.barplot,
                 sort_by=["models"],
                 mse=lambda error: np.sqrt(error.mean()),
                 models=name_models,
                 )


    def name_models(model):
        if model == "LR":
            return "Graph Linear Regression"
        elif model == "LR_Extra":
            return "Graph Temp Wind Linear Regression"
        elif model == "NN_Extra":
            return "Graph Temp Wind NeuralNetwork"
        elif model == "SnapshotMeanModelmean":
            return "Average in space"
        elif "Pipeline" in model:
            return "Ensamble"


    models_order = ["Average in space", "Graph Linear Regression", "Graph Temp Wind Linear Regression",
                    "Graph Temp Wind NeuralNetwork", "Ensamble"]
    stations_order = ["OPERA", "BASCH", "PA13", "PA07", "PA18", "ELYS", "PA12", "BONAP"]

    generic_plot(data_manager,
                 x="station", y="mse", label="models",
                 plot_func=NamedPartial(sns.barplot, orient="vertical", hue_order=models_order,
                                        order=stations_order),
                 sort_by=["models"],
                 mse=lambda error: np.sqrt(error.mean()),
                 models=name_models,
                 )

    generic_plot(data_manager, x="l1_error", y="model",
                 plot_func=NamedPartial(sns.boxenplot, orient="horizontal", order=stations_order),
                 sort_by=["mse"],
                 l1_error=lambda error: np.abs(error).ravel(),
                 mse=lambda error: np.sqrt(error.mean()),
                 # xlim=(0, 20)
                 )

    generic_plot(data_manager, x="mse", y="model",
                 plot_func=NamedPartial(sns.violinplot, orient="horizontal", inner="stick", order=stations_order),
                 sort_by=["mse"],
                 mse=lambda error: np.sqrt(np.nanmean(error)),
                 models=name_models,
                 # xlim=(0, 100),
                 # model=["SnapshotMeanModelmean", "A+SMM,GMM,SMMTCMN", "GlobalMeanModelmean"]
                 )

    generic_plot(data_manager, x="model", y="time_to_fit", plot_func=sns.boxenplot)
    generic_plot(data_manager, x="model", y="time_to_estimate", plot_func=sns.boxenplot)

    generic_plot(data_manager, x="station", y="mse", label="models",
                 plot_func=NamedPartial(spiderplot, hue=models_order),
                 mse=lambda error: np.sqrt(error.mean()),
                 models=name_models,
                 )

    generic_plot(data_manager, x="station", y="l1_error", label="model", plot_func=sns.boxenplot,
                 sort_by=["mse"],
                 l1_error=lambda error: np.abs(error).ravel(),
                 mse=lambda error: np.sqrt(error.mean()),
                 )
