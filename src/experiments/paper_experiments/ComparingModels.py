from collections import OrderedDict
from datetime import datetime
from functools import partial

import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from spiderplot import spiderplot

import src.config as config
from PerplexityLab.DataManager import DataManager
from PerplexityLab.LabPipeline import LabPipeline
from PerplexityLab.miscellaneous import NamedPartial, copy_main_script_version
from PerplexityLab.visualization import generic_plot
from src.experiments.paper_experiments.PreProcessPaper import train_test_model, train_test_averagers, stations2test, \
    plot_pollution_map_in_graph, times_future, graph, pollution_past
from src.experiments.paper_experiments.params4runs import path2latex_figures
from src.lib.FeatureExtractors.GraphFeatureExtractors import label_prop, diffusion_eq
from src.lib.Models.BaseModel import ModelsSequenciator, \
    medianse, NONE_OPTIM_METHOD, mse, GRAD
from src.lib.Models.TrueStateEstimationModels.AverageModels import SnapshotMeanModel
from src.lib.Models.TrueStateEstimationModels.GraphModels import GraphEmissionsNeigEdgeModel
from src.lib.Models.TrueStateEstimationModels.KernelModels import GaussianKernelModel
from src.lib.Modules import Optim

if __name__ == "__main__":
    k_neighbours = 10
    # experiment_name = f"MapExtraRegressors{if_true_str(shuffle, '_Shuffled')}" \
    #                   f"{if_true_str(simulation, '_Sim')}{if_true_str(filter_graph, '_Gfiltered')}"
    experiment_name = "Paper2"

    data_manager = DataManager(
        path=config.results_dir,
        name=experiment_name,
        country_alpha_code="FR",
        trackCO2=True
    )
    copy_main_script_version(__file__, data_manager.path)

    base_models = [
        SnapshotMeanModel(summary_statistic="mean"),
        GaussianKernelModel(sigma=Optim(start=0.01266096365565058, lower=0.001, upper=0.1),
                            beta=Optim(start=np.log(1), lower=np.log(0.01), upper=np.log(2)),
                            distrust=0,
                            name="", loss=mse, optim_method=GRAD, niter=1000, verbose=False)

    ]
    # 621.5069384089682 = [2.87121906 0.16877082 1.04179242 1.23798909 3.42959526 3.56328527]
    models = [
        ModelsSequenciator(
            name="AvgKrigging",
            models=[
                SnapshotMeanModel(summary_statistic="mean"),
                GaussianKernelModel(sigma=Optim(start=0.01266096365565058, lower=0.001, upper=0.1),
                                    beta=Optim(start=np.log(1), lower=np.log(0.01), upper=np.log(2)),
                                    distrust=0,
                                    name="", loss=mse, optim_method=GRAD, niter=1000, verbose=False)
            ],
            transition_model=[
                Pipeline([("LR", LinearRegression())]),
                Pipeline([("Lss", LassoCV(selection="cyclic"))])
            ]
        ),
        # ModelsSequenciator(
        #     name="LR",
        #     models=[
        #         SnapshotMeanModel(summary_statistic="mean"),
        #         GraphEmissionsNeigEdgeModel(
        #             extra_regressors=[],
        #             k_neighbours=k_neighbours,
        #             # model=Pipeline(steps=[("zscore", StandardScaler()), ("LR", LinearRegression())]),
        #             model=Pipeline(steps=[("zscore", StandardScaler()), ("Lasso", LassoCV(selection="cyclic"))]),
        #             niter=2, verbose=True,
        #             optim_method=NONE_OPTIM_METHOD,
        #             loss=medianse)
        #     ],
        #     transition_model=[
        #         Pipeline([("LR", LinearRegression())]),
        #         Pipeline([("Lss", LassoCV(selection="cyclic"))])
        #     ]
        # ),
        # ModelsSequenciator(
        #     name="LR_Extra",
        #     models=[
        #         SnapshotMeanModel(summary_statistic="mean"),
        #         GraphEmissionsNeigEdgeModel(
        #             extra_regressors=["temperature", "wind"],
        #             k_neighbours=k_neighbours,
        #             # model=Pipeline(steps=[("zscore", StandardScaler()), ("LR", LinearRegression())]),
        #             model=Pipeline(steps=[("zscore", StandardScaler()), ("Lasso", LassoCV(selection="cyclic"))]),
        #             niter=2, verbose=True,
        #             optim_method=NONE_OPTIM_METHOD,
        #             loss=medianse)
        #     ],
        #     transition_model=[
        #         Pipeline([("LR", LinearRegression())]),
        #         Pipeline([("Lss", LassoCV(selection="cyclic"))])
        #     ]
        # ),
        # ModelsSequenciator(
        #     name="NN_Extra",
        #     models=[
        #         SnapshotMeanModel(summary_statistic="mean"),
        #         GraphEmissionsNeigEdgeModel(
        #             extra_regressors=["temperature", "wind"],
        #             k_neighbours=k_neighbours,
        #             # model=Pipeline(steps=[("zscore", StandardScaler()), ("LR", LinearRegression())]),
        #             # model=Pipeline(steps=[("zscore", StandardScaler()), ("Lasso", LassoCV(selection="cyclic"))]),
        #             model=Pipeline(
        #                 steps=[("zscore", StandardScaler()), ("NN", MLPRegressor(hidden_layer_sizes=(20, 20,),
        #                                                                          activation="relu",  # 'relu',
        #                                                                          learning_rate_init=0.1,
        #                                                                          learning_rate="adaptive",
        #                                                                          early_stopping=True,
        #                                                                          solver="adam",
        #                                                                          max_iter=10000))]),
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
    lab.define_new_block_of_functions(
        "train_individual_models",
        *list(map(train_test_model,
                  base_models +
                  models
                  )),
        recalculate=False
    )
    lab.define_new_block_of_functions(
        "model",
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
                  )),
        recalculate=False
    )

    lab.execute(
        data_manager,
        num_cores=10,
        forget=False,
        save_on_iteration=None,
        # station=["BONAP"],  # stations2test
        station=stations2test
    )

    # ----- Plotting results ----- #

    model_names = OrderedDict([
        ("SnapshotMeanModelmean", "Average in space"),
        ("GaussianKernelModel", "Krigging"),
        ("AvgKrigging", "Average - Krigging"),
        ("LR", "Graph \n Linear Regression"),
        ("LR_Extra", "Graph Temp Wind \n Linear Regression"),
        ("NN_Extra", "Graph Temp Wind \n NeuralNetwork"),
    ])

    models_order = list(model_names.values()) + ["Ensamble"]
    models_order.remove("Krigging")
    models2plot = set(data_manager["model"])
    models2plot.remove("GaussianKernelModel")
    models2plot = list(models2plot)


    def name_models(model):
        if "Pipeline" in model:
            return "Ensamble"
        else:
            return model_names[model]


    plot_pollution_map_in_graph(
        data_manager=data_manager,
        folder=path2latex_figures,
        time=times_future[4],
        # diffusion_method=partial(diffusion_eq, path=data_manager.path, graph=graph, diffusion_coef=1,
        #                          absorption_coef=0.01, recalculate=False),
        # diffusion_method=partial(label_prop, graph=graph,
        #                          edge_function=lambda data: 1.0 / data["length"],
        #                          lamb=1, iter=10, p=0.5),
        # diffusion_method=lambda f: label_prop(f=f + np.random.uniform(size=np.shape(f)), graph=graph,
        #                                       edge_function=lambda data: 1.0 / data["length"],
        #                                       lamb=1, iter=10, p=0.1),
        diffusion_method=lambda f: f,
        # time=times_future[4], Screenshot_48.8580073_2.3342828_13_2022_12_8_13_15
        # time=pollution_past.index[11],
        station="OPERA",
        plot_by=["model", "station"],
        num_cores=1, models=name_models, model=[
            # "SnapshotMeanModelmean", "AvgKrigging",
            "GaussianKernelModel"
        ],
        nodes_indexes=np.arange(len(graph)), s=10,
        cmap=sns.color_palette("autumn", as_cmap=True), alpha=0.7, dpi=300,
        format=".pdf")

    # plot_pollution_map_in_graph(
    #     data_manager=data_manager,
    #     folder=path2latex_figures,
    #     time=times_future[4], station="OPERA",
    #     plot_by=["model", "station"],
    #     num_cores=10, models=name_models, model=["LR", "LR_Extra", "NN_Extra", "Ensamble"], s=10,
    #     nodes_indexes=np.arange(len(graph)),
    #     # nodes_indexes=np.random.choice(len(graph), size=2000, replace=False),
    #     cmap=sns.color_palette("coolwarm", as_cmap=True), alpha=0.7, dpi=300,
    #     format=".pdf")

    generic_plot(
        data_manager=data_manager,
        folder=path2latex_figures,
        x="models", y="mse",
        plot_func=NamedPartial(sns.barplot, orient="vertical", order=models_order, errorbar=("ci", 0)),
        # plot_func=sns.barplot,
        sort_by=["models"],
        mse=lambda error: np.sqrt(error.mean()),
        models=name_models,
        model=models2plot,
        dpi=300,
        format=".pdf"
    )

    stations_order = ["OPERA", "BASCH", "PA13", "PA07", "PA18", "ELYS", "PA12", "BONAP"]

    generic_plot(
        data_manager=data_manager,
        folder=path2latex_figures,
        x="station", y="mse", label="models",
        plot_func=NamedPartial(sns.barplot, orient="vertical", hue_order=models_order,
                               order=stations_order),
        sort_by=["models"],
        mse=lambda error: np.sqrt(error.mean()),
        models=name_models,
        model=models2plot,
        dpi=300,
        format=".pdf"
    )

    generic_plot(data_manager, x="l1_error", y="model",
                 plot_func=NamedPartial(sns.boxenplot, orient="horizontal", order=stations_order),
                 sort_by=["mse"],
                 l1_error=lambda error: np.abs(error).ravel(),
                 mse=lambda error: np.sqrt(error.mean()),
                 model=models2plot
                 # xlim=(0, 20)
                 )

    generic_plot(data_manager, x="mse", y="model",
                 plot_func=NamedPartial(sns.violinplot, orient="horizontal", inner="stick", order=stations_order),
                 sort_by=["mse"],
                 mse=lambda error: np.sqrt(np.nanmean(error)),
                 models=name_models,
                 model=models2plot
                 # xlim=(0, 100),
                 # model=["SnapshotMeanModelmean", "A+SMM,GMM,SMMTCMN", "GlobalMeanModelmean"]
                 )

    generic_plot(data_manager, x="model", y="time_to_fit", plot_func=sns.boxenplot, model=models2plot)
    generic_plot(data_manager, x="model", y="time_to_estimate", plot_func=sns.boxenplot, model=models2plot)

    generic_plot(data_manager, x="station", y="mse", label="models",
                 plot_func=NamedPartial(spiderplot, hue=models_order),
                 mse=lambda error: np.sqrt(error.mean()),
                 models=name_models,
                 model=models2plot
                 )

    generic_plot(data_manager, x="station", y="l1_error", label="model", plot_func=sns.boxenplot,
                 sort_by=["mse"],
                 l1_error=lambda error: np.abs(error).ravel(),
                 mse=lambda error: np.sqrt(error.mean()),
                 model=models2plot
                 )
