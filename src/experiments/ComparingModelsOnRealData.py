import numpy as np
import seaborn as sns
from spiderplot import spiderplot

import src.config as config
from PerplexityLab.DataManager import DataManager
from PerplexityLab.LabPipeline import LabPipeline
from src.experiments.PreProcess import longer_distance, loo4test, train_test_model, station_coordinates
from src.experiments.config_experiments import num_cores
from src.lib.Models.BaseModel import Bounds, mse, UNIFORM, ModelsSequenciator, \
    ModelsAverager
from src.lib.Models.TrueStateEstimationModels.AverageModels import SnapshotMeanModel, GlobalMeanModel, \
    SnapshotWeightedModel
from src.lib.Models.TrueStateEstimationModels.TrafficConvolution import TrafficMeanModel, TrafficConvolutionModel, \
    gaussker
from PerplexityLab.miscellaneous import NamedPartial
from PerplexityLab.visualization import generic_plot

if __name__ == "__main__":
    experiment_name = "ModelComparison"

    data_manager = DataManager(
        path=config.results_dir,
        name=experiment_name
    )

    models_first = [
        ModelsSequenciator(models=[
            SnapshotMeanModel(summary_statistic="mean"),
            TrafficMeanModel(summary_statistic="mean"),
        ])
        ,
        ModelsAverager(models=[GlobalMeanModel(),
                               TrafficMeanModel(summary_statistic="mean"),
                               SnapshotMeanModel(summary_statistic="mean"),
                               ModelsSequenciator(models=[
                                   SnapshotMeanModel(summary_statistic="mean"),
                                   TrafficMeanModel(summary_statistic="mean"),
                               ])
                               ],
                       positive=True, fit_intercept=False),
        SnapshotMeanModel(summary_statistic="mean"),
        GlobalMeanModel(),
        TrafficMeanModel(summary_statistic="mean"),
    ]

    models_second = [
        ModelsSequenciator(models=[
            SnapshotMeanModel(summary_statistic="mean"),
            TrafficConvolutionModel(conv_kernel=0, normalize=False,
                                    sigma=Bounds(0, 2 * longer_distance),
                                    loss=mse, optim_method=UNIFORM, niter=10, verbose=True)
            ,
        ])
        ,
        ModelsAverager(models=[GlobalMeanModel(),
                               TrafficMeanModel(summary_statistic="mean"),
                               SnapshotMeanModel(summary_statistic="mean"),
                               ModelsSequenciator(models=[
                                   SnapshotMeanModel(summary_statistic="mean"),
                                   TrafficMeanModel(summary_statistic="mean"),
                               ])
                               ],
                       positive=True, fit_intercept=False),
        ModelsSequenciator(models=[
            SnapshotMeanModel(summary_statistic="mean"),
            TrafficConvolutionModel(conv_kernel=gaussker, normalize=False,
                                    sigma=Bounds(0, 2 * longer_distance),
                                    loss=mse, optim_method=UNIFORM, niter=10, verbose=True)],
            name="SnapTrafficConv"),
        ModelsAverager(models=[
            GlobalMeanModel(),
            TrafficConvolutionModel(conv_kernel=gaussker, normalize=False,
                                    sigma=Bounds(0, 2 * longer_distance),
                                    loss=mse, optim_method=UNIFORM, niter=10, verbose=True),
            TrafficMeanModel(summary_statistic="mean"),
            TrafficMeanModel(summary_statistic="median"),
            SnapshotMeanModel(summary_statistic="mean"),
            SnapshotMeanModel(summary_statistic="median"),
            SnapshotWeightedModel(positive=True, fit_intercept=True),
        ],
            positive=True, fit_intercept=False),
    ]

    for models in [models_first]:  # , models_second
        lab = LabPipeline()
        # lab.define_new_block_of_functions("true_values", loo4test)
        lab.define_new_block_of_functions("model", *list(map(train_test_model, models)))
        lab.execute(
            data_manager,
            num_cores=num_cores,
            forget=True,
            recalculate=True,
            save_on_iteration=None,
            station=station_coordinates.columns.to_list()
        )

        # ----- Plotting results ----- #
        generic_plot(data_manager, x="station", y="mse", label="model", plot_func=spiderplot,
                     mse=lambda error: np.sqrt(error.mean()))

        generic_plot(data_manager, x="station", y="l1_error", label="model", plot_func=sns.boxenplot,
                     sort_by=["mse"],
                     l1_error=lambda error: np.abs(error).ravel(),
                     mse=lambda error: np.sqrt(error.mean()))

        generic_plot(data_manager, x="l1_error", y="model", plot_func=NamedPartial(sns.boxenplot, orient="horizontal"),
                     sort_by=["mse"],
                     l1_error=lambda error: np.abs(error).ravel(),
                     mse=lambda error: np.sqrt(error.mean()), xlim=(0, 20))

        generic_plot(data_manager, x="mse", y="model", plot_func=NamedPartial(sns.boxenplot, orient="horizontal"),
                     sort_by=["mse"],
                     mse=lambda error: np.sqrt(np.nanmean(error)), xlim=(0, 20))

        # generic_plot(data_manager, x="station", y="mse", label="model", plot_func=spiderplot,
        #              mse=lambda estimation, future_pollution:
        #              np.sqrt(((estimation - future_pollution.values.ravel()).ravel() ** 2).mean()))
        #
        # generic_plot(data_manager, x="station", y="error", label="model", plot_func=sns.boxenplot,
        #              error=lambda estimation, future_pollution:
        #              np.abs((estimation - future_pollution.values.ravel()).ravel()))

        generic_plot(data_manager, x="model", y="time_to_fit", plot_func=sns.boxenplot)
        generic_plot(data_manager, x="model", y="time_to_estimate", plot_func=sns.boxenplot)
