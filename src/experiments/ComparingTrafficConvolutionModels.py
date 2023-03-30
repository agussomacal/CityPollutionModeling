from functools import partial

import numpy as np
import seaborn as sns
from spiderplot import spiderplot

import src.config as config
from src.DataManager import DataManager, dmfilter
from src.LabPipeline import LabPipeline
from src.experiments.PreProcess import longer_distance, train_test_model, station_coordinates
from src.experiments.config_experiments import num_cores, shuffle
from src.lib.Models.BaseModel import Bounds, mse, UNIFORM, ModelsSequenciator, \
    ModelsAverager, LOGUNIFORM
from src.lib.Models.TrueStateEstimationModels.AverageModels import SnapshotMeanModel, GlobalMeanModel
from src.lib.Models.TrueStateEstimationModels.TrafficConvolution import TrafficMeanModel, TrafficConvolutionModel, \
    gaussker
from src.performance_utils import NamedPartial
from src.viz_utils import generic_plot


def get_traffic_conv_params(losses):
    if isinstance(losses, list):
        losses = losses[-1]  # assumes the traffic convolution is the last model
    if losses is None:
        return None, None
    losses = losses.reset_index()
    return losses["sigma"], losses["loss"]


if __name__ == "__main__":
    niter = 20
    experiment_name = f"TrafficConvolutionModelComparison_{shuffle}"

    data_manager = DataManager(
        path=config.results_dir,
        name=experiment_name
    )

    traffic_convolution = lambda: TrafficConvolutionModel(conv_kernel=gaussker, normalize=False,
                                                          sigma=Bounds(longer_distance / 10, 2 * longer_distance),
                                                          loss=mse, optim_method=LOGUNIFORM, niter=niter, verbose=True)
    traffic_convolution_norm = lambda: TrafficConvolutionModel(conv_kernel=gaussker, normalize=True,
                                                               sigma=Bounds(longer_distance / 10, 2 * longer_distance),
                                                               loss=mse, optim_method=LOGUNIFORM, niter=niter,
                                                               verbose=True)
    models = [
        SnapshotMeanModel(summary_statistic="mean"),
        GlobalMeanModel(),
        TrafficMeanModel(summary_statistic="mean"),
        traffic_convolution(),
        ModelsSequenciator(models=[
            SnapshotMeanModel(summary_statistic="mean"),
            traffic_convolution(),
        ]),
        ModelsAverager(models=[
            TrafficMeanModel(summary_statistic="mean"),
            SnapshotMeanModel(summary_statistic="mean"),
            traffic_convolution()
        ],
            positive=True, fit_intercept=False),
        traffic_convolution_norm(),
        ModelsSequenciator(models=[
            SnapshotMeanModel(summary_statistic="mean"),
            traffic_convolution_norm(),
        ]),
        ModelsAverager(models=[
            TrafficMeanModel(summary_statistic="mean"),
            SnapshotMeanModel(summary_statistic="mean"),
            traffic_convolution_norm()
        ],
            positive=True, fit_intercept=False),
    ]

    lab = LabPipeline()
    # lab.define_new_block_of_functions("true_values", loo4test)
    lab.define_new_block_of_functions("model", *list(map(train_test_model, models)))
    lab.execute(
        data_manager,
        num_cores=num_cores,
        forget=True,
        recalculate=False,
        save_on_iteration=4,
        station=station_coordinates.columns.to_list()
    )

    # ----- Plotting results ----- #
    # list(zip(*dmfilter(data_manager, names=["model", "losses"],
    #                    model=[model for model in set(data_manager["model"]) if
    #                           "TrafficConvolution" in model or "TCM" in model]).values()))
    generic_plot(data_manager, x="sigma", y="loss", label="model", plot_func=NamedPartial(sns.lineplot, marker="o"),
                 log="y",
                 sigma=lambda losses: get_traffic_conv_params(losses)[0],
                 loss=lambda losses: get_traffic_conv_params(losses)[1],
                 model=[model for model in set(data_manager["model"]) if
                        "TrafficConvolution" in model or "TCM" in model])

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
