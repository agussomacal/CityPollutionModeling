from functools import partial

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from spiderplot import spiderplot

import src.config as config
from PerplexityLab.DataManager import DataManager, dmfilter, apply
from PerplexityLab.LabPipeline import LabPipeline
from PerplexityLab.miscellaneous import NamedPartial, if_true_str
from PerplexityLab.visualization import generic_plot, save_fig
from src.experiments.PreProcess import longer_distance, train_test_model, distance_between_stations_pixels, \
    train_test_averagers, simulation, stations2test
from src.experiments.config_experiments import num_cores, shuffle
from src.lib.DataProcessing.TrafficProcessing import load_background
from src.lib.Models.BaseModel import ModelsSequenciator, \
    LOGUNIFORM, medianse
from src.lib.Modules import Bounds
from src.lib.Models.TrueStateEstimationModels.AverageModels import SnapshotMeanModel, GlobalMeanModel
from src.lib.Models.TrueStateEstimationModels.TrafficConvolution import TrafficMeanModel, TrafficConvolutionModel, \
    gaussker


def get_traffic_conv_params(losses):
    if isinstance(losses, list):
        losses = losses[-1]  # assumes the traffic convolution is the last model
    if isinstance(losses, list):
        losses = losses[-1]  # assumes the traffic convolution is the last model
    if losses is None:
        return None, None
    losses = losses.reset_index()
    return losses["sigma"], losses["loss"]


def plot_kernel_in_map(data_manager, screenshot_period, station="OPERA", model="TrafficConvolutionModelgaussker"):
    plt.close("all")
    with save_fig(data_manager.path, "CharacteristicTrafficConvDistance.png"):
        plt.imshow(load_background(screenshot_period))
        d = apply(dmfilter(data_manager, names=["losses"], model=[model]), names=["losses"],
                  sigma=lambda losses: get_traffic_conv_params(losses)[0],
                  loss=lambda losses: get_traffic_conv_params(losses)[1])
        sigma_optim = pd.concat(d["sigma"], axis=1).mean(axis=1).values[
            np.argmin(pd.concat(d["loss"], axis=1).mean(axis=1))]
        ker = gaussker(distance_between_stations_pixels.loc[station, :], sigma=sigma_optim)

        plt.scatter(*list(zip(*ker.index))[::-1], s=1, c=ker.values, marker=".", cmap="jet")
        plt.tight_layout()


if __name__ == "__main__":
    niter = 25
    experiment_name = f"TrafficConvolutionModelComparison{if_true_str(shuffle, '_Shuffled')}{if_true_str(simulation, '_Sim')}"

    data_manager = DataManager(
        path=config.results_dir,
        name=experiment_name
    )

    traffic_convolution = lambda: TrafficConvolutionModel(conv_kernel=gaussker, normalize=False,
                                                          sigma=Bounds(longer_distance.ravel()[0] / 10,
                                                                       2 * longer_distance),
                                                          loss=medianse, optim_method=LOGUNIFORM, niter=niter,
                                                          verbose=True)
    traffic_convolution_norm = lambda: TrafficConvolutionModel(conv_kernel=gaussker, normalize=True,
                                                               sigma=Bounds(longer_distance.ravel()[0] / 10,
                                                                            2 * longer_distance),
                                                               loss=medianse, optim_method=LOGUNIFORM, niter=niter,
                                                               verbose=True)
    base_models = [
        SnapshotMeanModel(summary_statistic="mean"),
        GlobalMeanModel()
    ]
    models = [
        TrafficMeanModel(summary_statistic="mean"),
        traffic_convolution(),
        ModelsSequenciator(models=[
            SnapshotMeanModel(summary_statistic="mean"),
            traffic_convolution(),
        ]),
        traffic_convolution_norm(),
        ModelsSequenciator(models=[
            SnapshotMeanModel(summary_statistic="mean"),
            traffic_convolution_norm(),
        ]),
    ]

    lab = LabPipeline()
    lab.define_new_block_of_functions("train_individual_models", *list(map(train_test_model, base_models + models)))
    lab.define_new_block_of_functions("model",
                                      *list(map(partial(train_test_averagers, positive=True, fit_intercept=False),
                                                [[model] for model in base_models + models] +
                                                [base_models + [model] for model in models]
                                                )))

    lab.execute(
        data_manager,
        num_cores=num_cores,
        forget=False,
        recalculate=False,
        save_on_iteration=4,
        station=stations2test  # station_coordinates.columns.to_list()[:2]
    )

    # ----- Plotting results ----- #
    generic_plot(data_manager, x="sigma", y="loss", label="model", plot_func=NamedPartial(sns.lineplot, marker="o"),
                 log="y",
                 sigma=lambda losses: get_traffic_conv_params(losses)[0],
                 loss=lambda losses: get_traffic_conv_params(losses)[1],
                 model=[model for model in set(data_manager["model"]) if
                        "TrafficConvolution" in model or "TCM" in model]
                 )
    # plot_kernel_in_map(data_manager, 15, station="OPERA", model="TrafficConvolutionModelgaussker")
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
                 model=["SnapshotMeanModelmean", "A+SMM,GMM,SMMTCMN", "GlobalMeanModelmean"]
                 )

    # generic_plot(data_manager, x="station", y="mse", label="model", plot_func=spiderplot,
    #              mse=lambda estimation, future_pollution:
    #              np.sqrt(((estimation - future_pollution.values.ravel()).ravel() ** 2).mean()))
    #
    # generic_plot(data_manager, x="station", y="error", label="model", plot_func=sns.boxenplot,
    #              error=lambda estimation, future_pollution:
    #              np.abs((estimation - future_pollution.values.ravel()).ravel()))

    generic_plot(data_manager, x="model", y="time_to_fit", plot_func=sns.boxenplot)
    generic_plot(data_manager, x="model", y="time_to_estimate", plot_func=sns.boxenplot)

    generic_plot(data_manager, x="station", y="mse", label="model", plot_func=spiderplot,
                 mse=lambda error: np.sqrt(error.mean()))

    generic_plot(data_manager, x="station", y="l1_error", label="model", plot_func=sns.boxenplot,
                 sort_by=["mse"],
                 l1_error=lambda error: np.abs(error).ravel(),
                 mse=lambda error: np.sqrt(error.mean()),
                 ylim=(0, 100))
