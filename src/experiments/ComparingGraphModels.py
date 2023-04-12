from functools import partial

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from spiderplot import spiderplot

import src.config as config
from src.DataManager import DataManager, dmfilter, apply
from src.LabPipeline import LabPipeline
from src.experiments.PreProcess import longer_distance, train_test_model, station_coordinates, \
    distance_between_stations_pixels, train_test_averagers, simulation, stations2test
from src.experiments.config_experiments import num_cores, shuffle, filter_graph
from src.lib.DataProcessing.TrafficProcessing import load_background
from src.lib.Models.BaseModel import Bounds, mse, UNIFORM, ModelsSequenciator, \
    ModelsAverager, LOGUNIFORM, medianse, GRAD, Optim, CMA
from src.lib.Models.TrueStateEstimationModels.AverageModels import SnapshotMeanModel, GlobalMeanModel
from src.lib.Models.TrueStateEstimationModels.GraphModels import HEqStaticModel
from src.lib.Models.TrueStateEstimationModels.TrafficConvolution import TrafficMeanModel, TrafficConvolutionModel, \
    gaussker
from src.performance_utils import NamedPartial, if_true_str
from src.viz_utils import generic_plot, save_fig

if __name__ == "__main__":
    niter = 1000
    experiment_name = f"TrafficGraphModelComparison{if_true_str(shuffle, '_Shuffled')}" \
                      f"{if_true_str(simulation, '_Sim')}{if_true_str(filter_graph, '_Gfiltered')}"

    data_manager = DataManager(
        path=config.results_dir,
        name=experiment_name,
        country_alpha_code="NL",
        trackCO2=True
    )

    base_models = [
        SnapshotMeanModel(summary_statistic="mean"),
        GlobalMeanModel()
    ]
    models = [
        HEqStaticModel(
            # absorption=Optim(start=1, lower=1e-6, upper=2),
            # diffusion=Optim(start=1, lower=1e-6, upper=2),
            # green=Optim(1, 0, 1), yellow=Optim(1, 0, 1),
            # red=Optim(1, 0, 1), dark_red=Optim(1, 0, 1),
            absorption=Optim(start=1, lower=0, upper=np.inf),
            diffusion=Optim(start=1, lower=0, upper=np.inf),
            green=Optim(1, None, None), yellow=Optim(2, None, None),
            red=Optim(4, None, None), dark_red=Optim(8, None, None),
            name="", loss=medianse, optim_method=CMA, verbose=True,
            niter=niter,
            k_neighbours=2),
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
        recalculate=True,
        save_on_iteration=4,
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
                 model=["SnapshotMeanModelmean", "A+SMM,GMM,SMMTCMN", "GlobalMeanModelmean"]
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
