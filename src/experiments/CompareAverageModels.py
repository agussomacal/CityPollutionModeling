from functools import partial

import numpy as np
import seaborn as sns
from sklearn.linear_model import LassoCV
from sklearn.pipeline import Pipeline
from spiderplot import spiderplot

import src.config as config
from PerplexityLab.DataManager import DataManager
from PerplexityLab.LabPipeline import LabPipeline
from PerplexityLab.miscellaneous import NamedPartial, copy_main_script_version
from PerplexityLab.visualization import generic_plot
from src.experiments.PreProcess import train_test_model, station_coordinates, train_test_averagers
from src.experiments.config_experiments import num_cores, stations2test
from src.lib.Models.BaseModel import Bounds, UNIFORM, Optim, NONE_OPTIM_METHOD, GRAD
from src.lib.Models.TrueStateEstimationModels.AverageModels import SnapshotMeanModel, GlobalMeanModel, SnapshotPCAModel, \
    SnapshotBLUEModel, SnapshotQuantileModel

if __name__ == "__main__":
    experiment_name = "AverageModels"

    data_manager = DataManager(
        path=config.results_dir,
        name=experiment_name,
        country_alpha_code="FR",
        trackCO2=True
    )
    copy_main_script_version(__file__, data_manager.path)

    models = [
        SnapshotBLUEModel(sensor_distrust=Optim(start=0, lower=0, upper=1), optim_method=GRAD, niter=100,
                          verbose=False),
        # GlobalMeanModel(),
        SnapshotMeanModel(summary_statistic="mean"),
        SnapshotPCAModel(n_components=Optim(start=1, lower=1, upper=7), niter=7, summary_statistic="mean",
                         optim_method=UNIFORM),
        SnapshotQuantileModel()
    ]

    lab = LabPipeline()
    lab.define_new_block_of_functions("train_individual_models", *list(map(train_test_model,
                                                                           models
                                                                           )))
    lab.define_new_block_of_functions("model",
                                      *list(map(partial(train_test_averagers,
                                                        aggregator=Pipeline([("LR", LassoCV(selection="random"))])),
                                                [[model] for model in models]
                                                )))

    lab.execute(
        data_manager,
        num_cores=10,
        forget=False,
        recalculate=False,
        save_on_iteration=1,
        station=stations2test  # station_coordinates.columns.to_list()[:2]
    )

    # ----- Plotting results ----- #
    generic_plot(data_manager, x="station", y="mse", label="model", plot_func=sns.barplot,
                 sort_by=["station"],
                 mse=lambda error: np.sqrt(error.mean()),
                 station=stations2test,
                 )

    generic_plot(data_manager, x="l1_error", y="model", plot_func=NamedPartial(sns.boxenplot, orient="horizontal"),
                 sort_by=["mse"],
                 l1_error=lambda error: np.abs(error).ravel(),
                 mse=lambda error: np.sqrt(error.mean()),
                 station=stations2test,
                 # xlim=(0, 20)
                 )

    generic_plot(data_manager, x="mse", y="model",
                 plot_func=NamedPartial(sns.violinplot, orient="horizontal", inner="stick"),
                 sort_by=["mse"],
                 mse=lambda error: np.sqrt(np.nanmean(error)),
                 # xlim=(0, 100),
                 # model=["SnapshotMeanModelmean", "A+SMM,GMM,SMMTCMN", "GlobalMeanModelmean"]
                 )

    generic_plot(data_manager, x="model", y="time_to_fit", plot_func=sns.boxenplot)
    generic_plot(data_manager, x="model", y="time_to_estimate", plot_func=sns.boxenplot)

    generic_plot(data_manager, x="station", y="mse", label="model", plot_func=spiderplot,
                 mse=lambda error: np.sqrt(error.mean()))

    generic_plot(data_manager, x="station", y="l1_error", label="model", plot_func=sns.boxenplot,
                 sort_by=["mse"],
                 l1_error=lambda error: np.abs(error).ravel(),
                 mse=lambda error: np.sqrt(error.mean()),
                 )
    # # ----- Plotting results ----- #
    # generic_plot(data_manager, x="station", y="mse", label="model", plot_func=spiderplot,
    #              mse=lambda error: np.sqrt(error.mean()))
    #
    # generic_plot(data_manager, x="station", y="l1_error", label="model", plot_func=sns.boxenplot,
    #              sort_by=["mse"],
    #              l1_error=lambda error: np.abs(error).ravel(),
    #              mse=lambda error: np.sqrt(error.mean()))
    #
    # generic_plot(data_manager, x="l1_error", y="model", plot_func=NamedPartial(sns.boxenplot, orient="horizontal"),
    #              sort_by=["mse"],
    #              l1_error=lambda error: np.abs(error).ravel(),
    #              mse=lambda error: np.sqrt(error.mean()))
    #
    # generic_plot(data_manager, x="mse", y="model", plot_func=NamedPartial(sns.boxenplot, orient="horizontal"),
    #              sort_by=["mse"],
    #              mse=lambda error: np.sqrt(np.nanmean(error)))
    #
    # generic_plot(data_manager, x="model", y="time_to_fit", plot_func=sns.boxenplot)
    # generic_plot(data_manager, x="model", y="time_to_estimate", plot_func=sns.boxenplot)
