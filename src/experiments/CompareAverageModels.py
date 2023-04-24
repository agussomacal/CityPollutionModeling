import numpy as np
import seaborn as sns
from spiderplot import spiderplot

import src.config as config
from PerplexityLab.DataManager import DataManager
from PerplexityLab.LabPipeline import LabPipeline
from src.experiments.PreProcess import loo4test, train_test_model, station_coordinates
from src.experiments.config_experiments import num_cores
from src.lib.Models.TrueStateEstimationModels.AverageModels import SnapshotMeanModel, GlobalMeanModel, \
    SnapshotWeightedModel, SnapshotWeightedStd
from PerplexityLab.miscellaneous import NamedPartial
from PerplexityLab.visualization import generic_plot

if __name__ == "__main__":
    experiment_name = "AverageModels"

    data_manager = DataManager(
        path=config.results_dir,
        name=experiment_name
    )

    models = [
        GlobalMeanModel(),
        SnapshotMeanModel(summary_statistic="mean"),
        SnapshotMeanModel(summary_statistic="median"),
        SnapshotWeightedStd(),
        SnapshotWeightedModel(positive=True, fit_intercept=True)
        # SnapshotWeightedModel(positive=False, fit_intercept=True),
        # ModelsSequenciator(models=[GlobalMeanModel(),
        #                            SnapshotWeightedModel(positive=True, fit_intercept=False)],
        #                    name="Global_SnapW"),
        # TrafficMeanModel(summary_statistic="mean"),
        # TrafficMeanModel(summary_statistic="median"),
        # ModelsSequenciator(models=[SnapshotMeanModel(summary_statistic="mean"),
        #                            TrafficMeanModel(summary_statistic="mean")],
        #                    name="SnapshotPollutionTraffic")
    ]

    lab = LabPipeline()
    # lab.define_new_block_of_functions("true_values", loo4test)
    lab.define_new_block_of_functions("model", *list(map(train_test_model, models)))
    lab.execute(
        data_manager,
        num_cores=num_cores,
        forget=True,
        recalculate=False,
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
                 mse=lambda error: np.sqrt(error.mean()))

    generic_plot(data_manager, x="mse", y="model", plot_func=NamedPartial(sns.boxenplot, orient="horizontal"),
                 sort_by=["mse"],
                 mse=lambda error: np.sqrt(np.nanmean(error)))

    generic_plot(data_manager, x="model", y="time_to_fit", plot_func=sns.boxenplot)
    generic_plot(data_manager, x="model", y="time_to_estimate", plot_func=sns.boxenplot)
