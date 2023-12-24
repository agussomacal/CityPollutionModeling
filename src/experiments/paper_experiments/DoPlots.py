from collections import OrderedDict

import numpy as np
import seaborn as sns

from PerplexityLab.DataManager import DataManager
from PerplexityLab.miscellaneous import NamedPartial
from PerplexityLab.visualization import generic_plot
from src import config
from src.experiments.paper_experiments.params4runs import path2latex_figures

data_manager = DataManager(
    path=config.paper_experiments_dir,
    emissions_path=config.results_dir,
    name="NumericalResults",
    country_alpha_code="FR",
    trackCO2=True
)
data_manager.load()

model_names = OrderedDict([
    ("BLUEModelBLUE", "BLUE"),
    ("ExponentialKernelModel", "Krigging"),
    ("AvgKrigging", "Global Average \n Krigging"),
    ("SnapshotMeanModelmean", "Average in space"),
    ("LR", "Graph \n Linear Regression"),
    ("LR_Extra", "Graph Temp Wind \n Linear Regression"),
    ("NN_Extra", "Graph Temp Wind \n NeuralNetwork"),
    ("Pipeline(steps=[('LR', LassoCV(selection='random'))])(SMM,EKM,AK,LR,LRE,NNE)", "Ensemble"),
    ("Pipeline(steps=[('LR', LassoCV(selection='random'))])(NNE)", "Ensemble NN")
])

models_order = list(model_names.values()) + ["Ensemble"]
# models_order.remove("Krigging")
models2plot = set(data_manager["model"])
# models2plot.remove("ExponentialKernelModel")
models2plot = list(models2plot)


# runsinfo.append_info(
#     average=model_names["SnapshotMeanModelmean"].replace(" \n", ""),
#     krigging=model_names["ExponentialKernelModel"].replace(" \n", ""),
#     avgkrigging=model_names["AvgKrigging"].replace(" \n", ""),
#     lm=model_names["LR"].replace(" \n", ""),
#     lmextra=model_names["LR_Extra"].replace(" \n", ""),
#     nn=model_names["NN_Extra"].replace(" \n", ""),
#     ensemble="Ensemble",
# )


def name_models(model_name):
    # if "Pipeline" in model_name:
    #     return "Ensemble"
    # else:
    return model_names[model_name]


# generic_plot(
#     data_manager=data_manager,
#     # folder=path2latex_figures,
#     # x="models",
#     x="model_name",
#     y="mse",
#     plot_func=NamedPartial(sns.barplot, orient="vertical",
#                            # order=models_order
#                            # , errorbar=("ci", 0)
#                            ),
#     # plot_func=sns.barplot,
#     # sort_by=["models"],
#     mse=lambda error: np.NAN if error is None else np.sqrt(error.mean()),
#     # models=name_models,
#     # model=models2plot,
#     dpi=300,
#     axes_xy_proportions=(12, 8),
#     # format=".pdf"
# )

stations_order = ["OPERA", "BASCH", "PA13", "PA07", "PA18", "ELYS", "PA12", ] + ["BONAP", "HAUS", "CELES"]
generic_plot(
    name="ErrorPLot",
    data_manager=data_manager,
    # folder=path2latex_figures,
    y="station", x="mse", label="models",
    plot_func=NamedPartial(sns.lineplot,
                           orient="y",
                           hue_order=models_order,
                           # order=stations_order
                           sort=True,
                           markers=True, dashes=True,
                           marker="o", linestyle="--"
                           ),
    sort_by=["models"],
    mse=lambda error: np.NAN if error is None else np.sqrt(error.mean()),
    models=name_models,
    model=models2plot,
    station=stations_order,
    xlim=(None, 20),
    dpi=300,
    # format=".pdf"
)
