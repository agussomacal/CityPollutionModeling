from collections import OrderedDict, namedtuple

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from PerplexityLab.miscellaneous import NamedPartial
from PerplexityLab.visualization import generic_plot, LegendOutsidePlot, save_fig
from src.experiments.paper_experiments.PreProcessPaper import graph
from src.experiments.paper_experiments.SourceModels import data_manager
from src.experiments.paper_experiments.params4runs import screenshot_period
from src.lib.DataProcessing.TrafficProcessing import load_background
from src.lib.FeatureExtractors.GraphFeatureExtractors import get_graph_node_positions
from src.lib.Models.TrueStateEstimationModels.PhysicsModel import get_diffusion_matrix, get_absorption_matrix, \
    get_geometric_basis
from src.lib.visualization_tools import FillBetweenInfo, plot_errors, plot_estimation_map_in_graph

PlotStyle = namedtuple("PlotStyle", "color marker linestyle linewidth size", defaults=["black", "o", "--", None, None])
cblue, corange, cgreen, cred, cpurple, cbrown, cpink, cgray, cyellow, ccyan = sns.color_palette("tab10")
cblack = (0, 0, 0)
cwhite = (1, 1, 1)

# train_with_relative_error = False
# data_manager = DataManager(
#     path=config.paper_experiments_dir,
#     emissions_path=config.results_dir,
#     name="SourceModelsRelErr" if train_with_relative_error else "SourceModels",
#     country_alpha_code="FR",
#     trackCO2=True
# )
data_manager.load()

OptimModel = "BLUE"
BaselineModel = "Spatial Avg"  # "Average in space"
Krigging = "Kernel"  # "Krigging"

# model_names = OrderedDict([
#     ("Spatial Avg", BaselineModel),
#     # ("SL Spatial Avg", "SL " + BaselineModel),
#     ("BLUE", OptimModel),
#
#     ("Kernel", Krigging),
#
#     ("SourceModel_Poly2Lasso+", "SourceModel_Poly2Lasso+"),
#     ("SourceModel_Poly2Lasso_avg+", "SourceModel_Poly2Lasso_avg+"),
# ])

model_names = list(set(data_manager["individual_models"]))
model_names = dict(zip(model_names, model_names))
modelstyle_basics = OrderedDict([
    ("Spatial Avg", PlotStyle(color=cred, marker=None, linestyle=":")),
    ("BLUE", PlotStyle(color=cblue, marker=None, linestyle=":")),
    # ("ExponentialFit", PlotStyle(color=cyellow, marker="<", linestyle=":", linewidth=2)),
])

# model_style = OrderedDict([
#     ("Spatial Avg", PlotStyle(color=cred, marker=None, linestyle=":")),
#     # ("SL Spatial Avg", PlotStyle(color=cred, marker="*", linestyle=":")),
#
#     ("BLUE", PlotStyle(color=cblue, marker=None, linestyle=":")),
#     # ("BLUE_DU", PlotStyle(color=cblue, marker=None, linestyle="--", linewidth=2)),
#     # ("BLUE_DI", PlotStyle(color=cblue, marker="o", linestyle="-", linewidth=1)),
#     # ("Kernel", PlotStyle(color=cyellow, marker=None, linestyle=":")),
#     # ("Gaussian", PlotStyle(color=cyellow, marker="*", linestyle="--", linewidth=2)),
#     # ("Exponential", PlotStyle(color=cyellow, marker="o", linestyle="--", linewidth=3)),
#     # ("ExponentialD", PlotStyle(color=cred, marker=".", linestyle="--", linewidth=2)),
#     ("ExponentialFit", PlotStyle(color=cyellow, marker="o", linestyle="-", linewidth=2)),
#     # ("ExponentialOld", PlotStyle(color=cyellow, marker="o", linestyle="-", linewidth=5)),
#
#     # ("SourceModel_Poly2Lasso+", PlotStyle(color=cgreen, marker=".", linestyle="--")),
#     # ("SourceModel_Poly2Lasso_avg", PlotStyle(color=cgreen, marker=".", linestyle=":")),
#     # ("SourceModel_Poly1Lasso_avg", PlotStyle(color=corange, marker="o", linestyle=":")),
#     # ("SourceModel_Poly1Lasso_avg_TWHW", PlotStyle(color=cgreen, marker="o", linestyle=":")),
#     # ("SourceModel_NN_avg_TWHW", PlotStyle(color=cblue, marker="o", linestyle=":")),
#     # ("SourceModel_RF_avg_TWHW", PlotStyle(color=cred, marker="o", linestyle=":")),
#
#     # ("PCAAfterSourceModel_LM_TWHW", PlotStyle(color=cblack, marker="o", linestyle=":")),
#     # ("LapAfterSourceModel_LM_TWHW", PlotStyle(color=cblack, marker="o", linestyle="-.")),
#     #
#     # ("PCASourceModel_Poly1Lasso_avg_TWHW", PlotStyle(color=cred, marker="o", linestyle="-")),
#     #
#     # ("PCAAfterSourceModel_LM_TWHW", PlotStyle(color=corange, marker="o", linestyle="-")),
#     # ("PCA_ProjectionFullSourceModel_LM_TWHW", PlotStyle(color=cblue, marker="o", linestyle="-")),
#
#     ("node_linear_TWHW", PlotStyle(color=cgreen, marker=".", linestyle="--", linewidth=2)),
#     # ("node_poly2_TWHW", PlotStyle(color=cpurple, marker=".", linestyle=":", linewidth=2)),
#     # ("node_nn_TWHW", PlotStyle(color=cblue, marker="o", linestyle="-", linewidth=2)),
#     # ("node_RF_TWHW", PlotStyle(color=cred, marker="*", linestyle="-.", linewidth=2)),
#
#     # ("geometrical_linear_TWHW", PlotStyle(color=cgreen, marker=".", linestyle="--", linewidth=2)),
#     # ("geometrical_poly2_TWHW", PlotStyle(color=cpurple, marker=".", linestyle=":", linewidth=2)),
#     ("geometrical_nn_TWHW", PlotStyle(color=cblue, marker="o", linestyle="-", linewidth=2)),
#     ("geometrical_RF_TWHW", PlotStyle(color=cred, marker="*", linestyle="-.", linewidth=2)),
#
#     # ("pca_linear_TWHW", PlotStyle(color=cgreen, marker=".", linestyle="--", linewidth=2)),
#     # ("pca_poly2_TWHW", PlotStyle(color=cpurple, marker=".", linestyle=":", linewidth=2)),
#     # ("pca_nn_TWHW", PlotStyle(color=cblue, marker="o", linestyle="-", linewidth=2)),
#     # ("pca_RF_TWHW", PlotStyle(color=cred, marker="*", linestyle="-.", linewidth=2)),
#
#     # ("both_linear_TWHW", PlotStyle(color=cred, marker="o", linestyle="--", linewidth=2)),
#     # ("both_poly2_TWHW", PlotStyle(color=cred, marker="o", linestyle="-", linewidth=2)),
#     # ("both_nn_TWHW", PlotStyle(color=cred, marker="o", linestyle="-", linewidth=2)),
#     # ("both_RF_TWHW", PlotStyle(color=cred, marker="o", linestyle="-.", linewidth=2)),
#
#     # ("EnsembleAvg", PlotStyle(color=cblack, marker="o", linestyle="-", linewidth=3)),
#     # ("EnsembleAvgNoCV", PlotStyle(color=cblack, marker=".", linestyle="--", linewidth=3)),
#     # ("EnsembleAvgNoCV_RF", PlotStyle(color=cblack, marker=".", linestyle="-.", linewidth=3)),
#     # ("EnsembleAvgNoCV_nn", PlotStyle(color=cblack, marker=".", linestyle="-", linewidth=3)),
#     # ("EnsembleAvgNoCV_average", PlotStyle(color=cblack, marker=".", linestyle="--", linewidth=3)),
#     # ("EnsembleAvgNoCV_std", PlotStyle(color=cblack, marker=".", linestyle="-.", linewidth=3)),
#     # ("EnsembleAvgNoCV_cv", PlotStyle(color=cblack, marker=".", linestyle="-", linewidth=3)),
#     ("EnsembleAvgNoCV_Lasso", PlotStyle(color=cblack, marker=".", linestyle=":", linewidth=3)),
#     # ("EnsembleAvgNoCV_weighted_average", PlotStyle(color=cblack, marker=".", linestyle="--", linewidth=3)),
#
#     # ("DistanceRational", PlotStyle(color=cpink, marker="o", linestyle="-", linewidth=3)),
#
#     # # ("LaplacianSourceModel_Poly1Lasso_avg_TW", PlotStyle(color=cpurple, marker="o", linestyle="-")),
#     # ("PCASourceModel_Poly1Lasso_avg_TWHW", PlotStyle(color=cgreen, marker="o", linestyle="-")),
#     # ("PCASourceModel_Poly1NN_avg_TWHW", PlotStyle(color=cblue, marker="o", linestyle="-")),
#     # ("PCASourceModel_Poly1RF_avg_TWHW", PlotStyle(color=cred, marker="o", linestyle="-")),
#     # # ("LaplacianSourceModel_Poly1Lasso_avg_TWHW", PlotStyle(color=cpurple, marker="o", linestyle="-")),
#     #
#     # ("V2LaplacianSourceModel_Poly1Lasso_avg_TWHW", PlotStyle(color=cgreen, marker="*", linestyle="--", linewidth=2)),
#     # ("V2LaplacianSourceModel_NN_avg_TWHW", PlotStyle(color=cblue, marker="*", linestyle="--", linewidth=2)),
#     # ("V2LaplacianSourceModel_RF_avg_TWHW", PlotStyle(color=cred, marker="*", linestyle="--", linewidth=2)),
#
#     # ("PCASourceModel_Poly1Lasso_avg_TWHW_nei+", PlotStyle(color=ccyan, marker="o", linestyle="-")),
#     # ("PhysicsModel", PlotStyle(color=cgray, marker="*", linestyle="--", linewidth=2)),
#     # ("Ensemble", PlotStyle(color=cblack, marker="*", linestyle="-", linewidth=3)),
#     # ("Ensemble2", PlotStyle(color=cblack, marker="o", linestyle=":", linewidth=3)),
#     # ("EnsembleKernel", PlotStyle(color=cblack, marker="*", linestyle="--", linewidth=3)),
#     # ("EnsembleKernelAvg", PlotStyle(color=cblack, marker="*", linestyle="--", linewidth=3)),
#     # ("EnsembleKernelBLUE", PlotStyle(color=cblack, marker="o", linestyle="-", linewidth=3)),
#     # ("EnsembleKernelMSE", PlotStyle(color=cblack, marker="o", linestyle="-", linewidth=3)),
#     # ("EnsembleBLUE", PlotStyle(color=cpurple, marker="o", linestyle="-", linewidth=3)),
#     # ("EnsembleAvg", PlotStyle(color=cblack, marker="o", linestyle="-", linewidth=3)),
#
#     # ("SoftDiffusion", PlotStyle(color=cblue, marker="o", linestyle="-", linewidth=3)),
#
#     # ("SourceModel_Poly1Lasso_avg_TWGW", PlotStyle(color=cred, marker="o", linestyle="-")),
#     # ("SourceModel_Poly2Lasso_avg_TW", PlotStyle(color=cgreen, marker="o", linestyle="-")),
#
# ])


groups = [
    # ---------- No regressors ---------- #
    # ("LinearModels", [
    #     ("node_linear_", PlotStyle(color=cgreen, marker=".", linestyle="--", linewidth=2)),
    #     ("geometrical_linear_", PlotStyle(color=cpurple, marker=".", linestyle="--", linewidth=2)),
    #     ("pca_linear_", PlotStyle(color=cblue, marker=".", linestyle="--", linewidth=2)),
    #     ("both_linear_", PlotStyle(color=cred, marker="o", linestyle="--", linewidth=2)),
    # ]),
    # ("QuadraticModels", [
    #     ("node_poly2_", PlotStyle(color=cgreen, marker=".", linestyle="--", linewidth=2)),
    #     ("geometrical_poly2_", PlotStyle(color=cpurple, marker=".", linestyle="--", linewidth=2)),
    #     ("pca_poly2_", PlotStyle(color=cblue, marker=".", linestyle="--", linewidth=2)),
    #     ("both_poly2_", PlotStyle(color=cred, marker="o", linestyle="--", linewidth=2)),
    # ]),
    # ("RFModels", [
    #     ("node_RF_", PlotStyle(color=cgreen, marker=".", linestyle="--", linewidth=2)),
    #     ("geometrical_RF_", PlotStyle(color=cpurple, marker=".", linestyle="--", linewidth=2)),
    #     ("pca_RF_", PlotStyle(color=cblue, marker=".", linestyle="--", linewidth=2)),
    #     ("both_RF_", PlotStyle(color=cred, marker="o", linestyle="--", linewidth=2)),
    # ]),
    # ("NNModels", [
    #     ("node_nn_", PlotStyle(color=cgreen, marker=".", linestyle="--", linewidth=2)),
    #     ("geometrical_nn_", PlotStyle(color=cpurple, marker=".", linestyle="--", linewidth=2)),
    #     ("pca_nn_", PlotStyle(color=cblue, marker=".", linestyle="--", linewidth=2)),
    #     ("both_nn_", PlotStyle(color=cred, marker="o", linestyle="--", linewidth=2)),
    # ]),
    # ("Poly2NNModels", [
    #     ("node_poly2NN_", PlotStyle(color=cgreen, marker=".", linestyle="--", linewidth=2)),
    #     ("geometrical_poly2NN_", PlotStyle(color=cpurple, marker=".", linestyle="--", linewidth=2)),
    #     ("pca_poly2NN_", PlotStyle(color=cblue, marker=".", linestyle="--", linewidth=2)),
    #     ("both_poly2NN_", PlotStyle(color=cred, marker="o", linestyle="--", linewidth=2)),
    # ]),
    # ("Poly3", [
    #     ("node_poly3_", PlotStyle(color=cgreen, marker=".", linestyle="--", linewidth=2)),
    #     ("geometrical_poly3_", PlotStyle(color=cpurple, marker=".", linestyle="--", linewidth=2)),
    #     ("pca_poly3_", PlotStyle(color=cblue, marker=".", linestyle="--", linewidth=2)),
    #     ("both_poly3_", PlotStyle(color=cred, marker="o", linestyle="--", linewidth=2)),
    # ]),
    # # ---------- TWAAA regressors ---------- #
    # ("LinearModels_TWAAA", [
    #     ("node_linear_TWAAA", PlotStyle(color=cgreen, marker=".", linestyle="--", linewidth=2)),
    #     ("geometrical_linear_TWAAA", PlotStyle(color=cpurple, marker=".", linestyle="--", linewidth=2)),
    #     ("pca_linear_TWAAA", PlotStyle(color=cblue, marker=".", linestyle="--", linewidth=2)),
    #     ("both_linear_TWAAA", PlotStyle(color=cred, marker="o", linestyle="--", linewidth=2)),
    # ]),
    # ("QuadraticModels_TWAAA", [
    #     ("node_poly2_TWAAA", PlotStyle(color=cgreen, marker=".", linestyle="--", linewidth=2)),
    #     ("geometrical_poly2_TWAAA", PlotStyle(color=cpurple, marker=".", linestyle="--", linewidth=2)),
    #     ("pca_poly2_TWAAA", PlotStyle(color=cblue, marker=".", linestyle="--", linewidth=2)),
    #     ("both_poly2_TWAAA", PlotStyle(color=cred, marker="o", linestyle="--", linewidth=2)),
    # ]),
    # ("RFModels_TWAAA", [
    #     ("node_RF_TWAAA", PlotStyle(color=cgreen, marker=".", linestyle="--", linewidth=2)),
    #     ("geometrical_RF_TWAAA", PlotStyle(color=cpurple, marker=".", linestyle="--", linewidth=2)),
    #     ("pca_RF_TWAAA", PlotStyle(color=cblue, marker=".", linestyle="--", linewidth=2)),
    #     ("both_RF_TWAAA", PlotStyle(color=cred, marker="o", linestyle="--", linewidth=2)),
    # ]),
    # ("NNModels_TWAAA", [
    #     ("node_nn_TWAAA", PlotStyle(color=cgreen, marker=".", linestyle="--", linewidth=2)),
    #     ("geometrical_nn_TWAAA", PlotStyle(color=cpurple, marker=".", linestyle="--", linewidth=2)),
    #     ("pca_nn_TWAAA", PlotStyle(color=cblue, marker=".", linestyle="--", linewidth=2)),
    #     ("both_nn_TWAAA", PlotStyle(color=cred, marker="o", linestyle="--", linewidth=2)),
    # ]),
    # ("Poly2NNModels_TWAAA", [
    #     ("node_poly2NN_TWAAA", PlotStyle(color=cgreen, marker=".", linestyle="--", linewidth=2)),
    #     ("geometrical_poly2NN_TWAAA", PlotStyle(color=cpurple, marker=".", linestyle="--", linewidth=2)),
    #     ("pca_poly2NN_TWAAA", PlotStyle(color=cblue, marker=".", linestyle="--", linewidth=2)),
    #     ("both_poly2NN_TWAAA", PlotStyle(color=cred, marker="o", linestyle="--", linewidth=2)),
    # ]),
    # ("Poly3_TWAAA", [
    #     ("node_poly3_TWAAA", PlotStyle(color=cgreen, marker=".", linestyle="--", linewidth=2)),
    #     ("geometrical_poly3_TWAAA", PlotStyle(color=cpurple, marker=".", linestyle="--", linewidth=2)),
    #     ("pca_poly3_TWAAA", PlotStyle(color=cblue, marker=".", linestyle="--", linewidth=2)),
    #     ("both_poly3_TWAAA", PlotStyle(color=cred, marker="o", linestyle="--", linewidth=2)),
    # ]),
    # # ---------- TWAA regressors ---------- #
    # ("LinearModels_TWAA", [
    #     ("node_linear_TWAA", PlotStyle(color=cgreen, marker=".", linestyle="--", linewidth=2)),
    #     ("geometrical_linear_TWAA", PlotStyle(color=cpurple, marker=".", linestyle="--", linewidth=2)),
    #     ("pca_linear_TWAA", PlotStyle(color=cblue, marker=".", linestyle="--", linewidth=2)),
    #     ("both_linear_TWAA", PlotStyle(color=cred, marker="o", linestyle="--", linewidth=2)),
    # ]),
    # ("QuadraticModels_TWAA", [
    #     ("node_poly2_TWAA", PlotStyle(color=cgreen, marker=".", linestyle="--", linewidth=2)),
    #     ("geometrical_poly2_TWAA", PlotStyle(color=cpurple, marker=".", linestyle="--", linewidth=2)),
    #     ("pca_poly2_TWAA", PlotStyle(color=cblue, marker=".", linestyle="--", linewidth=2)),
    #     ("both_poly2_TWAA", PlotStyle(color=cred, marker="o", linestyle="--", linewidth=2)),
    # ]),
    # ("RFModels_TWAA", [
    #     ("node_RF_TWAA", PlotStyle(color=cgreen, marker=".", linestyle="--", linewidth=2)),
    #     ("geometrical_RF_TWAA", PlotStyle(color=cpurple, marker=".", linestyle="--", linewidth=2)),
    #     ("pca_RF_TWAA", PlotStyle(color=cblue, marker=".", linestyle="--", linewidth=2)),
    #     ("both_RF_TWAA", PlotStyle(color=cred, marker="o", linestyle="--", linewidth=2)),
    # ]),
    # ("NNModels_TWAA", [
    #     ("node_nn_TWAA", PlotStyle(color=cgreen, marker=".", linestyle="--", linewidth=2)),
    #     ("geometrical_nn_TWAA", PlotStyle(color=cpurple, marker=".", linestyle="--", linewidth=2)),
    #     ("pca_nn_TWAA", PlotStyle(color=cblue, marker=".", linestyle="--", linewidth=2)),
    #     ("both_nn_TWAA", PlotStyle(color=cred, marker="o", linestyle="--", linewidth=2)),
    # ]),
    # ("Poly2NNModels_TWAA", [
    #     ("node_poly2NN_TWAA", PlotStyle(color=cgreen, marker=".", linestyle="--", linewidth=2)),
    #     ("geometrical_poly2NN_TWAA", PlotStyle(color=cpurple, marker=".", linestyle="--", linewidth=2)),
    #     ("pca_poly2NN_TWAA", PlotStyle(color=cblue, marker=".", linestyle="--", linewidth=2)),
    #     ("both_poly2NN_TWAA", PlotStyle(color=cred, marker="o", linestyle="--", linewidth=2)),
    # ]),
    # ("Poly3_TWAA", [
    #     ("node_poly3_TWAA", PlotStyle(color=cgreen, marker=".", linestyle="--", linewidth=2)),
    #     ("geometrical_poly3_TWAA", PlotStyle(color=cpurple, marker=".", linestyle="--", linewidth=2)),
    #     ("pca_poly3_TWAA", PlotStyle(color=cblue, marker=".", linestyle="--", linewidth=2)),
    #     ("both_poly3_TWAA", PlotStyle(color=cred, marker="o", linestyle="--", linewidth=2)),
    # ]),
    # ---------- TW regressors ---------- #
    # ("LinearModels_TW", [
    #     ("node_linear_TW", PlotStyle(color=cgreen, marker=".", linestyle="--", linewidth=2)),
    #     ("geometrical_linear_TW", PlotStyle(color=cpurple, marker=".", linestyle="--", linewidth=2)),
    #     ("pca_linear_TW", PlotStyle(color=cblue, marker=".", linestyle="--", linewidth=2)),
    #     ("geometrical_log_linear_TW", PlotStyle(color=cred, marker=".", linestyle="-", linewidth=2)),
    #     ("pca_log_linear_TW", PlotStyle(color=cblack, marker=".", linestyle="-", linewidth=2)),
    # ]),
    # ("QuadraticModels_TW", [
    #     ("node_poly2_TW", PlotStyle(color=cgreen, marker=".", linestyle="--", linewidth=2)),
    #     ("geometrical_poly2_TW", PlotStyle(color=cpurple, marker=".", linestyle="--", linewidth=2)),
    #     ("pca_poly2_TW", PlotStyle(color=cblue, marker=".", linestyle="--", linewidth=2)),
    #     ("geometrical_log_poly2_TW", PlotStyle(color=cred, marker="o", linestyle="-", linewidth=2)),
    #     ("pca_log_poly2_TW", PlotStyle(color=cblack, marker=".", linestyle="-", linewidth=2)),
    # ]),
    # ("RFModels_TW", [
    #     ("node_RF_TW", PlotStyle(color=cgreen, marker=".", linestyle="--", linewidth=2)),
    #     ("geometrical_RF_TW", PlotStyle(color=cpurple, marker=".", linestyle="--", linewidth=2)),
    #     ("pca_RF_TW", PlotStyle(color=cblue, marker=".", linestyle="--", linewidth=2)),
    #     ("geometrical_log_RF_TW", PlotStyle(color=cred, marker="o", linestyle="-", linewidth=2)),
    #     ("pca_log_RF_TW", PlotStyle(color=cblack, marker=".", linestyle="-", linewidth=2)),
    # ]),
    # ("NNModels_TW", [
    #     ("node_nn_TW", PlotStyle(color=cgreen, marker=".", linestyle="--", linewidth=2)),
    #     ("geometrical_nn_TW", PlotStyle(color=cpurple, marker=".", linestyle="--", linewidth=2)),
    #     ("pca_nn_TW", PlotStyle(color=cblue, marker=".", linestyle="--", linewidth=2)),
    #     ("geometrical_log_nn_TW", PlotStyle(color=cred, marker="o", linestyle="-", linewidth=2)),
    #     ("pca_log_nn_TW", PlotStyle(color=cblack, marker=".", linestyle="-", linewidth=2)),
    # ]),
    # ("Poly2NNModels_TW", [
    #     ("node_poly2NN_TW", PlotStyle(color=cgreen, marker=".", linestyle="--", linewidth=2)),
    #     ("geometrical_poly2NN_TW", PlotStyle(color=cpurple, marker=".", linestyle="--", linewidth=2)),
    #     ("pca_poly2NN_TW", PlotStyle(color=cblue, marker=".", linestyle="--", linewidth=2)),
    #     ("geometrical_log_poly2NN_TW", PlotStyle(color=cred, marker="o", linestyle="-", linewidth=2)),
    #     ("pca_log_poly2NN_TW", PlotStyle(color=cblack, marker=".", linestyle="-", linewidth=2)),
    # ]),
    # ("Poly3_TW", [
    #     ("node_poly3_TW", PlotStyle(color=cgreen, marker=".", linestyle="--", linewidth=2)),
    #     ("geometrical_poly3_TW", PlotStyle(color=cpurple, marker=".", linestyle="--", linewidth=2)),
    #     ("pca_poly3_TW", PlotStyle(color=cblue, marker=".", linestyle="--", linewidth=2)),
    #     ("geometrical_log_poly3_TW", PlotStyle(color=cred, marker="o", linestyle="-", linewidth=2)),
    #     ("pca_log_poly3_TW", PlotStyle(color=cblack, marker=".", linestyle="-", linewidth=2)),
    # ]),
    # ---------- TW regressors ---------- #
    # ("LinearModels_TW_10", [
    #     ("node_linear_TW", PlotStyle(color=cgreen, marker=".", linestyle="--", linewidth=2)),
    #     ("geometrical_linear_TW_10", PlotStyle(color=cpurple, marker=".", linestyle="--", linewidth=2)),
    #     ("pca_linear_TW_10", PlotStyle(color=cblue, marker=".", linestyle="--", linewidth=2)),
    #     ("geometrical_log_linear_TW_10", PlotStyle(color=cred, marker=".", linestyle="-", linewidth=2)),
    #     ("pca_log_linear_TW_10", PlotStyle(color=cblack, marker=".", linestyle="-", linewidth=2)),
    # ]),
    # ("QuadraticModels_TW_10", [
    #     ("node_poly2_TW", PlotStyle(color=cgreen, marker=".", linestyle="--", linewidth=2)),
    #     ("geometrical_poly2_TW_10", PlotStyle(color=cpurple, marker=".", linestyle="--", linewidth=2)),
    #     ("pca_poly2_TW_10", PlotStyle(color=cblue, marker=".", linestyle="--", linewidth=2)),
    #     ("geometrical_log_poly2_TW_10", PlotStyle(color=cred, marker="o", linestyle="-", linewidth=2)),
    #     ("pca_log_poly2_TW_10", PlotStyle(color=cblack, marker=".", linestyle="-", linewidth=2)),
    # ]),
    # ("RFModels_TW_10", [
    #     ("node_RF_TW", PlotStyle(color=cgreen, marker=".", linestyle="--", linewidth=2)),
    #     ("geometrical_RF_TW_10", PlotStyle(color=cpurple, marker=".", linestyle="--", linewidth=2)),
    #     ("pca_RF_TW_10", PlotStyle(color=cblue, marker=".", linestyle="--", linewidth=2)),
    #     ("geometrical_log_RF_TW_10", PlotStyle(color=cred, marker="o", linestyle="-", linewidth=2)),
    #     ("pca_log_RF_TW_10", PlotStyle(color=cblack, marker=".", linestyle="-", linewidth=2)),
    # ]),
    # ("NNModels_TW_10", [
    #     ("node_nn_TW", PlotStyle(color=cgreen, marker=".", linestyle="--", linewidth=2)),
    #     ("geometrical_nn_TW_10", PlotStyle(color=cpurple, marker=".", linestyle="--", linewidth=2)),
    #     ("pca_nn_TW_10", PlotStyle(color=cblue, marker=".", linestyle="--", linewidth=2)),
    #     ("geometrical_log_nn_TW_10", PlotStyle(color=cred, marker="o", linestyle="-", linewidth=2)),
    #     ("pca_log_nn_TW_10", PlotStyle(color=cblack, marker=".", linestyle="-", linewidth=2)),
    # ]),
    # ("Poly2NNModels_TW", [
    #     ("node_poly2NN_TW_10", PlotStyle(color=cgreen, marker=".", linestyle="--", linewidth=2)),
    #     ("geometrical_poly2NN_TW_10", PlotStyle(color=cpurple, marker=".", linestyle="--", linewidth=2)),
    #     ("pca_poly2NN_TW_10", PlotStyle(color=cblue, marker=".", linestyle="--", linewidth=2)),
    #     ("geometrical_log_poly2NN_TW_10", PlotStyle(color=cred, marker="o", linestyle="-", linewidth=2)),
    #     ("pca_log_poly2NN_TW_10", PlotStyle(color=cblack, marker=".", linestyle="-", linewidth=2)),
    # ]),
    # ("Poly3_TW_10", [
    #     ("node_poly3_TW", PlotStyle(color=cgreen, marker=".", linestyle="--", linewidth=2)),
    #     ("geometrical_poly3_TW_10", PlotStyle(color=cpurple, marker=".", linestyle="--", linewidth=2)),
    #     ("pca_poly3_TW_10", PlotStyle(color=cblue, marker=".", linestyle="--", linewidth=2)),
    #     ("geometrical_log_poly3_TW_10", PlotStyle(color=cred, marker="o", linestyle="-", linewidth=2)),
    #     ("pca_log_poly3_TW_10", PlotStyle(color=cblack, marker=".", linestyle="-", linewidth=2)),
    # ]),
    # ---------- TW 0.005 regressors ---------- #
    # ("LinearModels_TW_005", [
    #     ("node_linear_TW_005", PlotStyle(color=cgreen, marker=".", linestyle="--", linewidth=2)),
    #     ("geometrical_linear_TW_005", PlotStyle(color=cpurple, marker=".", linestyle="--", linewidth=2)),
    #     ("pca_linear_TW_005", PlotStyle(color=cblue, marker=".", linestyle="--", linewidth=2)),
    #     ("both_linear_TW_005", PlotStyle(color=cred, marker="o", linestyle="--", linewidth=2)),
    # ]),
    # ("QuadraticModels_TW_005", [
    #     ("node_poly2_TW_005", PlotStyle(color=cgreen, marker=".", linestyle="--", linewidth=2)),
    #     ("geometrical_poly2_TW_005", PlotStyle(color=cpurple, marker=".", linestyle="--", linewidth=2)),
    #     ("pca_poly2_TW_005", PlotStyle(color=cblue, marker=".", linestyle="--", linewidth=2)),
    #     ("both_poly2_TW_005", PlotStyle(color=cred, marker="o", linestyle="--", linewidth=2)),
    # ]),
    # ("RFModels_TW_005", [
    #     ("node_RF_TW_005", PlotStyle(color=cgreen, marker=".", linestyle="--", linewidth=2)),
    #     ("geometrical_RF_TW_005", PlotStyle(color=cpurple, marker=".", linestyle="--", linewidth=2)),
    #     ("pca_RF_TW_005", PlotStyle(color=cblue, marker=".", linestyle="--", linewidth=2)),
    #     ("both_RF_TW_005", PlotStyle(color=cred, marker="o", linestyle="--", linewidth=2)),
    # ]),
    # ("NNModels_TW_005", [
    #     ("node_nn_TW_005", PlotStyle(color=cgreen, marker=".", linestyle="--", linewidth=2)),
    #     ("geometrical_nn_TW_005", PlotStyle(color=cpurple, marker=".", linestyle="--", linewidth=2)),
    #     ("pca_nn_TW_005", PlotStyle(color=cblue, marker=".", linestyle="--", linewidth=2)),
    #     ("both_nn_TW_005", PlotStyle(color=cred, marker="o", linestyle="--", linewidth=2)),
    # ]),
    # ("Poly2NNModels_TW_005", [
    #     ("node_poly2NN_TW_005", PlotStyle(color=cgreen, marker=".", linestyle="--", linewidth=2)),
    #     ("geometrical_poly2NN_TW_005", PlotStyle(color=cpurple, marker=".", linestyle="--", linewidth=2)),
    #     ("pca_poly2NN_TW_005", PlotStyle(color=cblue, marker=".", linestyle="--", linewidth=2)),
    #     ("both_poly2NN_TW_005", PlotStyle(color=cred, marker="o", linestyle="--", linewidth=2)),
    # ]),
    # ("Poly3_TW_005", [
    #     ("node_poly3_TW_005", PlotStyle(color=cgreen, marker=".", linestyle="--", linewidth=2)),
    #     ("geometrical_poly3_TW_005", PlotStyle(color=cpurple, marker=".", linestyle="--", linewidth=2)),
    #     ("pca_poly3_TW_005", PlotStyle(color=cblue, marker=".", linestyle="--", linewidth=2)),
    #     ("both_poly3_TW_005", PlotStyle(color=cred, marker="o", linestyle="--", linewidth=2)),
    # ]),
    # ---------- 0.005 only regressors ---------- #
    # ("LinearModels_only005_10", [
    #     ("node_linear_only005", PlotStyle(color=cgreen, marker=".", linestyle="--", linewidth=2)),
    #     ("geometrical_linear_only005_10", PlotStyle(color=cpurple, marker=".", linestyle="--", linewidth=2)),
    #     ("pca_linear_only005_10", PlotStyle(color=cblue, marker=".", linestyle="--", linewidth=2)),
    #     ("geometrical_log_linear_only005_10", PlotStyle(color=cred, marker="*", linestyle="--", linewidth=2)),
    #     ("pca_log_linear_only005_10", PlotStyle(color=corange, marker="o", linestyle="--", linewidth=2)),
    # ]),
    # ("QuadraticModels_only005_10", [
    #     ("node_poly2_only005", PlotStyle(color=cgreen, marker=".", linestyle="--", linewidth=2)),
    #     ("geometrical_poly2_only005_10", PlotStyle(color=cpurple, marker=".", linestyle="--", linewidth=2)),
    #     ("pca_poly2_only005_10", PlotStyle(color=cblue, marker=".", linestyle="--", linewidth=2)),
    #     ("geometrical_log_poly2_only005_10", PlotStyle(color=cred, marker="*", linestyle="--", linewidth=2)),
    #     ("pca_log_poly2_only005_10", PlotStyle(color=corange, marker="o", linestyle="--", linewidth=2)),
    # ]),
    # ("RFModels_only005_10", [
    #     ("node_RF_only005", PlotStyle(color=cgreen, marker=".", linestyle="--", linewidth=2)),
    #     ("geometrical_RF_only005_10", PlotStyle(color=cpurple, marker=".", linestyle="--", linewidth=2)),
    #     ("pca_RF_only005_10", PlotStyle(color=cblue, marker=".", linestyle="--", linewidth=2)),
    #     ("geometrical_log_RF_only005_10", PlotStyle(color=cred, marker="*", linestyle="--", linewidth=2)),
    #     ("pca_log_RF_only005_10", PlotStyle(color=corange, marker="o", linestyle="--", linewidth=2)),
    # ]),
    # ("NNModels_only005_10", [
    #     ("node_nn_only005", PlotStyle(color=cgreen, marker=".", linestyle="--", linewidth=2)),
    #     ("geometrical_nn_only005_10", PlotStyle(color=cpurple, marker=".", linestyle="--", linewidth=2)),
    #     ("pca_nn_only005_10", PlotStyle(color=cblue, marker=".", linestyle="--", linewidth=2)),
    #     ("geometrical_log_nn_only005_10", PlotStyle(color=cred, marker="o", linestyle="--", linewidth=2)),
    #     ("pca_log_nn_only005_10", PlotStyle(color=corange, marker="o", linestyle="--", linewidth=2)),
    # ]),
    # ("Poly2NNModels_only005", [
    #     ("node_poly2NN_only005_10", PlotStyle(color=cgreen, marker=".", linestyle="--", linewidth=2)),
    #     ("geometrical_poly2NN_only005_10", PlotStyle(color=cpurple, marker=".", linestyle="--", linewidth=2)),
    #     ("pca_poly2NN_only005_10", PlotStyle(color=cblue, marker=".", linestyle="--", linewidth=2)),
    #     ("geometrical_log_poly2NN_only005_10", PlotStyle(color=cred, marker="*", linestyle="--", linewidth=2)),
    #     ("pca_log_poly2NN_only005_10", PlotStyle(color=corange, marker="o", linestyle="--", linewidth=2)),
    # ]),
    # ("Poly3_only005_10", [
    #     ("node_poly3_only005", PlotStyle(color=cgreen, marker=".", linestyle="--", linewidth=2)),
    #     ("geometrical_poly3_only005_10", PlotStyle(color=cpurple, marker=".", linestyle="--", linewidth=2)),
    #     ("pca_poly3_only005_10", PlotStyle(color=cblue, marker=".", linestyle="--", linewidth=2)),
    #     ("geometrical_poly3_only005_10", PlotStyle(color=cred, marker="*", linestyle="--", linewidth=2)),
    #     ("pca_log_poly3_only005_10", PlotStyle(color=corange, marker="o", linestyle="--", linewidth=2)),
    # ]),
    # ---------- regressor average models ---------- #
    ("SPSN", [
        ("node_linear_SPNS", PlotStyle(color=cgreen, marker=".", linestyle="--", linewidth=2)),
        ("node_poly2_SPNS", PlotStyle(color=cblue, marker=".", linestyle="--", linewidth=2)),
        ("node_RF_SPNS", PlotStyle(color=cred, marker=".", linestyle="--", linewidth=2)),
        ("node_nn_SPNS", PlotStyle(color=cpurple, marker=".", linestyle="--", linewidth=2)),
        ("node_poly2NN_SPNS", PlotStyle(color=corange, marker=".", linestyle="--", linewidth=2)),
        ("node_poly3_SPNS", PlotStyle(color=cblack, marker=".", linestyle="--", linewidth=2)),
    ]),
    # ---------- Chosen for paper ---------- #
    # ("Chosen", [
    #     ("node_linear_TW", PlotStyle(color=cgreen, marker="*", linestyle="-", linewidth=2, size=50)),
    #     ("geometrical_poly2NN_", PlotStyle(color=cred, marker="o", linestyle="--", linewidth=2, size=50)),
    #     ("pca_log_poly2_only005_10", PlotStyle(color=corange, marker=".", linestyle="-.", linewidth=1, size=50)),
    #
    #     # ("node_linear_TW", PlotStyle(color=cgreen, marker=".", linestyle="--", linewidth=2)),
    #     # # ("geometrical_poly2NN_", PlotStyle(color=cred, marker=".", linestyle="--", linewidth=2)),
    #     # ("geometrical_poly2NN_", PlotStyle(color=cred, marker="o", linestyle="--", linewidth=2)),
    #     # # ("geometrical_poly3_TW_005", PlotStyle(color=corange, marker=".", linestyle="--", linewidth=2)),
    #     # ("pca_log_poly2_only005_10", PlotStyle(color=corange, marker="*", linestyle="--", linewidth=2)),
    #     # # ("pca_linear_TW", PlotStyle(color=cblue, marker=".", linestyle="--", linewidth=2)),
    #
    #     # ("geometrical_nn_", PlotStyle(color=cgreen, marker=".", linestyle="--", linewidth=2)),
    #     # ("geometrical_poly3_TW", PlotStyle(color=cred, marker=".", linestyle="--", linewidth=2)),
    #     # ("geometrical_poly3_TWAAA", PlotStyle(color=corange, marker=".", linestyle="--", linewidth=2)),
    #     # ("EnsembleAvgNoCV_Lasso", PlotStyle(color=cblack, marker=".", linestyle="-", linewidth=2)),
    #     # ("EnsembleAvgNoCV_avg", PlotStyle(color=cblack, marker="o", linestyle="-.", linewidth=2)),
    #     # ("EnsembleAvgNoCV_Poly2", PlotStyle(color=cblack, marker="o", linestyle="-.", linewidth=2)),
    #     ("Ensemble", PlotStyle(color=cblack, marker="", linestyle="--", linewidth=2, size=None)),
    #     # ("Ensemble2", PlotStyle(color=cgray, marker="*", linestyle="-.", linewidth=2)),
    #
    #     # ("geometrical_poly2_TWAAA", PlotStyle(color=cpink, marker=".", linestyle="--", linewidth=2)),
    #     # ("geometrical_poly2NN_TWAAA", PlotStyle(color=cbrown, marker=".", linestyle="--", linewidth=2)),
    # ]),
]

map_names = None
# map_names = OrderedDict(
#     [
#         ("Spatial Avg", "Spatial average"),
#         ("BLUE", "BLUE"),
#         ("ExponentialFit", "Kriging"),
#         ("node_linear_TW", "Source"),
#         ("pca_log_poly2_only005_10", "Physical-PCA"),
#         ("geometrical_poly2NN_", "Physical-Laplacian"),
#         ("Ensemble", "Ensemble")
#     ])
stations_order = ["BONAP", "CELES", "HAUS", "OPERA", "PA13", "PA07", "PA18", "BASCH", "ELYS", "PA12", ]

if __name__ == "__main__":
    for name, group_style in groups:
        model_style = modelstyle_basics.copy()
        model_style.update(group_style)
        model_style = {model_names[k]: v for k, v in model_style.items() if k in model_names}
        models_order = list(model_names.values())
        models2plot = list(model_style.keys())

        # df = list(make_data_frames(data_manager, var_names=["relative_error", "station"], group_by=[],
        #                            individual_models="BLUE",
        #                            relative_error=lambda station, error, estimation,
        #                                                  ground_truth: np.NAN if error is None else np.abs(np.round(
        #                                np.mean((estimation[:, np.newaxis] - ground_truth[:, np.newaxis]) / ground_truth[:,
        #                                                                                                    np.newaxis]),
        #                                decimals=3))))[0][1].set_index("station")

        # for kernel_wins, stations_order in zip([True, False],
        #                                        [["HAUS", "OPERA", "PA07", "ELYS", "PA13", ],  #
        #                                         ["BONAP", "CELES", "BASCH", "PA18", "PA12", ]]):

        # ["OPERA", "HAUS", "BASCH", "PA13", "PA07", "BONAP", "CELES", "ELYS", "PA18", "PA12", ]
        # ["OPERA", "PA15L", "HAUS", "BASCH", "PA13", "PA07", "BONAP", "ELYS", "CELES", "PA12", "PA18", ]
        xlim = (3, 14)
        generic_plot(
            # format=".pdf",
            name=f"RMSE_{name}",
            data_manager=data_manager,
            # folder=path2latex_figures,
            y="station", x="RMSE", label="models",
            plot_func=NamedPartial(plot_errors, model_style=model_style, map_names=map_names,
                                   hue_order=models_order, orient="y", sort=True,
                                   y_order=stations_order,
                                   fill_between=FillBetweenInfo(model1=OptimModel, model2=BaselineModel,
                                                                model3=Krigging, model4=None,
                                                                color_low=model_style[OptimModel].color,
                                                                color_middle=cyellow,
                                                                color_high=model_style[BaselineModel].color,
                                                                alpha=0.15),
                                   xlim=xlim
                                   ),
            sort_by=["individual_models"],
            # Station=lambda station: station,
            RMSE=lambda error: np.NAN if error is None else np.sqrt(error.mean()),
            cor=lambda error, estimation, ground_truth: np.NAN if error is None else np.nansum(
                (estimation[:, np.newaxis] - np.nanmean(estimation[:, np.newaxis])) / np.nanstd(
                    estimation[:, np.newaxis]) *
                (ground_truth[:, np.newaxis] - np.nanmean(ground_truth[:, np.newaxis])) / np.nanstd(
                    ground_truth[:, np.newaxis])
            ) / (len(estimation) - 1),
            COE=lambda error, ground_truth, estimation: np.NAN if error is None else 1 - np.sum(
                np.abs(estimation - ground_truth)) / np.sum(
                np.abs(ground_truth - np.mean(ground_truth))),
            MB=lambda error, estimation, ground_truth: np.NAN if error is None else np.mean(estimation - ground_truth),
            models=lambda individual_models: model_names[individual_models],
            individual_models=models2plot,
            station=stations_order,
            dpi=300,
            axes_xy_proportions=(8, 12),
            axis_font_dict={'color': 'black', 'weight': 'normal', 'size': 16},
            labels_font_dict={'color': 'black', 'weight': 'normal', 'size': 18},
            legend_font_dict={'weight': 'normal', "size": 18, 'stretch': 'normal'},
            font_family="amssymb",
            uselatex=True,
            xlabel=fr"RMSE",
            ylabel=r"Stations",
            xlim=xlim,
            # create_preimage_data=True,
            # only_create_preimage_data=False
            legend_outside_plot=LegendOutsidePlot(loc="center right",
                                                  extra_y_top=0.01, extra_y_bottom=0.065,
                                                  extra_x_left=0.125, extra_x_right=0.275),
            # legend_outside_plot=LegendOutsidePlot(loc="lower center",
            #                                       extra_y_top=0.01, extra_y_bottom=0.275,
            #                                       extra_x_left=0.125, extra_x_right=0.075),
        )

    wuvbd
    # exit()

    for name, group_style in groups:
        xlim = (0, 0.5)
        generic_plot(
            # format=".pdf",
            name=f"RMSRE_{name}",
            data_manager=data_manager,
            # folder=path2latex_figures,
            y="station", x="RMSRE", label="models",
            plot_func=NamedPartial(plot_errors, model_style=model_style,
                                   hue_order=models_order, orient="y", sort=True,
                                   y_order=stations_order,
                                   fill_between=FillBetweenInfo(model1=OptimModel, model2=BaselineModel,
                                                                model3=Krigging, model4=None,
                                                                color_low=model_style[OptimModel].color,
                                                                color_middle=cyellow,
                                                                color_high=model_style[BaselineModel].color,
                                                                alpha=0.15),
                                   xlim=xlim
                                   ),
            sort_by=["individual_models"],
            # Station=lambda station: station,
            RMSRE=lambda error, ground_truth, estimation: np.NAN if error is None else np.sqrt(np.mean(
                (((ground_truth[:, np.newaxis] - estimation[:, np.newaxis]) / ground_truth[:, np.newaxis])) ** 2)),
            models=lambda individual_models: model_names[individual_models],
            individual_models=models2plot,
            station=stations_order,
            dpi=300,
            axes_xy_proportions=(8, 10),
            axis_font_dict={'color': 'black', 'weight': 'normal', 'size': 16},
            labels_font_dict={'color': 'black', 'weight': 'normal', 'size': 18},
            legend_font_dict={'weight': 'normal', "size": 18, 'stretch': 'normal'},
            font_family="amssymb",
            uselatex=True,
            xlabel="RMSRE",
            ylabel=r"Stations",
            xlim=xlim,
            # create_preimage_data=True,
            # only_create_preimage_data=False
            legend_outside_plot=LegendOutsidePlot(loc="lower center",
                                                  extra_y_top=0.01, extra_y_bottom=0.45,
                                                  extra_x_left=0.125, extra_x_right=0.075),
        )

        xlim = (0.75, 1)
        generic_plot(
            # format=".pdf",
            name=f"cor_{name}",
            data_manager=data_manager,
            # folder=path2latex_figures,
            y="station", x="cor", label="models",
            plot_func=NamedPartial(plot_errors, model_style=model_style,
                                   hue_order=models_order, orient="y", sort=True,
                                   y_order=stations_order,
                                   fill_between=FillBetweenInfo(model1=BaselineModel, model2=OptimModel,
                                                                model3=None, model4=Krigging,
                                                                color_low=model_style[BaselineModel].color,
                                                                color_middle=cyellow,
                                                                color_high=model_style[OptimModel].color,
                                                                alpha=0.15),
                                   xlim=xlim
                                   ),
            sort_by=["individual_models"],
            # Station=lambda station: station,
            RMSE=lambda error: np.NAN if error is None else np.sqrt(error.mean()),
            cor=lambda error, estimation, ground_truth: np.NAN if error is None else np.nansum(
                (estimation[:, np.newaxis] - np.nanmean(estimation[:, np.newaxis])) / np.nanstd(
                    estimation[:, np.newaxis]) *
                (ground_truth[:, np.newaxis] - np.nanmean(ground_truth[:, np.newaxis])) / np.nanstd(
                    ground_truth[:, np.newaxis])
            ) / (len(estimation) - 1),
            COE=lambda error, ground_truth, estimation: np.NAN if error is None else 1 - np.sum(
                np.abs(estimation - ground_truth)) / np.sum(
                np.abs(ground_truth - np.mean(ground_truth))),
            MB=lambda error, estimation, ground_truth: np.NAN if error is None else np.mean(estimation - ground_truth),
            models=lambda individual_models: model_names[individual_models],
            individual_models=models2plot,
            station=stations_order,
            dpi=300,
            axes_xy_proportions=(8, 10),
            axis_font_dict={'color': 'black', 'weight': 'normal', 'size': 16},
            labels_font_dict={'color': 'black', 'weight': 'normal', 'size': 18},
            legend_font_dict={'weight': 'normal', "size": 18, 'stretch': 'normal'},
            font_family="amssymb",
            uselatex=True,
            xlabel="Correlation",
            ylabel=r"Stations",
            xlim=xlim,
            # create_preimage_data=True,
            # only_create_preimage_data=False
            legend_outside_plot=LegendOutsidePlot(loc="lower center",
                                                  extra_y_top=0.01, extra_y_bottom=0.3,
                                                  extra_x_left=0.125, extra_x_right=0.075),
        )

        xlim = (-0.4, 0.8)
        generic_plot(
            # format=".pdf",
            name=f"COE_{name}",
            data_manager=data_manager,
            # folder=path2latex_figures,
            y="station", x="COE", label="models",
            plot_func=NamedPartial(plot_errors, model_style=model_style,
                                   hue_order=models_order, orient="y", sort=True,
                                   y_order=stations_order,
                                   fill_between=FillBetweenInfo(model1=BaselineModel, model2=OptimModel,
                                                                model3=None, model4=Krigging,
                                                                color_low=model_style[BaselineModel].color,
                                                                color_middle=cyellow,
                                                                color_high=model_style[OptimModel].color,
                                                                alpha=0.15),
                                   xlim=xlim
                                   ),
            sort_by=["individual_models"],
            # Station=lambda station: station,
            RMSE=lambda error: np.NAN if error is None else np.sqrt(error.mean()),
            cor=lambda error, estimation, ground_truth: np.NAN if error is None else np.nansum(
                (estimation[:, np.newaxis] - np.nanmean(estimation[:, np.newaxis])) / np.nanstd(
                    estimation[:, np.newaxis]) *
                (ground_truth[:, np.newaxis] - np.nanmean(ground_truth[:, np.newaxis])) / np.nanstd(
                    ground_truth[:, np.newaxis])
            ) / (len(estimation) - 1),
            COE=lambda error, ground_truth, estimation: np.NAN if error is None else 1 - np.sum(
                np.abs(estimation - ground_truth)) / np.sum(
                np.abs(ground_truth - np.mean(ground_truth))),
            MB=lambda error, estimation, ground_truth: np.NAN if error is None else np.mean(estimation - ground_truth),
            models=lambda individual_models: model_names[individual_models],
            individual_models=models2plot,
            station=stations_order,
            dpi=300,
            axes_xy_proportions=(8, 10),
            axis_font_dict={'color': 'black', 'weight': 'normal', 'size': 16},
            labels_font_dict={'color': 'black', 'weight': 'normal', 'size': 18},
            legend_font_dict={'weight': 'normal', "size": 18, 'stretch': 'normal'},
            font_family="amssymb",
            uselatex=True,
            xlabel="COE",
            ylabel=r"Stations",
            xlim=xlim,
            # create_preimage_data=True,
            # only_create_preimage_data=False
            legend_outside_plot=LegendOutsidePlot(loc="lower center",
                                                  extra_y_top=0.01, extra_y_bottom=0.3,
                                                  extra_x_left=0.125, extra_x_right=0.075),
        )

        xlim = (0, 15)
        generic_plot(
            # format=".pdf",
            name=f"MB_{name}",
            data_manager=data_manager,
            # folder=path2latex_figures,
            y="station", x="MB", label="models",
            plot_func=NamedPartial(plot_errors, model_style=model_style,
                                   hue_order=models_order, orient="y", sort=True,
                                   y_order=stations_order,
                                   fill_between=FillBetweenInfo(model1=OptimModel, model2=BaselineModel,
                                                                model3=Krigging, model4=None,
                                                                color_low=model_style[OptimModel].color,
                                                                color_middle=cyellow,
                                                                color_high=model_style[BaselineModel].color,
                                                                alpha=0.15),
                                   xlim=xlim
                                   ),
            sort_by=["individual_models"],
            # Station=lambda station: station,
            RMSE=lambda error: np.NAN if error is None else np.sqrt(error.mean()),
            cor=lambda error, estimation, ground_truth: np.NAN if error is None else np.nansum(
                (estimation[:, np.newaxis] - np.nanmean(estimation[:, np.newaxis])) / np.nanstd(
                    estimation[:, np.newaxis]) *
                (ground_truth[:, np.newaxis] - np.nanmean(ground_truth[:, np.newaxis])) / np.nanstd(
                    ground_truth[:, np.newaxis])
            ) / (len(estimation) - 1),
            COE=lambda error, ground_truth, estimation: np.NAN if error is None else 1 - np.sum(
                np.abs(estimation - ground_truth)) / np.sum(
                np.abs(ground_truth - np.mean(ground_truth))),
            MB=lambda error, estimation, ground_truth: np.NAN if error is None else np.abs(
                np.mean(estimation - ground_truth)),
            models=lambda individual_models: model_names[individual_models],
            individual_models=models2plot,
            station=stations_order,
            dpi=300,
            axes_xy_proportions=(8, 10),
            axis_font_dict={'color': 'black', 'weight': 'normal', 'size': 16},
            labels_font_dict={'color': 'black', 'weight': 'normal', 'size': 18},
            legend_font_dict={'weight': 'normal', "size": 18, 'stretch': 'normal'},
            font_family="amssymb",
            uselatex=True,
            xlabel="MB",
            ylabel=r"Stations",
            xlim=xlim,
            # create_preimage_data=True,
            # only_create_preimage_data=False
            legend_outside_plot=LegendOutsidePlot(loc="lower center",
                                                  extra_y_top=0.01, extra_y_bottom=0.3,
                                                  extra_x_left=0.125, extra_x_right=0.075),
        )

    # stations_coordinates = get_stations_lat_long()
    # for metric in ["RMSE", ]:
    #     for d in ["mean", "median", "min"]:
    #         generic_plot(
    #             # format=".pdf",
    #             name=f"{metric}_Distance{d}_Kernel",
    #             data_manager=data_manager,
    #             # folder=path2latex_figures,
    #             x=f"{d}_distance", y=metric, label="models",
    #             plot_func=sns.lineplot,
    #             distance=lambda station: np.sqrt(
    #                 np.sum((stations_coordinates[[station]].values - stations_coordinates.loc[:,
    #                                                                  ~stations_coordinates.columns.isin([station])]) ** 2,
    #                        axis=0)),
    #             min_distance=lambda distance: np.min(distance),
    #             mean_distance=lambda distance: np.mean(distance),
    #             median_distance=lambda distance: np.median(distance),
    #             sort_by=["individual_models"],
    #             RMSE=lambda error: np.NAN if error is None else np.sqrt(error.mean()),
    #             models=lambda individual_models: model_names[individual_models],
    #             individual_models=["Kernel"],
    #             dpi=300,
    #             axes_xy_proportions=(8, 10),
    #             axis_font_dict={'color': 'black', 'weight': 'normal', 'size': 16},
    #             labels_font_dict={'color': 'black', 'weight': 'normal', 'size': 18},
    #             legend_font_dict={'weight': 'normal', "size": 18, 'stretch': 'normal'},
    #             font_family="amssymb",
    #             uselatex=True,
    #             xlabel=f"{d} distance",
    #             ylabel=r"Stations",
    #             # create_preimage_data=True,
    #             # only_create_preimage_data=False
    #             legend_outside_plot=LegendOutsidePlot(loc="lower center",
    #                                                   extra_y_top=0.01, extra_y_bottom=0.3,
    #                                                   extra_x_left=0.125, extra_x_right=0.075),
    #         )

    # plot laplacian eigenfuncs
    k_max = 10
    Kd = get_diffusion_matrix(path=data_manager.path, filename=f"diffusion_matrix", graph=graph,
                              recalculate=False, verbose=False)
    Ms = get_absorption_matrix(path=data_manager.path, filename=f"absorption_matrix", graph=graph,
                               recalculate=False, verbose=False)
    vk, vm = get_geometric_basis(path=data_manager.path,
                                 filename=f"LaplacianModel_basis_k{k_max}",
                                 Kd=Kd, Ms=Ms,
                                 k=k_max,
                                 recalculate=False, verbose=False)

    levels = np.array([16.4, 20.8, 29.7, 34.6, 47.3, 60])
    img = load_background(screenshot_period)
    node_positions = get_graph_node_positions(graph)
    for i, e in enumerate(vk.T):
        with save_fig(paths=data_manager.path, filename=f"LaplacianEigen_{i}"):
            fig, ax = plt.subplots()
            plot_estimation_map_in_graph(ax, long=node_positions[:, 0], lat=node_positions[:, 1],
                                         estimation=np.log10(np.abs(e)),
                                         img=img,
                                         cmap=sns.color_palette("coolwarm", as_cmap=True, n_colors=10),
                                         s=5, alpha=0.5, bar=True,
                                         estimation_limit_vals=(0, 1), levels=0, long_bounds=None, lat_bounds=None,
                                         n_ticks=5, method='linear', norm=None)
        # with save_fig(paths=data_manager.path, filename=f"HistLaplacianEigen_{i}"):
        #     fig, ax = plt.subplots()
        #     plt.hist(np.log10(np.abs(e)), bins=int(np.sqrt(len(node_positions))))
