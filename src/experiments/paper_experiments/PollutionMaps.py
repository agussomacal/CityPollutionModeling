from collections import OrderedDict

import numpy as np
import pandas as pd
import seaborn as sns

from PerplexityLab.visualization import LegendOutsidePlot
from src.experiments.paper_experiments.DoPlotsSourceModels import BaselineModel, cred, PlotStyle, cgreen, cblue, \
    cpurple, corange, cblack
from src.experiments.paper_experiments.SourceModels import data_manager

name2name = dict(pd.DataFrame.from_dict(data_manager[["individual_models", "model_name"]]).values.tolist())
levels = np.array([16.4, 20.8, 29.7, 34.6, 47.3, 60])

from src.experiments.paper_experiments.PreProcessPaper import plot_pollution_map_in_graph, graph, \
    pollution_future, plot_estimation_histogram

model_style = OrderedDict([
    ("geometrical_nn_", PlotStyle(color=cgreen, marker=".", linestyle="--", linewidth=2)),
    ("geometrical_poly2NN_", PlotStyle(color=cblue, marker=".", linestyle="--", linewidth=2)),
    ("geometrical_poly3_", PlotStyle(color=cred, marker=".", linestyle="--", linewidth=2)),
    ("geometrical_poly2_", PlotStyle(color=cpurple, marker=".", linestyle="--", linewidth=2)),
    ("geometrical_poly3_TWHW", PlotStyle(color=corange, marker=".", linestyle="--", linewidth=2)),
    # ("EnsembleAvgNoCV_Lasso", PlotStyle(color=cblack, marker="o", linestyle="--", linewidth=2)),
    ("EnsembleAvgNoCV_avg", PlotStyle(color=cblack, marker="o", linestyle="--", linewidth=2)),
    # ("geometrical_poly2_TWHW", PlotStyle(color=cpink, marker=".", linestyle="--", linewidth=2)),
    # ("geometrical_poly2NN_TWHW", PlotStyle(color=cbrown, marker=".", linestyle="--", linewidth=2)),
])

# levels=10,
# levels=np.array([40, 45, 46, 47, 48, 50, 52, 55, 60]),
# levels=np.array([0, 11.3, 16.4, 20.8, 25.6, 29.7, 34.6, 40.2, 47.3, 60]),
# levels=np.array([10, 20, 22.5, 25, 27.5, 30, 35, 40, 45, 50, 60]),
for ix in [25, 30, 40]:
    plot_pollution_map_in_graph(
        data_manager=data_manager,
        # folder=path2latex_figures,
        # time=times_future[20],
        # name="plot_map_1AM",
        time=pollution_future.index[ix],
        name=f"plot_map_{pollution_future.index[ix].hour}h",
        individual_models=list(model_style.keys()),
        # individual_models=["Exponential"],
        plot_by=["individual_models"],
        # time=times_future[4], Screenshot_48.8580073_2.3342828_13_2022_12_8_13_15
        # time=pollution_past.index[11],
        # models=lambda model_name: model_names[model_name],
        # model_name=[name2name["PCASourceModel_Poly1Lasso_avg_TWHW"]],
        # station="PA13",
        station="OPERA",
        # plot_by=["models", "station"],
        num_cores=1,
        nodes_indexes=np.arange(len(graph)),
        # cmap=sns.color_palette("coolwarm", as_cmap=True, n_colors=len(levels)),
        cmap=sns.color_palette("coolwarm", as_cmap=True, n_colors=len(levels)),
        # norm=colors.CenteredNorm(),
        # norm=colors.BoundaryNorm(boundaries=levels, ncolors=len(levels), clip=True),
        alpha=0.5,
        dpi=300,
        # limit_vals=(40, 50),
        estimation_limit_vals=(np.min(levels), np.max(levels)),
        # limit_vals=(0.1, 0.9),
        # limit_vals=np.quantile(np.ravel(pollution_future.values.mean(axis=1)), q=limit_vals),
        plot_nodes=True,
        s=10,
        levels=levels,
        num_points=1000,
        bar=True,
        method="linear"
        # format=".pdf"
    )

for ix in [25, 30, 40]:
    plot_estimation_histogram(
        data_manager=data_manager,
        spatial_avg_model_name=BaselineModel,
        spatial_avg_model_color=cred,
        # folder=path2latex_figures,
        # time=times_future[20],
        # name="plot_map_1AM",
        time=pollution_future.index[ix],
        name=f"histogram_of_pollution_predictions_{pollution_future.index[ix].hour}h",
        individual_models=list(model_style.keys()),
        # model_style=model_style,
        # individual_models=["Exponential"],
        # plot_by=["individual_models"],
        xlim=(10, 70),
        # station="PA13",
        station="OPERA",
        num_cores=1,
        nodes_indexes=np.arange(len(graph)),
        dpi=300,
        bar=True,
        axes_xy_proportions=(8, 6),
        axis_font_dict={'color': 'black', 'weight': 'normal', 'size': 16},
        labels_font_dict={'color': 'black', 'weight': 'normal', 'size': 18},
        legend_font_dict={'weight': 'normal', "size": 18, 'stretch': 'normal'},
        font_family="amssymb",
        uselatex=True,
        xlabel=r"$NO_2$ concentration",
        ylabel=r"Counts",
        add_legend=True,
        legend_outside_plot=LegendOutsidePlot(loc="lower center",
                                              extra_y_top=0.01, extra_y_bottom=0.4,
                                              extra_x_left=0.125, extra_x_right=0.075),
        # xlim=xlim,
        # format=".pdf"
    )
