import inspect
from collections import OrderedDict, namedtuple

import numpy as np
import seaborn as sns
from matplotlib import colors

from PerplexityLab.miscellaneous import NamedPartial, filter_dict
from PerplexityLab.visualization import generic_plot, LegendOutsidePlot, make_data_frames
from src.experiments.paper_experiments.SourceModels import data_manager

PlotStyle = namedtuple("PlotStyle", "color marker linestyle linewidth", defaults=["black", "o", "--", None])
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
model_style = OrderedDict([
    ("Spatial Avg", PlotStyle(color=cred, marker=None, linestyle=":")),
    # ("SL Spatial Avg", PlotStyle(color=cred, marker="*", linestyle=":")),

    ("BLUE", PlotStyle(color=cblue, marker=None, linestyle=":")),
    # ("BLUE_DU", PlotStyle(color=cblue, marker=None, linestyle="--", linewidth=2)),
    # ("BLUE_DI", PlotStyle(color=cblue, marker="o", linestyle="-", linewidth=1)),
    # ("Kernel", PlotStyle(color=cyellow, marker=None, linestyle=":")),
    # ("Gaussian", PlotStyle(color=cyellow, marker="*", linestyle="--", linewidth=2)),
    # ("Exponential", PlotStyle(color=cyellow, marker="o", linestyle="--", linewidth=3)),
    # ("ExponentialD", PlotStyle(color=cred, marker=".", linestyle="--", linewidth=2)),
    ("ExponentialFit", PlotStyle(color=cyellow, marker="o", linestyle="-", linewidth=2)),
    # ("ExponentialOld", PlotStyle(color=cyellow, marker="o", linestyle="-", linewidth=5)),

    # ("SourceModel_Poly2Lasso+", PlotStyle(color=cgreen, marker=".", linestyle="--")),
    # ("SourceModel_Poly2Lasso_avg", PlotStyle(color=cgreen, marker=".", linestyle=":")),
    # ("SourceModel_Poly1Lasso_avg", PlotStyle(color=corange, marker="o", linestyle=":")),
    # ("SourceModel_Poly1Lasso_avg_TWHW", PlotStyle(color=cgreen, marker="o", linestyle=":")),
    # ("SourceModel_NN_avg_TWHW", PlotStyle(color=cblue, marker="o", linestyle=":")),
    # ("SourceModel_RF_avg_TWHW", PlotStyle(color=cred, marker="o", linestyle=":")),

    # ("PCAAfterSourceModel_LM_TWHW", PlotStyle(color=cblack, marker="o", linestyle=":")),
    # ("LapAfterSourceModel_LM_TWHW", PlotStyle(color=cblack, marker="o", linestyle="-.")),
    #
    # ("PCASourceModel_Poly1Lasso_avg_TWHW", PlotStyle(color=cred, marker="o", linestyle="-")),
    #
    # ("PCAAfterSourceModel_LM_TWHW", PlotStyle(color=corange, marker="o", linestyle="-")),
    # ("PCA_ProjectionFullSourceModel_LM_TWHW", PlotStyle(color=cblue, marker="o", linestyle="-")),

    ("node_linear_TWHW", PlotStyle(color=cgreen, marker=".", linestyle="--", linewidth=2)),
    # ("node_poly2_TWHW", PlotStyle(color=cpurple, marker=".", linestyle=":", linewidth=2)),
    # ("node_nn_TWHW", PlotStyle(color=cblue, marker="o", linestyle="-", linewidth=2)),
    # ("node_RF_TWHW", PlotStyle(color=cred, marker="*", linestyle="-.", linewidth=2)),

    # ("geometrical_linear_TWHW", PlotStyle(color=cgreen, marker=".", linestyle="--", linewidth=2)),
    # ("geometrical_poly2_TWHW", PlotStyle(color=cpurple, marker=".", linestyle=":", linewidth=2)),
    ("geometrical_nn_TWHW", PlotStyle(color=cblue, marker="o", linestyle="-", linewidth=2)),
    ("geometrical_RF_TWHW", PlotStyle(color=cred, marker="*", linestyle="-.", linewidth=2)),

    # ("pca_linear_TWHW", PlotStyle(color=cgreen, marker=".", linestyle="--", linewidth=2)),
    # ("pca_poly2_TWHW", PlotStyle(color=cpurple, marker=".", linestyle=":", linewidth=2)),
    # ("pca_nn_TWHW", PlotStyle(color=cblue, marker="o", linestyle="-", linewidth=2)),
    # ("pca_RF_TWHW", PlotStyle(color=cred, marker="*", linestyle="-.", linewidth=2)),

    # ("both_linear_TWHW", PlotStyle(color=cred, marker="o", linestyle="--", linewidth=2)),
    # ("both_poly2_TWHW", PlotStyle(color=cred, marker="o", linestyle="-", linewidth=2)),
    # ("both_nn_TWHW", PlotStyle(color=cred, marker="o", linestyle="-", linewidth=2)),
    # ("both_RF_TWHW", PlotStyle(color=cred, marker="o", linestyle="-.", linewidth=2)),

    # ("EnsembleAvg", PlotStyle(color=cblack, marker="o", linestyle="-", linewidth=3)),
    # ("EnsembleAvgNoCV", PlotStyle(color=cblack, marker=".", linestyle="--", linewidth=3)),
    # ("EnsembleAvgNoCV_RF", PlotStyle(color=cblack, marker=".", linestyle="-.", linewidth=3)),
    # ("EnsembleAvgNoCV_nn", PlotStyle(color=cblack, marker=".", linestyle="-", linewidth=3)),
    # ("EnsembleAvgNoCV_average", PlotStyle(color=cblack, marker=".", linestyle="--", linewidth=3)),
    # ("EnsembleAvgNoCV_std", PlotStyle(color=cblack, marker=".", linestyle="-.", linewidth=3)),
    # ("EnsembleAvgNoCV_cv", PlotStyle(color=cblack, marker=".", linestyle="-", linewidth=3)),
    ("EnsembleAvgNoCV_Lasso", PlotStyle(color=cblack, marker=".", linestyle=":", linewidth=3)),
    # ("EnsembleAvgNoCV_weighted_average", PlotStyle(color=cblack, marker=".", linestyle="--", linewidth=3)),

    # ("DistanceRational", PlotStyle(color=cpink, marker="o", linestyle="-", linewidth=3)),

    # # ("LaplacianSourceModel_Poly1Lasso_avg_TW", PlotStyle(color=cpurple, marker="o", linestyle="-")),
    # ("PCASourceModel_Poly1Lasso_avg_TWHW", PlotStyle(color=cgreen, marker="o", linestyle="-")),
    # ("PCASourceModel_Poly1NN_avg_TWHW", PlotStyle(color=cblue, marker="o", linestyle="-")),
    # ("PCASourceModel_Poly1RF_avg_TWHW", PlotStyle(color=cred, marker="o", linestyle="-")),
    # # ("LaplacianSourceModel_Poly1Lasso_avg_TWHW", PlotStyle(color=cpurple, marker="o", linestyle="-")),
    #
    # ("V2LaplacianSourceModel_Poly1Lasso_avg_TWHW", PlotStyle(color=cgreen, marker="*", linestyle="--", linewidth=2)),
    # ("V2LaplacianSourceModel_NN_avg_TWHW", PlotStyle(color=cblue, marker="*", linestyle="--", linewidth=2)),
    # ("V2LaplacianSourceModel_RF_avg_TWHW", PlotStyle(color=cred, marker="*", linestyle="--", linewidth=2)),

    # ("PCASourceModel_Poly1Lasso_avg_TWHW_nei+", PlotStyle(color=ccyan, marker="o", linestyle="-")),
    # ("PhysicsModel", PlotStyle(color=cgray, marker="*", linestyle="--", linewidth=2)),
    # ("Ensemble", PlotStyle(color=cblack, marker="*", linestyle="-", linewidth=3)),
    # ("Ensemble2", PlotStyle(color=cblack, marker="o", linestyle=":", linewidth=3)),
    # ("EnsembleKernel", PlotStyle(color=cblack, marker="*", linestyle="--", linewidth=3)),
    # ("EnsembleKernelAvg", PlotStyle(color=cblack, marker="*", linestyle="--", linewidth=3)),
    # ("EnsembleKernelBLUE", PlotStyle(color=cblack, marker="o", linestyle="-", linewidth=3)),
    # ("EnsembleKernelMSE", PlotStyle(color=cblack, marker="o", linestyle="-", linewidth=3)),
    # ("EnsembleBLUE", PlotStyle(color=cpurple, marker="o", linestyle="-", linewidth=3)),
    # ("EnsembleAvg", PlotStyle(color=cblack, marker="o", linestyle="-", linewidth=3)),

    # ("SoftDiffusion", PlotStyle(color=cblue, marker="o", linestyle="-", linewidth=3)),

    # ("SourceModel_Poly1Lasso_avg_TWGW", PlotStyle(color=cred, marker="o", linestyle="-")),
    # ("SourceModel_Poly2Lasso_avg_TW", PlotStyle(color=cgreen, marker="o", linestyle="-")),

])

model_style = {model_names[k]: v for k, v in model_style.items() if k in model_names}
models_order = list(model_names.values())
models2plot = list(model_style.keys())

FillBetweenInfo = namedtuple("FillBetweenInfo",
                             ["model1", "model2", "model3", "model4", "color_low", "color_middle", "color_high",
                              "alpha"])


def plot_errors(data, x, y, hue, ax, y_order=None, model_style=None, fill_between: FillBetweenInfo = None, *args,
                **kwargs):
    # plot regions
    if fill_between is not None:
        if (fill_between.model1 is not None) and (fill_between.model1 in data[hue].values):
            df1 = data.loc[data[hue] == fill_between.model1].set_index(y, drop=True, inplace=False)
            df1 = df1 if y_order is None else df1.loc[y_order, :]
            ax.fill_betweenx(y=df1.index, x1=kwargs.get("xlim", (0, None))[0], x2=df1[x],
                             color=fill_between.color_low + (fill_between.alpha,))

        if fill_between.model2 is not None and fill_between.model2 in data[hue].values:
            df2 = data.loc[data[hue] == fill_between.model2].set_index(y, drop=True, inplace=False)
            df2 = df2 if y_order is None else df2.loc[y_order, :]
            ax.fill_betweenx(y=df2.index, x1=df2[x], x2=kwargs.get("xlim", (0, max(data[x])))[1] * 1.1,
                             color=fill_between.color_high + (fill_between.alpha,))

        if fill_between.model3 is not None and fill_between.model3 in data[hue].values:
            df3 = data.loc[data[hue] == fill_between.model3].set_index(y, drop=True, inplace=False)
            df3 = df3 if y_order is None else df3.loc[y_order, :]
            ax.fill_betweenx(y=df3.index, x1=df3[x], x2=kwargs.get("xlim", (0, max(data[x])))[1] * 1.1,
                             color=fill_between.color_middle + (fill_between.alpha,))

        if fill_between.model4 is not None and fill_between.model4 in data[hue].values:
            df4 = data.loc[data[hue] == fill_between.model4].set_index(y, drop=True, inplace=False)
            df4 = df4 if y_order is None else df4.loc[y_order, :]
            ax.fill_betweenx(y=df4.index, x1=kwargs.get("xlim", (0, None))[0], x2=df4[x],
                             color=fill_between.color_middle + (fill_between.alpha,))

    # plot models
    ins = inspect.getfullargspec(sns.lineplot)
    kw = filter_dict(ins.args + ins.kwonlyargs, kwargs)
    for method, df in data.groupby(hue, sort=False):
        df.set_index(y, inplace=True, drop=True)
        df = df if y_order is None else df.loc[y_order, :]
        sns.lineplot(
            x=df[x], y=df.index, label=method, ax=ax, alpha=1,
            color=model_style[method].color if model_style is not None else None,
            marker=model_style[method].marker if model_style is not None else None,
            linestyle=model_style[method].linestyle if model_style is not None else None,
            linewidth=model_style[method].linewidth if model_style is not None else None,
            **kw
        )


df = list(make_data_frames(data_manager, var_names=["relative_error", "station"], group_by=[],
                           individual_models="BLUE",
                           relative_error=lambda station, error, estimation,
                                                 ground_truth: np.NAN if error is None else np.abs(np.round(
                               np.mean((estimation[:, np.newaxis] - ground_truth[:, np.newaxis]) / ground_truth[:,
                                                                                                   np.newaxis]),
                               decimals=3))))[0][1].set_index("station")

# for kernel_wins, stations_order in zip([True, False],
#                                        [["HAUS", "OPERA", "PA07", "ELYS", "PA13", ],  #
#                                         ["BONAP", "CELES", "BASCH", "PA18", "PA12", ]]):


for kernel_wins, stations_order in zip([""], [
    # ["BONAP", "CELES", "HAUS", "OPERA", "PA13", "PA07", "PA18", "BASCH", "ELYS", "PA12", ]
    ["OPERA", "HAUS", "BASCH", "PA13", "PA07", "BONAP", "CELES", "ELYS", "PA18", "PA12", ]
    # ["OPERA", "PA15L", "HAUS", "BASCH", "PA13", "PA07", "BONAP", "CELES", "ELYS", "PA18", "PA12", ]
]):
    for metric in ["RMSE", ]:  # "RMSE",, "COE", "MB""cor"
        xlim = (3, 14)
        generic_plot(
            # format=".pdf",
            name=f"{metric}_KernelWins{kernel_wins}",
            data_manager=data_manager,
            # folder=path2latex_figures,
            y="station", x=metric, label="models",
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
            xlabel=fr"{metric}",
            ylabel=r"Stations",
            xlim=xlim,
            # create_preimage_data=True,
            # only_create_preimage_data=False
            legend_outside_plot=LegendOutsidePlot(loc="lower center",
                                                  extra_y_top=0.01, extra_y_bottom=0.45,
                                                  extra_x_left=0.125, extra_x_right=0.075),
        )

    for metric in ["RMSRE", ]:  # "RMSE",, "COE", "MB""cor"
        xlim = (0, 0.5)
        generic_plot(
            # format=".pdf",
            name=f"{metric}_KernelWins{kernel_wins}",
            data_manager=data_manager,
            # folder=path2latex_figures,
            y="station", x=metric, label="models",
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
            xlabel=fr"{metric}",
            ylabel=r"Stations",
            xlim=xlim,
            # create_preimage_data=True,
            # only_create_preimage_data=False
            legend_outside_plot=LegendOutsidePlot(loc="lower center",
                                                  extra_y_top=0.01, extra_y_bottom=0.45,
                                                  extra_x_left=0.125, extra_x_right=0.075),
        )

    #
    for metric in ["cor", ]:  # "RMSE",,"COE", "MB",
        xlim = (0.75, 1)
        generic_plot(
            # format=".pdf",
            name=f"{metric}_KernelWins{kernel_wins}",
            data_manager=data_manager,
            # folder=path2latex_figures,
            y="station", x=metric, label="models",
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
            xlabel=fr"{metric}",
            ylabel=r"Stations",
            xlim=xlim,
            # create_preimage_data=True,
            # only_create_preimage_data=False
            legend_outside_plot=LegendOutsidePlot(loc="lower center",
                                                  extra_y_top=0.01, extra_y_bottom=0.3,
                                                  extra_x_left=0.125, extra_x_right=0.075),
        )

    for metric in ["COE", ]:  # "RMSE",, "COE", "MB""cor"
        xlim = (-0.4, 0.8)
        generic_plot(
            # format=".pdf",
            name=f"{metric}_KernelWins{kernel_wins}",
            data_manager=data_manager,
            # folder=path2latex_figures,
            y="station", x=metric, label="models",
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
            xlabel=fr"{metric}",
            ylabel=r"Stations",
            xlim=xlim,
            # create_preimage_data=True,
            # only_create_preimage_data=False
            legend_outside_plot=LegendOutsidePlot(loc="lower center",
                                                  extra_y_top=0.01, extra_y_bottom=0.3,
                                                  extra_x_left=0.125, extra_x_right=0.075),
        )

    for metric in ["MB", ]:  # "RMSE",, "COE", "MB""cor"
        xlim = (0, 15)
        generic_plot(
            # format=".pdf",
            name=f"{metric}_KernelWins{kernel_wins}",
            data_manager=data_manager,
            # folder=path2latex_figures,
            y="station", x=metric, label="models",
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
            xlabel=fr"{metric}",
            ylabel=r"Stations",
            xlim=xlim,
            # create_preimage_data=True,
            # only_create_preimage_data=False
            legend_outside_plot=LegendOutsidePlot(loc="lower center",
                                                  extra_y_top=0.01, extra_y_bottom=0.3,
                                                  extra_x_left=0.125, extra_x_right=0.075),
        )

from src.experiments.paper_experiments.PreProcessPaper import plot_pollution_map_in_graph, graph, \
    pollution_future, plot_estimation_histogram
import pandas as pd

name2name = dict(pd.DataFrame.from_dict(data_manager[["individual_models", "model_name"]]).values.tolist())

for ix in [25, 30, 40]:
    plot_estimation_histogram(
        data_manager=data_manager,
        spatial_avg_model_name=BaselineModel,
        spatial_avg_model_color=model_style[BaselineModel].color,
        # folder=path2latex_figures,
        # time=times_future[20],
        # name="plot_map_1AM",
        time=pollution_future.index[ix],
        name=f"histogram_of_pollution_predictions_{pollution_future.index[ix].hour}h",
        individual_models=list(model_style.keys())[2:],
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

# levels=10,
# levels=np.array([40, 45, 46, 47, 48, 50, 52, 55, 60]),
# levels=np.array([0, 11.3, 16.4, 20.8, 25.6, 29.7, 34.6, 40.2, 47.3, 60]),
# levels=np.array([10, 20, 22.5, 25, 27.5, 30, 35, 40, 45, 50, 60]),
levels = np.array([16.4, 20.8, 29.7, 34.6, 47.3, 60])
for ix in [25, 30, 40]:
    plot_pollution_map_in_graph(
        data_manager=data_manager,
        # folder=path2latex_figures,
        # time=times_future[20],
        # name="plot_map_1AM",
        time=pollution_future.index[ix],
        name=f"plot_map_{pollution_future.index[ix].hour}h",
        individual_models=list(model_style.keys())[2:],
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
        log=False,
        bar=True,
        method="linear"
        # format=".pdf"
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
