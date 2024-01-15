import inspect
from collections import OrderedDict, namedtuple

import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler

from PerplexityLab.DataManager import DataManager
from PerplexityLab.miscellaneous import NamedPartial, filter_dict, filter_dict4func
from PerplexityLab.visualization import generic_plot, LegendOutsidePlot
from src import config
from src.experiments.paper_experiments.params4runs import path2latex_figures

PlotStyle = namedtuple("PlotStyle", "color marker linestyle", defaults=["black", "o", "--"])
cblue, corange, cgreen, cred, cpurple, cbrown, cpink, cgray, cyellow, ccyan = sns.color_palette("tab10")
cblack = (0, 0, 0)
cwhite = (1, 1, 1)

data_manager = DataManager(
    path=config.paper_experiments_dir,
    emissions_path=config.results_dir,
    name="SourceModels",
    country_alpha_code="FR",
    trackCO2=True
)
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
    ("SL Spatial Avg", PlotStyle(color=cred, marker="*", linestyle=":")),

    ("BLUE", PlotStyle(color=cblue, marker=None, linestyle=":")),
    ("Kernel", PlotStyle(color=cyellow, marker=None, linestyle=":")),

    # ("SourceModel_Poly2Lasso+", PlotStyle(color=cgreen, marker=".", linestyle="--")),
    # ("SourceModel_Poly2Lasso_avg", PlotStyle(color=cgreen, marker=".", linestyle=":")),
    ("SourceModel_Poly1Lasso_avg", PlotStyle(color=cgreen, marker="o", linestyle=":")),
    ("SourceModel_Poly1Lasso_avg_TW", PlotStyle(color=cgreen, marker="o", linestyle="-")),
    # ("SourceModel_Poly1Lasso_avg_TWGW", PlotStyle(color=cred, marker="o", linestyle="-")),
    # ("SourceModel_Poly2Lasso_avg_TW", PlotStyle(color=cgreen, marker="o", linestyle="-")),


])

model_style = {model_names[k]: v for k, v in model_style.items() if k in model_names}
models_order = list(model_names.values())
models2plot = list(model_style.keys())

FillBetweenInfo = namedtuple("FillBetweenInfo",
                             ["model1", "model2", "model3", "color_low", "color_middle", "color_high", "alpha"])


def plot_errors(data, x, y, hue, ax, y_order=None, model_style=None, fill_between: FillBetweenInfo = None, *args,
                **kwargs):
    # plot regions
    if fill_between is not None:
        df1 = data.loc[data[hue] == fill_between.model1].set_index(y, drop=True, inplace=False)
        df2 = data.loc[data[hue] == fill_between.model2].set_index(y, drop=True, inplace=False)
        df3 = data.loc[data[hue] == fill_between.model3].set_index(y, drop=True, inplace=False)
        df1 = df1 if y_order is None else df1.loc[y_order, :]
        df2 = df2 if y_order is None else df2.loc[y_order, :]
        df3 = df3 if y_order is None else df3.loc[y_order, :]
        ax.fill_betweenx(y=df1.index, x1=kwargs.get("xlim", (0, None))[0], x2=df1[x],
                         color=fill_between.color_low + (fill_between.alpha,))
        # ax.fill_betweenx(y=df1.index, x1=df1[x], x2=df2[x], color=fill_between.color_middle + (fill_between.alpha,))
        ax.fill_betweenx(y=df1.index, x1=df3[x], x2=kwargs.get("xlim", (0, max(data[x])))[1] * 1.1,
                         color=fill_between.color_middle + (fill_between.alpha,))
        ax.fill_betweenx(y=df1.index, x1=df2[x], x2=kwargs.get("xlim", (0, max(data[x])))[1] * 1.1,
                         color=fill_between.color_high + (fill_between.alpha,))

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
            **kw
        )


for kernel_wins, stations_order in zip([True, False],
                                       [["HAUS", "OPERA", "PA07", "PA13", "ELYS", ],  #
                                        ["BONAP", "CELES", "BASCH", "PA18", "PA12", ]]):
    for metric in ["RMSE", ]:  # "RMSE",, "COE", "MB""cor"
        xlim = (3, 18)
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
                                                                model3=Krigging,
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
                                                  extra_y_top=0.01, extra_y_bottom=0.3,
                                                  extra_x_left=0.125, extra_x_right=0.075),
        )
