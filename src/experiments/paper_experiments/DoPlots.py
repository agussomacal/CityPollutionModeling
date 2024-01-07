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
    name="NumericalResults",
    country_alpha_code="FR",
    trackCO2=True
)
data_manager.load()

OptimModel = "BLUE"
BaselineModel = "Average in space"
Krigging = "Krigging"

model_names = OrderedDict([
    ("SnapshotMeanModelmean", BaselineModel),
    ("BLUEModelBLUE", OptimModel),

    ("ExponentialKernelModel", Krigging),
    ("Pipeline(steps=[('LR', LassoCV(selection='random'))])(PM)", "Physics model"),
    ("Pipeline(steps=[('LR', LassoCV(selection='random'))])(PMK)", "Physics model 50"),
    ("Pipeline(steps=[('LR', LassoCV(selection='random'))])(LRE)", "T W Graph F  - Linear model"),
    ("Pipeline(steps=[('LR', LassoCV(selection='random'))])(NNE)", "T W Graph F - Neural network"),
    ("Pipeline(steps=[('LR', LassoCV(selection='random'))])(SMM,PM,LRE,NNE)", "Ensemble - no Kernel"),
    ("Pipeline(steps=[('LR', LassoCV(selection='random'))])(SMM,EKM,PM,LRE,NNE)", "Ensemble"),

    ("Pipeline(steps=[('LR', LassoCV(selection='random'))])(PK)", "Physics Kernel"),
    ("Pipeline(steps=[('LR', LassoCV(selection='random'))])(PM,PK)", "Ensemble Physics"),

    # ("BLUEModelBLUEI", "BLUE Distrust uniform"),
    # ("BLUEModelBLUEIU", "BLUE Distrust"),

    # ("AvgKrigging", "Global Average Krigging"),

    # ("PhysicsModel_k10", "Phisics model"),
    # ("Pipeline(steps=[('Id', IdentityTransformer())])(PM)", "Ensemble Avg - Physics model"),
    # ("Pipeline(steps=[('LR', LassoCV(selection='random'))])(PM,EKM)", "Ensemble - Physics model and Kernel"),
    # ("Pipeline(steps=[('LR', LassoCV(selection='random'))])(SMM,LRE,NNE)", "Ensemble - no Physics no Kernel"),
    # ("LR", "Graph F - Linear model"),
    # ("LR_Extra", "T W Graph F  - Linear model"),
    # ("NN_Extra", "T W Graph F - Neural network"),
    # ("Pipeline(steps=[('LR', LassoCV(selection='random'))])(SMM,EKM,PM,AK,LR,LRE,NNE)", "Ensemble"),
    # ("Pipeline(steps=[('LR', LassoCV(selection='random'))])(SMM,EKM,PM,LRE,NNE)", "Ensemble"),
    # ("Pipeline(steps=[('LR', LassoCV(selection='random'))])(NNE)", "T W Graph F - Neural network"),
    # ("Pipeline(steps=[('LR', LassoCV(selection='random'))])(LRE)", "T W Graph F  - Linear model"),
    # ("Pipeline(steps=[('LR', LassoCV(selection='random'))])(LR)", "Ensemble - Graph F - Linear model")
])

model_style = OrderedDict([

    ("SnapshotMeanModelmean", PlotStyle(color=cred, marker=None, linestyle=":")),
    ("BLUEModelBLUE", PlotStyle(color=cblue, marker=None, linestyle=":")),

    ("ExponentialKernelModel", PlotStyle(color=cyellow, marker=None, linestyle=":")),
    ("Pipeline(steps=[('LR', LassoCV(selection='random'))])(PM)", PlotStyle(color=cgreen, marker="o", linestyle="-")),
    ("Pipeline(steps=[('LR', LassoCV(selection='random'))])(PMK)", PlotStyle(color=cblue, marker="o", linestyle="-")),
    (
        "Pipeline(steps=[('LR', LassoCV(selection='random'))])(LRE)",
        PlotStyle(color=corange, marker=None, linestyle="-")),
    ("Pipeline(steps=[('LR', LassoCV(selection='random'))])(NNE)", PlotStyle(color=cred, marker=None, linestyle="-")),
    ("Pipeline(steps=[('LR', LassoCV(selection='random'))])(SMM,PM,LRE,NNE)",
     PlotStyle(color=cpurple, marker=None, linestyle="-.")),
    ("Pipeline(steps=[('LR', LassoCV(selection='random'))])(SMM,EKM,PM,LRE,NNE)",
     PlotStyle(color=cblack, marker="o", linestyle="--")),

    ("Pipeline(steps=[('LR', LassoCV(selection='random'))])(PK)", PlotStyle(color=corange, marker="o", linestyle="-")),
    ("Pipeline(steps=[('LR', LassoCV(selection='random'))])(PM,PK)",
     PlotStyle(color=cpurple, marker="o", linestyle="-")),

    # ("BLUEModelBLUE", PlotStyle(color=cblue, marker=None, linestyle=":")),
    # # ("PhysicsModel_k10", PlotStyle(color=cyellow, marker="o", linestyle="-")),
    # ("Pipeline(steps=[('LR', LassoCV(selection='random'))])(PM)", PlotStyle(color=cblue, marker="o", linestyle="-")),
    # # ("Pipeline(steps=[('Id', IdentityTransformer())])(PM)", PlotStyle(color=cblue, marker="o", linestyle="--")),
    # # ("Pipeline(steps=[('LR', LassoCV(selection='random'))])(PM,EKM)",
    # #  PlotStyle(color=cyellow, marker="o", linestyle="-")),
    # ("Pipeline(steps=[('LR', LassoCV(selection='random'))])(SMM,PM,LRE,NNE)",
    #  PlotStyle(color=cyellow, marker="o", linestyle="--")),
    # # ("BLUEModelBLUEI", PlotStyle(color=cblue, marker=".", linestyle="--")),
    # # ("BLUEModelBLUEIU", PlotStyle(color=cblue, marker="o", linestyle="-")),
    # ("ExponentialKernelModel", PlotStyle(color=corange, marker="o", linestyle="-")),
    # # ("AvgKrigging", PlotStyle(color=corange, marker=".", linestyle=":")),
    # ("SnapshotMeanModelmean", PlotStyle(color=cred, marker=None, linestyle=":")),
    # ("Pipeline(steps=[('LR', LassoCV(selection='random'))])(SMM,EKM,PM,AK,LR,LRE,NNE)",
    #  PlotStyle(color=cblack, marker="o", linestyle="-")),
    # ("Pipeline(steps=[('LR', LassoCV(selection='random'))])(SMM,LRE,NNE)",
    #  PlotStyle(color=cblack, marker="o", linestyle="--")),
    # ("Pipeline(steps=[('LR', LassoCV(selection='random'))])(NNE)", PlotStyle(color=cgreen, marker="o", linestyle="-")),
    # (
    #     "Pipeline(steps=[('LR', LassoCV(selection='random'))])(LRE)",
    #     PlotStyle(color=cpurple, marker="o", linestyle="-")),
    # # ("Pipeline(steps=[('LR', LassoCV(selection='random'))])(LR)", PlotStyle(color=cgray, marker=".", linestyle="-."))
])
model_style = {model_names[k]: v for k, v in model_style.items() if k in model_names}

# model_names.pop("LR")
# model_names.pop("LR_Extra")
# model_names.pop("NN_Extra")
models_order = list(model_names.values())
models2plot = list(model_names.keys())

# runsinfo.append_info(
#     average=model_names["SnapshotMeanModelmean"].replace(" \n", ""),
#     krigging=model_names["ExponentialKernelModel"].replace(" \n", ""),
#     avgkrigging=model_names["AvgKrigging"].replace(" \n", ""),
#     lm=model_names["LR"].replace(" \n", ""),
#     lmextra=model_names["LR_Extra"].replace(" \n", ""),
#     nn=model_names["NN_Extra"].replace(" \n", ""),
#     ensemble="Ensemble",
# )

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
            format=".pdf",
            name=f"{metric}_KernelWins{kernel_wins}",
            data_manager=data_manager,
            # folder=path2latex_figures,
            y="Station", x=metric, label="models",
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
            sort_by=["models"],
            Station=lambda station: station,
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
            models=lambda model_name: model_names[model_name],
            model_name=models2plot,
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

hdvbi

from src.experiments.paper_experiments.PreProcessPaper import plot_pollution_map_in_graph, graph, times_future

plot_pollution_map_in_graph(
    data_manager=data_manager,
    # folder=path2latex_figures,
    # time=times_future[20],
    # name="plot_map_1AM",
    time=times_future[25],
    name="plot_map_8AM",
    # diffusion_method=partial(diffusion_eq, path=data_manager.path, graph=graph, diffusion_coef=1,
    #                          absorption_coef=0.01, recalculate=False),
    # diffusion_method=partial(label_prop, graph=graph,
    #                          edge_function=lambda data: 1.0 / data["length"],
    #                          lamb=1, iter=10, p=0.5),
    # diffusion_method=lambda f: label_prop(f=f + np.random.uniform(size=np.shape(f)), graph=graph,
    #                                       edge_function=lambda data: 1.0 / data["length"],
    #                                       lamb=1, iter=10, p=0.1),
    diffusion_method=lambda f: f,
    # time=times_future[4], Screenshot_48.8580073_2.3342828_13_2022_12_8_13_15
    # time=pollution_past.index[11],
    models=lambda model_name: model_names[model_name],
    model_name=["Pipeline(steps=[('LR', LassoCV(selection='random'))])(PM)"],
    station="PA13",
    plot_by=["models", "station"],
    num_cores=1,
    nodes_indexes=np.arange(len(graph)), s=5,
    cmap=sns.color_palette("plasma", as_cmap=True),
    alpha=0.5,
    dpi=300,
    # max_val=100,
    plot_nodes=True,
    levels=0,
    # bar=True,
    # format=".pdf"
)
