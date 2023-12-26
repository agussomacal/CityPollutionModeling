from collections import OrderedDict, namedtuple

import numpy as np
import seaborn as sns

from PerplexityLab.DataManager import DataManager
from PerplexityLab.miscellaneous import NamedPartial
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

model_names = OrderedDict([
    ("BLUEModelBLUE", OptimModel),
    # ("BLUEModelBLUEI", "BLUE Distrust uniform"),
    # ("BLUEModelBLUEIU", "BLUE Distrust"),
    ("ExponentialKernelModel", "Krigging"),
    # ("AvgKrigging", "Global Average Krigging"),
    ("SnapshotMeanModelmean", BaselineModel),
    # ("LR", "Graph F - Linear model"),
    # ("LR_Extra", "T W Graph F  - Linear model"),
    # ("NN_Extra", "T W Graph F - Neural network"),
    ("Pipeline(steps=[('LR', LassoCV(selection='random'))])(SMM,EKM,AK,LR,LRE,NNE)", "Ensemble"),
    ("Pipeline(steps=[('LR', LassoCV(selection='random'))])(NNE)", "Ensemble - T W Graph F - Neural network"),
    ("Pipeline(steps=[('LR', LassoCV(selection='random'))])(LRE)", "Ensemble - T W Graph F  - Linear model"),
    # ("Pipeline(steps=[('LR', LassoCV(selection='random'))])(LR)", "Ensemble - Graph F - Linear model")
])

model_style = OrderedDict([
    ("BLUEModelBLUE", PlotStyle(color=cblue, marker=None, linestyle=":")),
    # ("BLUEModelBLUEI", PlotStyle(color=cblue, marker=".", linestyle="--")),
    # ("BLUEModelBLUEIU", PlotStyle(color=cblue, marker="o", linestyle="-")),
    ("ExponentialKernelModel", PlotStyle(color=corange, marker="o", linestyle="-")),
    # ("AvgKrigging", PlotStyle(color=corange, marker=".", linestyle=":")),
    ("SnapshotMeanModelmean", PlotStyle(color=cred, marker=None, linestyle=":")),
    ("Pipeline(steps=[('LR', LassoCV(selection='random'))])(SMM,EKM,AK,LR,LRE,NNE)",
     PlotStyle(color=cblack, marker="o", linestyle="--")),
    ("Pipeline(steps=[('LR', LassoCV(selection='random'))])(NNE)", PlotStyle(color=cgreen, marker="o", linestyle="-")),
    (
        "Pipeline(steps=[('LR', LassoCV(selection='random'))])(LRE)",
        PlotStyle(color=cpurple, marker="o", linestyle="-")),
    # ("Pipeline(steps=[('LR', LassoCV(selection='random'))])(LR)", PlotStyle(color=cgray, marker=".", linestyle="-."))
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
                             ["model1", "model2", "color_low", "color_middle", "color_high", "alpha"])


def plot_errors(data, x, y, hue, ax, y_order=None, model_style=None, fill_between: FillBetweenInfo = None, *args,
                **kwargs):
    # plot regions
    if fill_between is not None:
        df1 = data.loc[data[hue] == fill_between.model1].set_index(y, drop=True, inplace=False)
        df2 = data.loc[data[hue] == fill_between.model2].set_index(y, drop=True, inplace=False)
        df1 = df1 if y_order is None else df1.loc[y_order, :]
        df2 = df2 if y_order is None else df2.loc[y_order, :]
        ax.fill_betweenx(y=df1.index, x1=0, x2=df1[x], color=fill_between.color_low + (fill_between.alpha,))
        ax.fill_betweenx(y=df1.index, x1=df1[x], x2=df2[x], color=fill_between.color_middle + (fill_between.alpha,))
        ax.fill_betweenx(y=df1.index, x1=df2[x], x2=max(data[x]) * 1.1,
                         color=fill_between.color_high + (fill_between.alpha,))

    # plot models
    for method, df in data.groupby(hue, sort=False):
        df.set_index(y, inplace=True, drop=True)
        df = df if y_order is None else df.loc[y_order, :]
        sns.lineplot(
            x=df[x], y=df.index, label=method, ax=ax, alpha=1,
            color=model_style[method].color if model_style is not None else None,
            marker=model_style[method].marker if model_style is not None else None,
            linestyle=model_style[method].linestyle if model_style is not None else None,
            **kwargs
        )


# stations_order = ["OPERA", "BASCH", "PA13", "PA07", "PA18", "ELYS", "PA12", ] + ["BONAP", "HAUS", "CELES"]
# stations_order = ["ELYS", "HAUS", "OPERA", "PA13", "PA07"]
# stations_order = ["BASCH", "PA18", "PA12", "BONAP", "CELES"]
for kernel_wins, stations_order in zip([True, False],
                                       [["HAUS", "OPERA", "PA07", "PA13", "ELYS", ],  #
                                        ["BONAP", "CELES", "BASCH", "PA18", "PA12", ]]):
    generic_plot(
        name=f"ErrorPlot_KernelWins{kernel_wins}",
        data_manager=data_manager,
        # folder=path2latex_figures,
        y="station", x="mse", label="models",
        plot_func=NamedPartial(plot_errors, model_style=model_style,
                               hue_order=models_order, orient="y", sort=True,
                               y_order=stations_order,
                               fill_between=FillBetweenInfo(model1=OptimModel, model2=BaselineModel,
                                                            color_low=model_style[OptimModel].color,
                                                            color_middle=cwhite,
                                                            color_high=model_style[BaselineModel].color,
                                                            alpha=0.15)
                               ),
        sort_by=["models"],
        mse=lambda error: np.NAN if error is None else np.sqrt(error.mean()),
        models=lambda model_name: model_names[model_name],
        model_name=models2plot,
        station=stations_order,
        # xlim=(None, 20),
        dpi=300,
        format=".pdf",
        axes_xy_proportions=(8, 10),
        axis_font_dict={'color': 'black', 'weight': 'normal', 'size': 16},
        labels_font_dict={'color': 'black', 'weight': 'normal', 'size': 18},
        legend_font_dict={'weight': 'normal', "size": 18, 'stretch': 'normal'},
        font_family="amssymb",
        uselatex=True,
        xlabel=r"mse",
        ylabel=r"stations",
        # create_preimage_data=True,
        # only_create_preimage_data=False
        legend_outside_plot=LegendOutsidePlot(loc="lower center",
                                              extra_y_top=0.01, extra_y_bottom=0.25,
                                              extra_x_left=0.125, extra_x_right=0.075),
    )
