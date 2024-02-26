from collections import OrderedDict, namedtuple

import numpy as np
import seaborn as sns
from sklearn.linear_model import LassoCV
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

import src.config as config
from PerplexityLab.DataManager import DataManager
from PerplexityLab.LabPipeline import LabPipeline
from PerplexityLab.miscellaneous import NamedPartial
from PerplexityLab.miscellaneous import copy_main_script_version
from PerplexityLab.visualization import generic_plot, LegendOutsidePlot
from src.experiments.paper_experiments.PreProcessPaper import train_test_model, stations2test, station_coordinates, \
    graph, times_all, traffic_by_edge, plot_pollution_map_in_graph, pollution_future
from src.experiments.paper_experiments.params4runs import runsinfo
from src.lib.Models.BaseModel import mse, NONE_OPTIM_METHOD, BAYES
from src.lib.Models.SensorDependentModels.BLUEFamily import BLUEModel
from src.lib.Models.TrueStateEstimationModels.AverageModels import SnapshotMeanModel
from src.lib.Models.TrueStateEstimationModels.KernelModels import ExponentialKernelModel
from src.lib.Models.TrueStateEstimationModels.PhysicsModel import NodeSourceModel, ProjectionFullSourceModel, \
    ModelsAggregatorNoCVTwoModels
from src.lib.Modules import Optim
from src.lib.visualization_tools import FillBetweenInfo, plot_errors

cblue, corange, cgreen, cred, cpurple, cbrown, cpink, cgray, cyellow, ccyan = sns.color_palette("tab10")
k_neighbours = 10
hidden_layer_sizes = (20, 20,)
activation = "logistic"
learning_rate_init = 0.1
learning_rate = "adaptive"
early_stopping = True
solver = "Adam"
max_iter = 10000
runsinfo.append_info(
    kneighbours=k_neighbours,
    hiddenlayers=len(hidden_layer_sizes),
    neurons=hidden_layer_sizes[0] if len(set(hidden_layer_sizes)) == 1 else hidden_layer_sizes,
    activation=activation,
    solver=solver
)

train_with_relative_error = False
data_manager = DataManager(
    path=config.paper_clean_experiments_dir,
    emissions_path=config.results_dir,
    name="Rivendel",  # shuffle False all stations no convolution for node traffic
    country_alpha_code="FR",
    trackCO2=True
)

copy_main_script_version(__file__, data_manager.path)
pca_log_poly2_only005_10 = ProjectionFullSourceModel(
    train_with_relative_error=train_with_relative_error,
    path4preprocess=data_manager.path, graph=graph,
    spacial_locations=station_coordinates,
    times=times_all,
    traffic_by_edge=traffic_by_edge,
    redo_preprocessing=False,
    name="", loss=mse, optim_method=NONE_OPTIM_METHOD,
    verbose=True, niter=25, sigma0=1,
    lnei=1, k_max=10,
    source_model=Pipeline([("PF", PolynomialFeatures(degree=2)),
                           ("LR", LassoCV(selection="cyclic", positive=False, cv=len(stations2test) - 1))]),
    sources_dist=[0.005],
    substract_mean=True,
    extra_regressors=[],
    basis="pca_log",
    kv=5, ky=5, kr=5, kd=5,
    D0=0.0, A0=0.0,
    D1=0.0, A1=0.0,
    D2=0.0, A2=0.0,
    D3=0.0, A3=0.0,
    forward_weight0=0.0, source_weight0=1,
    forward_weight1=0.0, source_weight1=1,
    forward_weight2=0.0, source_weight2=1,
    forward_weight3=0.0, source_weight3=1,
)
node_linear_TW = NodeSourceModel(
    train_with_relative_error=train_with_relative_error,
    path4preprocess=data_manager.path, graph=graph,
    spacial_locations=station_coordinates,
    times=times_all,
    traffic_by_edge=traffic_by_edge,
    redo_preprocessing=False,
    name="", loss=mse, optim_method=NONE_OPTIM_METHOD,
    verbose=True, niter=1, sigma0=1,
    lnei=1,
    source_model=LassoCV(selection="cyclic", positive=False, cv=len(stations2test) - 1),
    substract_mean=True, sources_dist=[0],
    extra_regressors=["temperature", "wind"],
)
geometrical_poly3NN_ = ProjectionFullSourceModel(
    train_with_relative_error=train_with_relative_error,
    path4preprocess=data_manager.path, graph=graph,
    spacial_locations=station_coordinates,
    times=times_all,
    traffic_by_edge=traffic_by_edge,
    redo_preprocessing=False,
    name="", loss=mse, optim_method=NONE_OPTIM_METHOD,
    verbose=True, niter=25, sigma0=1,
    lnei=1, k_max=10,
    source_model=Pipeline([
        ("PF", PolynomialFeatures(degree=3)),
        ("NN", MLPRegressor(hidden_layer_sizes=hidden_layer_sizes,
                            activation=activation,  # 'relu',
                            learning_rate_init=learning_rate_init,
                            learning_rate=learning_rate,
                            early_stopping=early_stopping,
                            solver=solver.lower(),
                            max_iter=max_iter,
                            ),
         )]),
    sources_dist=[0],
    substract_mean=True,
    extra_regressors=[],
    basis="geometrical",
    kv=5, ky=5, kr=5, kd=5,
    D0=0.0, A0=0.0,
    D1=0.0, A1=0.0,
    D2=0.0, A2=0.0,
    D3=0.0, A3=0.0,
    forward_weight0=0.0, source_weight0=1,
    forward_weight1=0.0, source_weight1=1,
    forward_weight2=0.0, source_weight2=1,
    forward_weight3=0.0, source_weight3=1,
)

pca_linear_TW = ProjectionFullSourceModel(
    train_with_relative_error=train_with_relative_error,
    path4preprocess=data_manager.path, graph=graph,
    spacial_locations=station_coordinates,
    times=times_all,
    traffic_by_edge=traffic_by_edge,
    redo_preprocessing=False,
    name="", loss=mse, optim_method=NONE_OPTIM_METHOD,
    verbose=True, niter=25, sigma0=1,
    lnei=1, k_max=10,
    source_model=LassoCV(selection="cyclic", positive=False, cv=len(stations2test) - 1),
    sources_dist=[0],
    substract_mean=True,
    extra_regressors=["temperature", "wind"],
    basis="pca",
    kv=5, ky=5, kr=5, kd=5,
    D0=0.0, A0=0.0,
    D1=0.0, A1=0.0,
    D2=0.0, A2=0.0,
    D3=0.0, A3=0.0,
    forward_weight0=0.0, source_weight0=1,
    forward_weight1=0.0, source_weight1=1,
    forward_weight2=0.0, source_weight2=1,
    forward_weight3=0.0, source_weight3=1,
)

kriging = ExponentialKernelModel(
    alpha=Optim(start=None, lower=0.001, upper=10.0),
    sensor_distrust=Optim(start=None, lower=0.0, upper=1),
    name="", loss=mse, optim_method=NONE_OPTIM_METHOD, niter=100, verbose=True)

models = {
    "Spatial average": SnapshotMeanModel(summary_statistic="mean"),
    "BLUE": BLUEModel(name="BLUE", loss=mse, optim_method=NONE_OPTIM_METHOD, niter=1000, verbose=True),
    "Kriging": kriging,
    "Source": node_linear_TW,
    "Physical-Laplacian": geometrical_poly3NN_,
    "Physical-PCA": pca_log_poly2_only005_10,
    "Ensemble": ModelsAggregatorNoCVTwoModels(
        models=[node_linear_TW, geometrical_poly3NN_, pca_log_poly2_only005_10],
        aggregator="average",
        name=None,
        model2=kriging,
        sigma=0.01
    )
}

lab = LabPipeline()
lab.define_new_block_of_functions(
    "individual_models",
    *list(map(train_test_model, models.items())),
    recalculate=False
)

lab.execute(
    data_manager,
    num_cores=15,
    forget=False,
    save_on_iteration=1,
    # station=stations2test
    station=pollution_future.columns.tolist()
)

########################################################################################################################
#                                                       Plots
########################################################################################################################
PlotStyle = namedtuple("PlotStyle", "color marker linestyle linewidth size", defaults=["black", "o", "--", None, None])
cblack = (0, 0, 0)
cwhite = (1, 1, 1)
OptimModel = "BLUE"
BaselineModel = "Spatial average"  # "Average in space"
Krigging = "Kriging"  # "Krigging"
stations_order = ["BONAP", "CELES", "HAUS", "OPERA", "PA13", "PA07", "PA18", "BASCH", "PA12", "ELYS", ]
# stations_order = ["AUT", "BP_EST", "PA15L", "PA13", "OPERA", "HAUS", "CELES",
#                   "BONAP", "BASCH", "ELYS", "PA12", "PA18", "PA07", ]
# stations_order = pollution_future.columns.tolist()
model_names = list(set(data_manager["individual_models"]))

model_names = dict(zip(model_names, model_names))
model_style = OrderedDict([
    ("Spatial average", PlotStyle(color=cred, marker="", linestyle=":", size=0)),
    ("BLUE", PlotStyle(color=cblue, marker="", linestyle=":", size=0)),
    ("Kriging", PlotStyle(color=cyellow, marker="o", linestyle=":", linewidth=2, size=50)),
    ("Source", PlotStyle(color=cgreen, marker="o", linestyle="--", linewidth=2, size=50)),
    ("Physical-Laplacian", PlotStyle(color=cred, marker="o", linestyle="--", linewidth=2, size=50)),
    ("Physical-PCA", PlotStyle(color=corange, marker="o", linestyle="-.", linewidth=1, size=50)),
    ("Ensemble", PlotStyle(color=cblack, marker="o", linestyle="--", linewidth=3, size=50)),
])

models_order = list(model_style.values())
models2plot = list(model_style.keys())
map_names = dict(zip(model_style.keys(), model_style.keys()))

ylim = (3, 14)
# ylim = (3, 25)
generic_plot(
    # format=".pdf",
    name=f"RMSE",
    data_manager=data_manager,
    # folder=path2latex_figures,
    x="station", y="RMSE", label="models",
    plot_func=NamedPartial(
        plot_errors, model_style=model_style, map_names=map_names,
        hue_order=models_order,  # orient="x",
        sort=True,
        stations_order=stations_order,
        fill_between=FillBetweenInfo(
            model1=OptimModel, model2=BaselineModel,
            model3=Krigging, model4=None,
            color_low=model_style[OptimModel].color,
            color_middle=cyellow,
            color_high=model_style[BaselineModel].color,
            alpha=0.15),
        ylim=ylim
    ),
    sort_by=["individual_models"],
    station=stations2test,
    RMSE=lambda error: np.NAN if error is None else np.sqrt(error.mean()),
    models=lambda individual_models: model_names[individual_models],
    individual_models=models2plot,
    dpi=300,
    axes_xy_proportions=(12, 8),
    axis_font_dict={'color': 'black', 'weight': 'normal', 'size': 16},
    labels_font_dict={'color': 'black', 'weight': 'normal', 'size': 18},
    legend_font_dict={'weight': 'normal', "size": 18, 'stretch': 'normal'},
    font_family="amssymb",
    uselatex=True,
    ylabel=fr"RMSE",
    xlabel=r"Stations",
    ylim=ylim,
    # create_preimage_data=True,
    # only_create_preimage_data=False
    legend_outside_plot=LegendOutsidePlot(loc="center right",
                                          extra_y_top=0.01, extra_y_bottom=0.065,
                                          extra_x_left=0.1, extra_x_right=0.2)
)

ix = 25
plot_pollution_map_in_graph(
    data_manager=data_manager,
    time=pollution_future.index[ix],
    name=f"plot_map_{pollution_future.index[ix].hour}h",
    individual_models=list(model_style.keys())[2:],
    plot_by=["individual_models"],
    station="OPERA",
    num_cores=1,
    nodes_indexes=np.arange(len(graph)),
    cmap=sns.color_palette("coolwarm", as_cmap=True,
                           # n_colors=len(levels)
                           ),
    alpha=0.75,
    dpi=50,
    estimation_limit_vals=(45, 55),
    plot_nodes=True,
    s=10,
    # levels=levels,
    levels=0,
    num_points=1000,
    bar=True,
    method="linear"
    # format=".pdf"
)
