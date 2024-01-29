import inspect
from collections import namedtuple

import numpy as np
from matplotlib import colors, pylab as plt
from scipy.interpolate import griddata
import seaborn as sns

from PerplexityLab.miscellaneous import filter_dict
from src.lib.Modules import Bounds


def plot_estimation_map_in_graph(ax, long, lat, estimation, img, cmap='RdGy', s=20, alpha=0.5, bar=False,
                                 estimation_limit_vals=(0, 1), levels=0, long_bounds=None, lat_bounds=None,
                                 n_ticks=5, method='cubic', norm=None):
    long_bounds = long_bounds if long_bounds is not None else Bounds(lower=np.min(long), upper=np.max(long))
    lat_bounds = lat_bounds if lat_bounds is not None else Bounds(lower=np.min(lat), upper=np.max(lat))
    long = (long - long_bounds.lower) / (long_bounds.upper - long_bounds.lower) * np.shape(img)[1]
    lat = (lat - lat_bounds.lower) / (lat_bounds.upper - lat_bounds.lower) * np.shape(img)[0]
    extent = (min(long), max(long), min(lat), max(lat))

    if estimation_limit_vals[1] <= 1:
        # limit_vals = np.quantile(np.ravel(pollution_future.values.mean(axis=1)), q=limit_vals)
        estimation_limit_vals = np.quantile(estimation, q=estimation_limit_vals)
    if norm is None:
        norm = colors.Normalize(vmin=estimation_limit_vals[0],
                                vmax=estimation_limit_vals[1],
                                clip=True) if estimation_limit_vals is not None else None
    if s > 0:
        sc = ax.scatter(x=long,
                        y=lat,
                        c=estimation, cmap=cmap,
                        norm=norm,
                        s=s, alpha=alpha)

    estimation[estimation > estimation_limit_vals[1]] = estimation_limit_vals[1]
    estimation[estimation < estimation_limit_vals[0]] = estimation_limit_vals[0]
    grid_long, grid_lat = np.meshgrid(np.linspace(np.min(long), np.max(long), np.shape(img)[1]),
                                      np.linspace(np.min(lat), np.max(lat), np.shape(img)[0]))
    grid_z2 = griddata(np.transpose([long, lat]), estimation, (grid_long, grid_lat), method=method)
    if isinstance(levels, np.ndarray) or levels > 0:
        sc = ax.contourf(grid_z2, levels=levels, alpha=alpha, cmap=cmap,
                         norm=norm)
    else:
        sc = ax.imshow(grid_z2, extent=extent,
                       origin='lower', alpha=alpha, cmap=cmap,
                       norm=norm
                       # interpolation="bicubic",
                       # norm=colors.Normalize(vmin=0, vmax=max_val) if max_val is not None else None
                       )

    ax.imshow(img, extent=extent, alpha=1 - alpha)

    if bar:
        plt.colorbar(sc, ax=ax)
    ax.set_xticks(np.linspace(extent[0], extent[1], n_ticks),
                  np.round(np.linspace(long_bounds.lower, long_bounds.upper, n_ticks), decimals=2))
    ax.set_yticks(np.linspace(extent[2], extent[3], n_ticks),
                  np.round(np.linspace(lat_bounds.lower, lat_bounds.upper, n_ticks), decimals=2))


def plot_stations(station_coordinates, lat, long, color="red", marker="x", size=7, label=True):
    x = [np.argmin((l - long[0, :]) ** 2) for l in station_coordinates.T.long]
    y = [np.argmin((l - lat[:, 0]) ** 2) for l in station_coordinates.T.lat]
    plt.scatter(x, y, s=25, c=color, marker=marker, edgecolors="k")
    if label:
        for pos_x, pos_y, station_name in zip(x, y, station_coordinates.columns):
            plt.text(pos_x + 25, pos_y + 25, station_name, {'size': size, "color": color})


def plot_correlation_clustermap(corr, cmap="autumn", linewidths=0.5, annot=True, annot_size=10, figsize=(10, 10)):
    g = sns.clustermap(corr, cmap=cmap, figsize=figsize, annot=annot)
    mask = np.tril(np.ones_like(corr))
    # apply the inverse permutation to the mask
    mask = mask[np.argsort(g.dendrogram_row.reordered_ind), :]
    mask = mask[:, np.argsort(g.dendrogram_col.reordered_ind)]
    # run the clustermap again with the new mask
    g = sns.clustermap(corr, figsize=figsize,
                       linewidths=linewidths, cmap=cmap, annot=annot,
                       mask=mask, annot_kws={"size": annot_size})


FillBetweenInfo = namedtuple("FillBetweenInfo",
                             ["model1", "model2", "model3", "model4", "color_low", "color_middle", "color_high",
                              "alpha"])


def plot_errors_vertical(data, x, y, hue, ax, y_order=None, model_style=None, fill_between: FillBetweenInfo = None,
                map_names=None, *args, **kwargs):
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
    # data.sort_values(by=map_names.keys(), ascending=False, inplace=True)
    for method in (map_names.keys() if map_names is not None else np.unique(data[hue])):
        df = data.loc[data[hue] == method]
        # for method, df in data.groupby(hue, sort=False):
        df.set_index(y, inplace=True, drop=True)
        df = df if y_order is None else df.loc[y_order, :]
        if model_style[method].linestyle is not None or model_style[method].linewidth is not None:
            sns.lineplot(
                x=df[x], y=df.index,
                label=method if map_names is None else map_names[method],
                ax=ax, alpha=1,
                color=model_style[method].color if model_style is not None else None,
                marker=model_style[method].marker if model_style is not None else None,
                linestyle=model_style[method].linestyle if model_style is not None else None,
                linewidth=model_style[method].linewidth if model_style is not None else None,
                **kw
            )
        else:
            sns.scatterplot(
                x=df[x], y=df.index, label=method, ax=ax, alpha=1,
                color=model_style[method].color if model_style is not None else None,
                marker=model_style[method].marker if model_style is not None else None,
                size=model_style[method].size if model_style is not None else None,
                # **kw
            )


def plot_errors(data, x, y, hue, ax, stations_order=None, model_style=None, fill_between: FillBetweenInfo = None,
                map_names=None, *args, **kwargs):
    # plot regions
    if fill_between is not None:
        if (fill_between.model1 is not None) and (fill_between.model1 in data[hue].values):
            df1 = data.loc[data[hue] == fill_between.model1].set_index(x, drop=True, inplace=False)
            df1 = df1 if stations_order is None else df1.loc[stations_order, :]
            ax.fill_between(x=df1.index, y1=kwargs.get("xlim", (0, None))[0], y2=df1[y],
                            color=fill_between.color_low + (fill_between.alpha,))

        if fill_between.model2 is not None and fill_between.model2 in data[hue].values:
            df2 = data.loc[data[hue] == fill_between.model2].set_index(x, drop=True, inplace=False)
            df2 = df2 if stations_order is None else df2.loc[stations_order, :]
            ax.fill_between(x=df2.index, y1=df2[y], y2=kwargs.get("xlim", (0, max(data[y])))[1] * 1.1,
                            color=fill_between.color_high + (fill_between.alpha,))

        if fill_between.model3 is not None and fill_between.model3 in data[hue].values:
            df3 = data.loc[data[hue] == fill_between.model3].set_index(x, drop=True, inplace=False)
            df3 = df3 if stations_order is None else df3.loc[stations_order, :]
            ax.fill_between(x=df3.index, y1=df3[y], y2=kwargs.get("xlim", (0, max(data[y])))[1] * 1.1,
                            color=fill_between.color_middle + (fill_between.alpha,))

        if fill_between.model4 is not None and fill_between.model4 in data[hue].values:
            df4 = data.loc[data[hue] == fill_between.model4].set_index(x, drop=True, inplace=False)
            df4 = df4 if stations_order is None else df4.loc[stations_order, :]
            ax.fill_between(x=df4.index, y1=kwargs.get("xlim", (0, None))[0], y2=df4[y],
                            color=fill_between.color_middle + (fill_between.alpha,))

    # plot models
    ins = inspect.getfullargspec(sns.lineplot)
    kw = filter_dict(ins.args + ins.kwonlyargs, kwargs)
    # data.sort_values(by=map_names.keys(), ascending=False, inplace=True)
    for method in (map_names.keys() if map_names is not None else np.unique(data[hue])):
        df = data.loc[data[hue] == method]
        # for method, df in data.groupby(hue, sort=False):
        df.set_index(x, inplace=True, drop=True)
        df = df if stations_order is None else df.loc[stations_order, :]

        ax.scatter(
            x=df.index, y=df[y],
            c=model_style[method].color if model_style is not None else None,
            marker=model_style[method].marker if model_style is not None else None,
            s=model_style[method].size if model_style is not None else None,
            # **kw
        )

        sns.lineplot(
            x=df.index, y=df[y],
            label=method if map_names is None else map_names[method],
            ax=ax, alpha=1,
            color=model_style[method].color if model_style is not None else None,
            # marker=model_style[method].marker if model_style is not None else None,
            linestyle=model_style[method].linestyle if model_style is not None else None,
            linewidth=model_style[method].linewidth if model_style is not None else None,
            # size=model_style[method].size if model_style is not None else None,
            **kw
        )
