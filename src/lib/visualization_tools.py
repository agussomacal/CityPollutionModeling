import numpy as np
from matplotlib import colors, pylab as plt
from scipy.interpolate import griddata
import seaborn as sns

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
                        c=norm(estimation), cmap=cmap,
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
