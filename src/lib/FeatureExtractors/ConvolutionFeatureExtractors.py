import numpy as np
from scipy.spatial.distance import cdist

from PerplexityLab.miscellaneous import partial_filter, if_true_str
from src.lib.DataProcessing.TrafficProcessing import TRAFFIC_VALUES
from src.lib.FeatureExtractors.FeatureExtractorsBase import FeatureExtractor
from src.lib.Models.TrueStateEstimationModels.TrafficConvolution import gaussker


class FEConvolution(FeatureExtractor):
    def __init__(self, name, coords, kernel=gaussker, normalize=False, agg_func=np.sum, metric="euclidean", **kwargs):
        """

        :param mask: image shape with True False marking the pixels belonging to the category of interest.
        :param pixel_coordinates: [image_shape x [lat]
        """
        super().__init__(name=f"{name}{kernel.__name__}{if_true_str(normalize, 'norm')}{agg_func}{metric}", **kwargs)
        self.coords = coords
        self.kernel = kernel
        self.normalize = normalize
        self.agg_func = agg_func
        self.metric = metric
        self.dist = dict()

    def get_distance(self, spatial_coords: np.ndarray):
        for point in map(tuple, spatial_coords):
            if point not in self.dist:
                self.dist[tuple(point)] = cdist(XA=self.coords, XB=np.reshape(point, (1, 2)), metric=self.metric)
            yield self.dist[tuple(point)]

    def extract_features(self, spatial_coords, mask=None, *args, **kwargs):
        """

        :param spatial_coords: List of spatial coordinates. num points x spatial dimension
        :param mask: which points are to be considered.
        :param args:
        :param kwargs:
        :return:
        """
        if mask is None: mask = np.array([True] * len(self.coords))
        kernel2apply = partial_filter(self.kernel, **self.params)
        features_per_coord = []
        for distances_to_coord_i in self.get_distance(spatial_coords):
            features_per_coord.append(self.agg_func(kernel2apply(distances_to_coord_i[mask])))
            if self.normalize: features_per_coord[-1] /= self.agg_func(distances_to_coord_i)
        return features_per_coord


class FEConvolutionFixedPixels(FEConvolution):
    def __init__(self, name, mask, x_coords, y_coords, kernel=gaussker, normalize=False, agg_func=np.sum,
                 metric="euclidean", **kwargs):
        """

        :param mask: image shape with True False marking the pixels belonging to the category of interest.
        :param pixel_coordinates: [image_shape x [lat]
        """
        super().__init__(name=name, coords=np.transpose([x_coords[mask], y_coords[mask]]),
                         kernel=kernel, normalize=normalize, agg_func=agg_func, metric=metric, **kwargs)


class FEConvolutionTraffic(FEConvolution):
    def __init__(self, name, x_coords, y_coords, kernel=gaussker, normalize=False, agg_func=np.sum,
                 metric="euclidean", **kwargs):
        """

        :param mask: image shape with True False marking the pixels belonging to the category of interest.
        :param pixel_coordinates: [image_shape x [lat]
        """
        super().__init__(name=name, coords=np.transpose([x_coords, y_coords]),
                         kernel=kernel, normalize=normalize, agg_func=agg_func, metric=metric, **kwargs)
class FETrafficConvolution(FeatureExtractor):
    def __init__(self, kernel=gaussker, normalize=False, **kwargs):
        super().__init__(**kwargs)
        self.kernel = kernel
        self.normalize = normalize
        for k in TRAFFIC_VALUES.keys():
            if k not in kwargs.keys():
                kwargs[k] = 0
        super(FETrafficConvolution, self).__init__(name=f"{kernel.__name__}{if_true_str(normalize, 'Norm')}", **kwargs)

    def extract_features(self, traffic, target_positions, traffic_coords, distance_between_stations_pixels, *args,
                         **kwargs) -> pd.DataFrame:
        """
        traffic_coords: pd.DataFrame with columns the pixel_coord and rows 'lat' and 'long' associated to the pixel
        target_positions: pd.DataFrame with columns the name of the station and rows 'lat' and 'long'
        """
        if target_positions.columns[0] in distance_between_stations_pixels.index:
            dist = distance_between_stations_pixels.iloc[
                   list(distance_between_stations_pixels.index).index(target_positions.columns[0]), :]
            # dist = distance_between_stations_pixels.loc[target_positions.columns[0], traffic_coords.columns]
            # dist = distance_between_stations_pixels.loc[target_positions.columns[0], :].values
        else:
            dist = np.sqrt(
                ((traffic_coords.loc[["long", "lat"], :] -
                  target_positions.loc[["long", "lat"], :].values.reshape((2, 1)))
                 ** 2).sum())
        kernel = partial_filter(dist=dist, **self.params)(self.conv_kernel)
        reduced_traffic = reduce_traffic_to_colors(traffic, kernel=kernel)
        if self.normalize:
            reduced_traffic /= np.sum(kernel)
        return reduced_traffic

    def state_estimation(self, observed_stations, observed_pollution, traffic, target_positions: pd.DataFrame,
                         traffic_coords, distance_between_stations_pixels, **kwargs) -> np.ndarray:
        """
        traffic_coords: pd.DataFrame with columns the pixel_coord and rows 'lat' and 'long' associated to the pixel
        target_positions: pd.DataFrame with columns the name of the station and rows 'lat' and 'long'
        """
        estimated_average_pollution = []
        for name, tp in target_positions.items():
            reduced_traffic = self.convolve(traffic=traffic, target_positions=pd.DataFrame(tp),
                                            traffic_coords=traffic_coords,
                                            distance_between_stations_pixels=distance_between_stations_pixels)
            estimated_average_pollution.append(
                reduced_traffic @ pd.Series(filter_dict(reduced_traffic.columns, self.params)))
        return pd.concat(estimated_average_pollution, axis=1).values

    @pollution_agnostic
    def state_estimation_for_optim(self, observed_stations, observed_pollution, traffic,
                                   distance_between_stations_pixels, **kwargs) -> [np.ndarray,
                                                                                   np.ndarray]:
        """
        traffic_coords: pd.DataFrame with columns the pixel_coord and rows 'lat' and 'long' associated to the pixel
        target_positions: pd.DataFrame with columns the name of the station and rows 'lat' and 'long'
        """
        raise Exception("Reimplement with pollution_agnostic decorator.")
        # TODO: no need for cross validation because no pollution is used to infer
        target_pollution, reduced_traffic = \
            list(zip(*[(target_pollution,
                        partial_filter(distance_between_stations_pixels=distance_between_stations_pixels,
                                       **{**kwargs, **known_data})(self.convolve))
                       for known_data, target_pollution in
                       loo(observed_stations, observed_pollution, traffic, kwargs.get("stations2test", None))]))
        target_pollution = pd.concat(target_pollution)
        reduced_traffic = pd.concat(reduced_traffic)
        # use median to fit to avoid outliers to interfere
        lr = LinearRegression(fit_intercept=False).fit(np.median(reduced_traffic.values, axis=0, keepdims=True),
                                                       np.mean(target_pollution.values, axis=0, keepdims=True))
        # lr = LinearRegression(fit_intercept=False).fit(reduced_traffic.values, target_pollution)
        self.set_params(**dict(zip(reduced_traffic.columns, lr.coef_)))
        state_estimation = reduced_traffic @ pd.Series(filter_dict(reduced_traffic.columns, self.params))
        return state_estimation.values, target_pollution
