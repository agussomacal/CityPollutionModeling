import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from PerplexityLab.miscellaneous import partial_filter, if_true_str
from src.lib.FeatureExtractors.FeatureExtractorsBase import FeatureExtractor
from src.lib.Models.TrueStateEstimationModels.TrafficConvolution import gaussker

WaterColor = (156, 192, 249)
GreenAreaColor = (168, 218, 181)


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

    def get_distance(self, positions: pd.DataFrame):
        for point in map(tuple, positions.values.T):
            if point not in self.dist:
                self.dist[tuple(point)] = cdist(XA=self.coords, XB=np.reshape(point, (1, 2)), metric=self.metric)
            yield np.squeeze(self.dist[tuple(point)])

    def extract_features(self, times, positions: pd.DataFrame, mask=None, *args, **kwargs) -> np.ndarray:
        """

        :param positions: Dataframe with [x y] two rows and as many possitions as columns.
        :param mask: which points are to be considered: None means all or a matrix of Bool of size #times x #coords
        :param args:
        :param kwargs:
        :return:
        """
        if mask is None: mask = np.ones((len(times), len(self.coords)))
        kernel2apply = partial_filter(self.kernel, **self.params)
        features_per_coord = []
        for distances_to_coord_i in self.get_distance(positions):
            k = kernel2apply(distances_to_coord_i)
            features_per_coord.append(self.agg_func(mask * k[np.newaxis, :], axis=1))
            if self.normalize: features_per_coord[-1] /= self.agg_func(k)
        return np.transpose(features_per_coord)


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
