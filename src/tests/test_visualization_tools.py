import unittest

import numpy as np
from matplotlib import pyplot as plt

from PerplexityLab.visualization import save_fig
from src.config import tests_dir
from src.lib.visualization_tools import plot_estimation_map_in_graph


class TestDataManager(unittest.TestCase):
    def test_plot_estimation_map_in_graph(self):
        with save_fig(paths=tests_dir, filename="test_plot_estimation_map_in_graph"):
            fig, ax = plt.subplots()
            m = 1000
            long = np.random.uniform(low=2.3, high=2.4, size=(m,))
            lat = np.random.uniform(low=48.82, high=48.89, size=(m,))
            estimation = np.exp(-((long - np.mean(long)) ** 2 + 2*(lat - np.mean(lat)) ** 2) / 0.05)

            img = np.ones((800, 1000, 3))
            plot_estimation_map_in_graph(ax, long, lat, estimation, img, cmap='RdGy', s=20, alpha=0.5,
                                         bar=True, estimation_limit_vals=(0.05, 0.95), levels=10)
