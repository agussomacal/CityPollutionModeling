import numpy as np
import pandas as pd

from src.lib.Modules import ParametricModule


class FeatureExtractor(ParametricModule):
    def extract_features(self, times, positions: pd.DataFrame, *args, **kwargs) -> np.ndarray:
        raise Exception("Not implemented.")
