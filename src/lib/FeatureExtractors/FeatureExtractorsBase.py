from PerplexityLab.miscellaneous import if_true_str
from src.lib.Modules import ParametricModule


class FeatureExtractor(ParametricModule):
    def __init__(self, name="", **kwargs):
        super().__init__(name=f'FE{name}', **kwargs)

    def extract_features(self, *args, **kwargs):
        raise Exception("Not implemented.")
