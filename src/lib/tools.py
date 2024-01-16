import numpy as np
from sklearn.linear_model import LinearRegression


class IdentityTransformer(LinearRegression):
    def __init__(self):
        super().__init__(fit_intercept=False)

    def fit(self, X, y, sample_weight=None):
        self.coef_ = np.ones(np.shape(X)[1])
        self.intercept_ = 0
        return self
