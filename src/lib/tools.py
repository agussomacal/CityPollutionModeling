import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression


# class IdentityTransformer(BaseEstimator, TransformerMixin):
#     def __init__(self):
#         pass
#
#     def fit(self, input_array, y=None):
#         return self
#
#     def transform(self, input_array, y=None):
#         return input_array * 1
#
#     def predict(self, X):
#         return X


class IdentityTransformer(LinearRegression):
    def __init__(self):
        super().__init__(fit_intercept=False)

    def fit(self, X, y, sample_weight=None):
        self.coef_ = np.ones(np.shape(X)[1])
        self.intercept_ = 0
        return self
