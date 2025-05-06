from numba.np.arraymath import check_array
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class StandardScaler(BaseEstimator, TransformerMixin):
    def __init__(self, with_mean=True, with_std=True):
        self.with_mean = with_mean

    def fit(self, X, y=None):
        X = check_array(X)

        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.n_features_ = X.shape[1]

    def transform(self, X):
        check_is_fitted(self)
        X = check_array(X)

        assert self.n_features_ == X.shape[1]

        if self.with_mean:
            X = X - self.mean_

        return X / self.scale

