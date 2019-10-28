
from sklearn import skbase
import numpy as np

#https://www.kaggle.com/c/ashrae-energy-prediction/discussion/113784#latest-656376
class DatetimeConvertCyclical(skbase.BaseEstimator, skbase.TransformerMixin):
    def __init__(self):
        self.time_periods = {'second': 24 * 60 * 60,
                             'minute': 24 * 60,
                             'hour': 24,
                             'day': 30,
                             'dayofweek': 7,
                             'month': 12}

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        for period, value in self.time_periods.items():
            X[period] = getattr(X['timestamp'].dt, period)

            X['sin_' + period] = np.sin(2 * np.pi * X[period] / value)
            X['cos_' + period] = np.cos(2 * np.pi * X[period] / value)

            X.drop(str(period), axis=1, inplace=True)

        return X