# coding: utf-8
# pylint: skip-file
import math
import os
import unittest

from litemort import (LiteMORT,Mort_Preprocess)
import lightgbm as lgb
import numpy as np
from sklearn.base import clone
from sklearn import preprocessing
from sklearn.datasets import (load_boston, load_breast_cancer, load_digits,
                              load_iris, load_svmlight_file)
from sklearn.externals import joblib
from sklearn.metrics import log_loss, mean_squared_error
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.utils.estimator_checks import (_yield_all_checks, SkipTest,
                                            check_parameters_default_constructible)
isMORT=True

try:
    from sklearn.utils.estimator_checks import check_no_fit_attributes_set_in_init
    sklearn_at_least_019 = True
except ImportError:
    sklearn_at_least_019 = False


def multi_error(y_true, y_pred):
    return np.mean(y_true != y_pred)


def multi_logloss(y_true, y_pred):
    return np.mean([-math.log(y_pred[i][y]) for i, y in enumerate(y_true)])


class TestSklearn(unittest.TestCase):

    def test_binary_breast(self):
        params = {
            "objective": "binary", "metric": "logloss",'early_stop': 5, 'num_boost_round': 50,
            "verbosity": 1,
        }
        X, y = load_breast_cancer(True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
        if isMORT:
            mort = LiteMORT(params)
            mort.fit(X_train, y_train, eval_set=[(X_test, y_test)], params=params)
            result = mort.predict(X_test)
            ret = log_loss(y_test, mort.predict_proba(X_test))
        else:
            gbm = lgb.LGBMClassifier(n_estimators=50, silent=True)
            gbm.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=5, verbose=False)
            result = gbm.predict(X_test)
            ret = log_loss(y_test, gbm.predict_proba(X_test))
            self.assertAlmostEqual(ret, gbm.evals_result_['valid_0']['binary_logloss'][gbm.best_iteration_ - 1],places=5)
        self.assertLess(ret, 0.15)

    def test_binary_digits(self):
        from sklearn.datasets import load_digits
        from sklearn.model_selection import KFold
        rng = np.random.RandomState(1994)
        params = {
            "objective": "binary", "metric": "logloss", 'early_stop': 5, 'num_boost_round': 50,
            "verbosity": 1,
        }
        digits = load_digits(2)
        y = digits['target']
        X = digits['data']
        kf = KFold(n_splits=2, shuffle=True, random_state=rng)
        for train_index, test_index in kf.split(X, y):
            #xgb_model = cls(random_state=42).fit(X[train_index], y[train_index])
            #xgb_model.predict(X[test_index])
            mort = LiteMORT(params).fit(X[train_index], y[train_index])
            preds = mort.predict(X[test_index])
            labels = y[test_index]
            err = sum(1 for i in range(len(preds))
                      if int(preds[i] > 0.5) != labels[i]) / float(len(preds))
            assert err < 0.1


    def test_regression(self):
        params = {
            "objective": "regression",  'early_stop': 5, 'num_boost_round': 50, "verbosity": 1,
        }
        X, y = load_boston(True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
        if isMORT:
            mort = LiteMORT(params)
            mort.fit(X_train, y_train, eval_set=[(X_test, y_test)], params=params)
            ret = mean_squared_error(y_test, mort.predict(X_test))
        else:
            gbm = lgb.LGBMRegressor(n_estimators=50, silent=True)
            gbm.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=5, verbose=False)
            ret = mean_squared_error(y_test, gbm.predict(X_test))
            self.assertAlmostEqual(ret, gbm.evals_result_['valid_0']['l2'][gbm.best_iteration_ - 1], places=5)
        self.assertLess(ret, 16)

    def test_regression_boston_housing(self):
        rng = np.random.RandomState(1994)
        params = {
            "objective": "regression", 'early_stop': 5, 'num_boost_round': 50, "verbosity": 1,
        }
        from sklearn.metrics import mean_squared_error
        from sklearn.datasets import load_boston
        from sklearn.model_selection import KFold
        params = {
            "objective": "regression", 'early_stop': 5, 'num_boost_round': 50, "verbosity": 1,
        }
        boston = load_boston()
        y = boston['target']
        X = boston['data']
        kf = KFold(n_splits=2, shuffle=True, random_state=rng)
        for train_index, test_index in kf.split(X, y):
            #xgb_model = xgb.XGBRegressor().fit(X[train_index], y[train_index])
            mort = LiteMORT(params)
            mort.fit(X[train_index], y[train_index], params=params)
            preds = mort.predict(X[test_index])
            labels = y[test_index]
            assert mean_squared_error(preds, labels) < 25

    
    #@unittest.skipIf(not litemort.combat.PANDAS_INSTALLED, 'pandas is not installed')
    def test_pandas_categorical(self):
        params = {
            "objective": "binary", "metric": "logloss", 'early_stop': 5, 'num_boost_round': 50,
            "verbosity": 1,
        }
        import pandas as pd
        X = pd.DataFrame({"A": np.random.permutation(['a', 'b', 'c', 'd'] * 75),  # str
                          "B": np.random.permutation([1, 2, 3] * 100),  # int
                          "C": np.random.permutation([0.1, 0.2, -0.1, -0.1, 0.2] * 60),  # float
                          "D": np.random.permutation([True, False] * 150)})  # bool
        y = np.random.permutation([0, 1] * 150)
        X_test = pd.DataFrame({"A": np.random.permutation(['a', 'b', 'e'] * 20),
                               "B": np.random.permutation([1, 3] * 30),
                               "C": np.random.permutation([0.1, -0.1, 0.2, 0.2] * 15),
                               "D": np.random.permutation([True, False] * 30)})
        if True:
            X, X_test = Mort_Preprocess.OrdinalEncode_(X,X_test)
        for col in ["A", "B", "C", "D"]:
            X[col] = X[col].astype('category')
            X_test[col] = X_test[col].astype('category')
        #trn_data = lgb.Dataset(X, label=y)

        if isMORT:
            mort0 = LiteMORT(params).fit(X, y)
            pred0 = list(mort0.predict(X_test))
            mort1 = LiteMORT(params).fit(X, y, categorical_feature=[0])
            pred1 = list(mort1.predict(X_test))
            mort2 = LiteMORT(params).fit(X, y, categorical_feature=['A'])
            pred2 = list(mort2.predict(X_test))
            mort3 = LiteMORT(params).fit(X, y, categorical_feature=['A', 'B', 'C', 'D'])
            pred3 = list(mort3.predict(X_test))
        else:
            clf=lgb.sklearn.LGBMClassifier()
            gbm_ = clf.fit(X, y)
            gbm0 = lgb.sklearn.LGBMClassifier().fit(X, y)
            pred0 = list(gbm0.predict(X_test))
            gbm1 = lgb.sklearn.LGBMClassifier().fit(X, y, categorical_feature=[0])
            pred1 = list(gbm1.predict(X_test))
            gbm2 = lgb.sklearn.LGBMClassifier().fit(X, y, categorical_feature=['A'])
            pred2 = list(gbm2.predict(X_test))
            gbm3 = lgb.sklearn.LGBMClassifier().fit(X, y, categorical_feature=['A', 'B', 'C', 'D'])
            pred3 = list(gbm3.predict(X_test))
            gbm3.booster_.save_model('categorical.model')
            gbm4 = lgb.Booster(model_file='categorical.model')
            pred4 = list(gbm4.predict(X_test))
            pred_prob = list(gbm0.predict_proba(X_test)[:, 1])
            np.testing.assert_almost_equal(pred_prob, pred4)
        #np.testing.assert_almost_equal(pred0, pred1)
        #np.testing.assert_almost_equal(pred0, pred2)
        #np.testing.assert_almost_equal(pred0, pred3)




