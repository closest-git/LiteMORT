'''
    http://www.cnblogs.com/amazement/p/10341328.html
    包含相对引用的 module，不要直接利用  解释器执行(如果直接执行，这个文件名.py 对应的module __name__ 值就是  '__main__')
'''
import gc
import numpy as np
import pandas as pd
from ctypes import *
from sklearn import preprocessing
import os
from sklearn.datasets import (load_boston, load_breast_cancer, load_digits,load_iris, load_svmlight_file)
from sklearn.metrics import log_loss, mean_squared_error
from sklearn.model_selection import GridSearchCV, train_test_split
from LiteMORT import *

if __name__ == "__main__":
    rng = np.random.RandomState(1994)
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
        # xgb_model = xgb.XGBRegressor().fit(X[train_index], y[train_index])
        mort = LiteMORT(params)
        mort.fit(X[train_index], y[train_index], params=params)
        preds = mort.predict(X[test_index])
        labels = y[test_index]
        assert mean_squared_error(preds, labels) < 25

    params = {
        "objective": "binary", "metric": "logloss", 'early_stop': 5, 'num_boost_round': 50,
        "verbosity": 1,'subsample':1,
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
        #prepocess = Mort_Preprocess()
        X, X_test = Mort_Preprocess.OrdinalEncode_(X, X_test)
    for col in ["A", "B", "C", "D"]:
        X[col] = X[col].astype('category')
        X_test[col] = X_test[col].astype('category')
    mort0 = LiteMORT(params).fit(X, y)
    pred0 = list(mort0.predict(X_test))
    #del mort0
    #gc.collect()
    if True:
        mort1 = LiteMORT(params).fit(X, y, categorical_feature=[0])
        pred1 = list(mort1.predict(X_test))
        mort2 = LiteMORT(params).fit(X, y, categorical_feature=['A'])
        pred2 = list(mort2.predict(X_test))
        mort3 = LiteMORT(params).fit(X, y, categorical_feature=['A', 'B', 'C', 'D'])
        pred3 = list(mort3.predict(X_test))
        #np.testing.assert_almost_equal(pred1, pred1)
        np.testing.assert_almost_equal(pred1, pred2)
        #np.testing.assert_almost_equal(pred1, pred3)
        input("...")
    #ret = log_loss(y_test, mort.predict_proba(X_test))