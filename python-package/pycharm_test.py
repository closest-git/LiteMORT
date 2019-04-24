'''
    http://www.cnblogs.com/amazement/p/10341328.html
    包含相对引用的 module，不要直接利用  解释器执行(如果直接执行，这个文件名.py 对应的module __name__ 值就是  '__main__')
'''
import gc
import numpy as np
import pandas as pd
from sklearn import preprocessing
import os
from sklearn.datasets import (load_boston, load_breast_cancer, load_digits,load_iris, load_svmlight_file)
import time
import pickle
from sklearn.metrics import log_loss, mean_squared_error
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, train_test_split
import shap
from LiteMORT import *
import lightgbm as lgb
from sklearn import metrics
isMORT=True

def auc2(m, train, test,y_train,y_test):
    return (metrics.roc_auc_score(y_train,m.predict(train)),
                            metrics.roc_auc_score(y_test,m.predict(test)))

# https://www.kdnuggets.com/2018/03/catboost-vs-light-gbm-vs-xgboost.html
def test_fly_( ):
    import pandas as pd, numpy as np, time
    from sklearn.model_selection import train_test_split
    frac=0.1
    pkl_path = 'G:/kaggle/flight/flight_{}.pickle'.format(frac)
    if os.path.isfile(pkl_path):
        with open(pkl_path, "rb") as fp:  # Pickling
            [data] = pickle.load(fp)
    else:
        data = pd.read_csv("G:/kaggle/flight/flights.csv")
        data = data.sample(frac=frac, random_state=10)
        data = data[["MONTH", "DAY", "DAY_OF_WEEK", "AIRLINE", "FLIGHT_NUMBER", "DESTINATION_AIRPORT",
                     "ORIGIN_AIRPORT", "AIR_TIME", "DEPARTURE_TIME", "DISTANCE", "ARRIVAL_DELAY"]]
        data.dropna(inplace=True)
        data["ARRIVAL_DELAY"] = (data["ARRIVAL_DELAY"] > 10) * 1
        with open(pkl_path, "wb") as fp:  # Pickling
            pickle.dump([data], fp)
        os._exit(-1)

    cols = ["AIRLINE", "FLIGHT_NUMBER", "DESTINATION_AIRPORT", "ORIGIN_AIRPORT"]
    for item in cols:
        data[item] = data[item].astype("category").cat.codes + 1
    train, test, y_train, y_test = train_test_split(data.drop(["ARRIVAL_DELAY"], axis=1), data["ARRIVAL_DELAY"], random_state=10, test_size=0.25)

    if False:
        lg = lgb.LGBMClassifier(silent=False)
        param_dist = {"max_depth": [25,50, 75],
                      "learning_rate" : [0.01,0.05,0.1],
                      "num_leaves": [300,900,1200],
                      "n_estimators": [200]
                     }
        grid_search = GridSearchCV(lg, n_jobs=-1, param_grid=param_dist, cv = 3, scoring="roc_auc", verbose=5)
        grid_search.fit(train,y_train)
        grid_search.best_estimator_
    params = {  "objective": "binary",
                "metric": "binary_logloss",#""binary_logloss",
              "max_depth": 50, "learning_rate": 0.1, "num_leaves": 900, "n_estimators": 300}
    cate_features_name = ["MONTH", "DAY", "DAY_OF_WEEK", "AIRLINE", "DESTINATION_AIRPORT","ORIGIN_AIRPORT"]
    t0=time.time()
    if isMORT:
        model2 = LiteMORT(params).fit(train, y_train)
        y_predict = model2.predict(test,raw_score=True)[:,1]
        a1 = metrics.roc_auc_score(y_test,model2.predict(test,raw_score=True)[:,1])
        print("------ No categorical auc={}".format(a1))
        model2 = LiteMORT(params).fit(train, y_train, categorical_feature = cate_features_name)
        a2 = metrics.roc_auc_score(y_test,model2.predict(test,raw_score=True)[:,1])
        print("------ With Categorical auc={}".format(a2))
    else:
        d_train = lgb.Dataset(train, label=y_train,free_raw_data=False)
        # Without Categorical Features
        model2 = lgb.train(params, d_train)
        a1=auc2(model2, train, test,y_train,y_test)
        print("------ No categorical auc2={}".format(a1))

        #With Catgeorical Features
        model2 = lgb.train(params, d_train, categorical_feature = cate_features_name)
        a2=auc2(model2, train, test,y_train,y_test)
        print("------ With categorical auc2={}".format(a2))
        del d_train
        gc.collect()
    input("loss@test_fly_ is {} time={} model={}...".format(a2,time.time()-t0,model2))
    os._exit(-98)

def test_shap_adult_():
    shap.initjs()
    X,y = shap.datasets.adult()
    X_display,y_display = shap.datasets.adult(display=True)
    # create a train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)

    params = {
        "max_bin": 512,
        "learning_rate": 0.05,
        "boosting_type": "gbdt",
        "objective": "binary",
        "metric": "binary_logloss",
        "num_leaves": 10,
        "verbose": 1000,
        "min_data": 100,
        "boost_from_average": True,
        'early_stop': 50, 'num_boost_round': 10000,
    }
    if isMORT:
        model = LiteMORT(params).fit(X_train, y_train,eval_set=[(X_test,y_test)])
        result = model.predict(X_test)
        result = model.predict(X_test,raw_score=True)
    elif True:
        gbm = lgb.LGBMClassifier(n_estimators=10000, silent=True)
        gbm.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=50, verbose=False)
        result = gbm.predict(X_test)        #predict_proba+_le.inverse_transform
        result = gbm.predict_proba(X_test)
    else:   #晕!!! LGBMClassifier和lgb.train返回结果不一样
        d_train = lgb.Dataset(X_train, label=y_train)
        d_test = lgb.Dataset(X_test, label=y_test)
        model = lgb.train(params, d_train, 10000, valid_sets=[d_test], early_stopping_rounds=50, verbose_eval=1000)
        if False:#https://slundberg.github.io/shap/notebooks/Census%20income%20classification%20with%20LightGBM.html
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)
            shap.force_plot(explainer.expected_value, shap_values[0, :], X_display.iloc[0, :])
            shap.force_plot(explainer.expected_value, shap_values[:1000, :], X_display.iloc[:1000, :])
            shap.summary_plot(shap_values, X)
            plt.show()
        result = model.predict(X_test)
    loss = log_loss(y_test, result)
    input("loss@test_shap_adult_ is {} model={}...".format(loss,model))
    os._exit(-99)

def test_1():
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
        "verbosity": 1, 'subsample': 1,
    }
if __name__ == "__main__":
    test_fly_()
    #test_shap_adult_()
    nTree=100   #100

    rng = np.random.RandomState(1994)
    np.random.seed(42)
    params = {
        "objective": "binary", "metric": "logloss", 'early_stop': 5, 'num_boost_round': nTree,
        "verbosity": 1,
    }
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
    '''
    for col in ["A", "B", "C", "D"]:
        X[col] = X[col].astype('category')
        X_test[col] = X_test[col].astype('category')
    '''

    if True:
        isLabel=True
        gbm0 = lgb.sklearn.LGBMClassifier(n_estimators = nTree).fit(X, y)
        gbm0.booster_.save_model('gbm0.model')
        result = gbm0.predict(X_test,raw_score=isLabel,n_estimators = nTree)
        pred0 = list(gbm0.predict(X_test,raw_score=isLabel))
        gbm1 = lgb.sklearn.LGBMClassifier(n_estimators = nTree).fit(X, y, categorical_feature=[0])
        gbm1.booster_.save_model('gbm1.model')
        pred1 = list(gbm1.predict(X_test,raw_score=isLabel))
        gbm2 = lgb.sklearn.LGBMClassifier(n_estimators = nTree).fit(X, y, categorical_feature=['A'])
        pred2 = list(gbm2.predict(X_test,raw_score=isLabel))
        gbm3 = lgb.sklearn.LGBMClassifier(n_estimators = nTree).fit(X, y, categorical_feature=['A', 'B', 'C', 'D'])
        pred3 = list(gbm3.predict(X_test,raw_score=isLabel))
        np.testing.assert_almost_equal(pred0, pred1)
        np.testing.assert_almost_equal(pred0, pred2)
        np.testing.assert_almost_equal(pred0, pred3)
        '''
        gbm3.booster_.save_model('categorical.model')
        gbm4 = lgb.Booster(model_file='categorical.model')
        pred4 = list(gbm4.predict(X_test))
        pred_prob = list(gbm0.predict_proba(X_test)[:, 1])
        np.testing.assert_almost_equal(pred_prob, pred4)
        '''

    if False:
        mort0 = LiteMORT(params).fit(X, y)
        pred0 = list(mort0.predict(X_test))
    else:
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
    # gc.collect()
    #ret = log_loss(y_test, mort.predict_proba(X_test))