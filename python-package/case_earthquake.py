# https://www.kaggle.com/tocha4/lanl-master-s-approach

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import scipy as sc
import matplotlib.pyplot as plt
import seaborn as sns
import gc
import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)
from tqdm import tqdm_notebook
import datetime
import time
import random
from joblib import Parallel, delayed


import lightgbm as lgb
from tensorflow import keras
from gplearn.genetic import SymbolicRegressor
#from catboost import Pool, CatBoostRegressor
from litemort import *
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error,mean_squared_error
from sklearn.model_selection import GridSearchCV, KFold, RandomizedSearchCV
from sklearn.feature_selection import RFECV, SelectFromModel
import os
import sys
import pickle
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import NuSVR, SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor

today = datetime.date.today().strftime('%m%d')

isMORT = len(sys.argv)>1 and sys.argv[1] == "mort"
#isMORT = True
#some_rows=3000
some_rows=None
model_type='mort' if isMORT else 'lgb'
nVerbose = 500
pkl_path = 'G:/kaggle/Earthquake/data/anton_2_{}.pickle'.format(some_rows)
pkl_path = 'G:/kaggle/Earthquake/data/anton_cys0_{}.pickle'.format(some_rows)
eval_metric='l1'
min_error = mean_squared_error if eval_metric=='l1' else mean_absolute_error
params = {
            'n_estimators':50000,      #减少n_estimators 并不能控制overfit
        'early_stopping_rounds': 200,
        'num_leaves': 256,              #128
        #'max_bin':  64,
          'min_data_in_leaf': 32,       #79
          'objective': 'tweedie',    #'regression',
          'max_depth': -1,
          'learning_rate': 0.01,
          #"boosting": "gbdt",
          "bagging_freq": 5,
          "bagging_fraction": 1,#0.8126672064208567,   #0.8126672064208567,
          "bagging_seed": 11,
          "metric": 'mae',
          "verbosity": nVerbose,
          #'reg_alpha': 0.1302650970728192,
          #'reg_lambda': 0.3603427518866501,
          'colsample_bytree': 0.05
         }
print("params=\n{}\n".format(params))
submission = pd.read_csv('G:/kaggle/Earthquake/input/sample_submission.csv')

def Load_MoreDatas(paths):
    train_s=[]
    y_s=[]
    for path,nFile in paths:
        for i in range(nFile):
            path_X,path_y="{}/train_X_features_{}.csv".format(path,i+1),"{}/train_y_{}.csv".format(path,i+1)
            X_ = pd.read_csv(path_X)
            y_ = pd.read_csv(path_y, index_col=False,  header=None)
            train_s.append(X_)
            y_s.append(y_)
            print("X_[{}]@{}\ny_[{}]@{}".format(X_.shape,path_X,y_.shape,path_y))
    if len(train_s)>0:
        train_X = pd.concat(train_s, axis=0)
        y = pd.concat(y_s, axis=0)
    train_X = train_X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    print("Load_MoreDatas X_[{}] y_[{}]".format(train_X.shape,  y.shape))
    return train_X,y

if os.path.isfile(pkl_path):
    print("\n======load pickle file from {} ...".format(pkl_path))
    with open(pkl_path, "rb") as fp:  # Pickling
        [train_X, test_X, train_y] = pickle.load(fp)
    if some_rows is not None:
        train_X = train_X[:some_rows]
        test_X = test_X[:some_rows]
        train_y = train_y[:some_rows]
    print("\n======train_X={} test_X={} train_y={} \n".format(train_X.shape, test_X.shape, train_y.shape))
else:
    #train_X_2,y_2 = Load_MoreDatas([('G:/kaggle/Earthquake/data/cys/15000', 14),
    #                                ('G:/kaggle/Earthquake/data/cys/17000', 15)])
    train_X_0 = pd.read_csv("G:/kaggle/Earthquake/data/train_X_features_865_0.csv")
    train_X_1 = pd.read_csv("G:/kaggle/Earthquake/data/train_X_features_865_1.csv")
    y_0 = pd.read_csv("G:/kaggle/Earthquake/data/train_y_0.csv", index_col=False,  header=None)
    y_1 = pd.read_csv("G:/kaggle/Earthquake/data/train_y_1.csv", index_col=False,  header=None)
    train_X = pd.concat([train_X_0, train_X_1], axis=0)
    y = pd.concat([y_0, y_1], axis=0)

    train_X = train_X.reset_index(drop=True)
    print(train_X.shape)
    print(train_X.head())

    y = y.reset_index(drop=True)
    print(y[0].shape)
    train_y = pd.Series(y[0].values)
    test_X = pd.read_csv("G:/kaggle/Earthquake/data/test_X_features_10.csv")
    scaler = StandardScaler()
    train_columns = train_X.columns

    train_X[train_columns] = scaler.fit_transform(train_X[train_columns])
    test_X[train_columns] = scaler.transform(test_X[train_columns])
    with open(pkl_path, "wb") as fp:  # Pickling
        pickle.dump([train_X, test_X, train_y], fp)
    print("Save pickle file at {} train_X={} test_X={} train_y={}".format(pkl_path,train_X.shape, test_X.shape, train_y.shape))
    sys.exit(-2)

train_columns = train_X.columns
n_fold = 5      #n_fold=10 只是增加了过拟合，莫名其妙
folds = KFold(n_splits=n_fold, shuffle=True, random_state=42)

oof = np.zeros(len(train_X))
train_score = []
fold_idxs = []
# if PREDICTION:
predictions = np.zeros(len(test_X))

feature_importance_df = pd.DataFrame()
#run model
for fold_, (trn_idx, val_idx) in enumerate(folds.split(train_X,train_y.values)):
    t0=time.time()
    strLog = "fold {}".format(fold_)
    print(strLog)
    fold_idxs.append(val_idx)
    fold_importance_df = pd.DataFrame()
    fold_importance_df["Feature"] = train_columns

    X_train, X_valid = train_X[train_columns].iloc[trn_idx], train_X[train_columns].iloc[val_idx]
    y_train, y_valid = train_y.iloc[trn_idx], train_y.iloc[val_idx]
    if model_type == 'mort':
        params['objective'] = 'regression'
        # model = LiteMORT(params).fit(X_train, y_train, eval_set=[(X_valid, y_valid)])
        model = LiteMORT(params).fit_1(X_train, y_train, eval_set=[(X_valid, y_valid)])
    if model_type == 'cat':
        model = CatBoostRegressor(n_estimators=25000, verbose=-1, objective="MAE", loss_function="MAE", boosting_type="Ordered", task_type="GPU")
        model.fit(X_tr,
                  y_tr,
                  eval_set=[(X_val, y_val)],
    #               eval_metric='mae',
                  verbose=2500,
                  early_stopping_rounds=500)
    if model_type == 'lgb':
        model = lgb.LGBMRegressor(**params,  n_jobs=-1)#n_estimators=50000,
        model.fit(X_train, y_train,
                  eval_set=[(X_train, y_train), (X_valid, y_valid)], eval_metric='mae',
                  verbose=nVerbose, early_stopping_rounds=200)   #
        fold_importance_df["importance"] = model.feature_importances_[:len(train_columns)]
        fold_importance_df["fold"] = fold_ + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

    oof[val_idx] = model.predict(X_valid)
    fold_score = mean_absolute_error(oof[val_idx], y_valid)
    print("{}\tscore={:.4g} time={:.4g}".format(strLog,fold_score,time.time()-t0))

    #predictions
    predictions += model.predict(test_X[train_columns]) / folds.n_splits
    train_score.append(fold_score)

cv_score = mean_absolute_error(train_y, oof)
print(f"\n======After {n_fold} score = {cv_score:.3f}, CV_fold = {np.mean(train_score):.3f} | {np.std(train_score):.3f}", end=" ")




submission["time_to_failure"] = predictions
submission.to_csv(f'G:/kaggle/Earthquake/result/{model_type}_{today}_[{cv_score:.3f},{np.std(train_score):.3f}].csv', index=False)
submission.head()