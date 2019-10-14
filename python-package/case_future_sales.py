#https://www.kaggle.com/dhimananubhav/feature-engineering-xgboost

import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 100)

from itertools import product
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import seaborn as sns
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from xgboost import plot_importance
import time
import sys
import gc
import pickle
import random
from litemort import *

from bayes_opt import BayesianOptimization


def plot_features(booster, figsize):
    fig, ax = plt.subplots(1,1,figsize=figsize)
    return plot_importance(booster=booster, ax=ax)

isMORT = len(sys.argv)>1 and sys.argv[1] == "mort"
isMORT = True
alg='MORT' if isMORT else 'LGB'
#some_rows = 5000
some_rows = None
#data_root = '../input/'
data_root = "F:/Datasets/future_sales"

test  = pd.read_csv(f'{data_root}/test.csv').set_index('ID')
data = pd.read_pickle(f'{data_root}/data.pkl')
if some_rows is not None:
    nMost=data.shape[0]
    random.seed(42)
    subset = random.sample(range(nMost), some_rows)
    data = data.iloc[subset, :].reset_index(drop=True)
    print('====== Some Samples ... data={}'.format(data.shape))

data = data[[
    'date_block_num',
    'shop_id',
    'item_id',
    'item_cnt_month',
    'city_code',
    'item_category_id',
    'type_code',
    'subtype_code',
    'item_cnt_month_lag_1',
    'item_cnt_month_lag_2',
    'item_cnt_month_lag_3',
    'item_cnt_month_lag_6',
    'item_cnt_month_lag_12',
    'date_avg_item_cnt_lag_1',
    'date_item_avg_item_cnt_lag_1',
    'date_item_avg_item_cnt_lag_2',
    'date_item_avg_item_cnt_lag_3',
    'date_item_avg_item_cnt_lag_6',
    'date_item_avg_item_cnt_lag_12',
    'date_shop_avg_item_cnt_lag_1',
    'date_shop_avg_item_cnt_lag_2',
    'date_shop_avg_item_cnt_lag_3',
    'date_shop_avg_item_cnt_lag_6',
    'date_shop_avg_item_cnt_lag_12',
    'date_cat_avg_item_cnt_lag_1',
    'date_shop_cat_avg_item_cnt_lag_1',
    #'date_shop_type_avg_item_cnt_lag_1',
    #'date_shop_subtype_avg_item_cnt_lag_1',
    'date_city_avg_item_cnt_lag_1',
    'date_item_city_avg_item_cnt_lag_1',
    #'date_type_avg_item_cnt_lag_1',
    #'date_subtype_avg_item_cnt_lag_1',
    'delta_price_lag',
    'month',
    'days',
    'item_shop_last_sale',
    'item_last_sale',
    'item_shop_first_sale',
    'item_first_sale',
]]

X_train = data[data.date_block_num < 33].drop(['item_cnt_month'], axis=1)
Y_train = data[data.date_block_num < 33]['item_cnt_month']
X_valid = data[data.date_block_num == 33].drop(['item_cnt_month'], axis=1)
Y_valid = data[data.date_block_num == 33]['item_cnt_month']
X_test = data[data.date_block_num == 34].drop(['item_cnt_month'], axis=1)
print(f"X_train={X_train.shape} Y_train={Y_train.shape}")
print(f"X_valid={X_valid.shape} Y_valid={Y_valid.shape}")
print(f"X_test={X_test.shape} ")
del data
gc.collect();

params={'num_leaves': 550,   'n_estimators':1000,'early_stopping_rounds':100,
        'feature_fraction': 1,     'bagging_fraction': 1,
        'max_bin': 1024,
       "adaptive":'weight'
                  '',   #无效，晕
    #"learning_schedule":"adaptive",
     'max_depth': 10,
     'min_child_weight': 300,    #'min_data_in_leaf': 300,
     'learning_rate': 0.1,
     'objective': 'regression',
     'boosting_type': 'gbdt',
     'verbose': 1,
     'metric': {'rmse'}
}


def hyparam_core(num_leaves, feature_fraction, bagging_fraction, max_depth, learning_rate, min_data_in_leaf,max_bin):
    param_1 = params
    param_1['verbose']=0
    param_1["num_leaves"] = int(round(num_leaves))
    param_1['feature_fraction'] = max(min(feature_fraction, 1), 0)
    param_1['bagging_fraction'] = max(min(bagging_fraction, 1), 0)
    param_1['max_depth'] = int(round(max_depth))
    param_1['learning_rate'] = learning_rate
    param_1['min_data_in_leaf'] = int(round(min_data_in_leaf))
    param_1['max_bin'] = int(round(max_bin))

    model = LiteMORT(param_1).fit_1(X_train, Y_train, eval_set=[(X_valid, Y_valid)])
    Y_pred = model.predict(X_valid).clip(0, 20)
    score = np.sqrt(mean_squared_error(Y_pred, Y_valid))
    return -score

if isMORT:
    if False:   #BayesianOptimization
        pds = {'num_leaves': (547, 547),
               'feature_fraction': (1, 1),
               'bagging_fraction': (1, 1),
               'max_depth': (10,10),
               'learning_rate': (0.1, 0.1),
               'min_data_in_leaf': (20, 20),
               'max_bin': (128, 1024),
               }

        optimizer = BayesianOptimization(hyparam_core, pds, random_state=7)
        optimizer.maximize(init_points=5, n_iter=12)
        print(optimizer.max)
        for i, res in enumerate(optimizer.res):
            print("Iteration {}: \n\t{}".format(i, res))
        input(f"......BayesianOptimization is OK......")

    model = LiteMORT(params).fit_1(X_train,Y_train,eval_set=[(X_valid, Y_valid)])
else:
    model = XGBRegressor(
        max_depth=8,
        n_estimators=1000,
        min_child_weight=300,
        colsample_bytree=0.8,
        subsample=0.8,
        eta=0.3,
        seed=42)

    model.fit(
        X_train,
        Y_train,
        eval_metric="rmse",
        eval_set=[(X_train, Y_train), (X_valid, Y_valid)],
        verbose=True,
        early_stopping_rounds = 10)
    alg = 'xgboost'

Y_pred = model.predict(X_valid).clip(0, 20)
score = np.sqrt(mean_squared_error(Y_pred, Y_valid))
Y_test = model.predict(X_test).clip(0, 20)

if not isMORT:
    plot_features(model, (10, 14))

path=""
if some_rows is None:
    submission = pd.DataFrame({
        "ID": test.index,
        "item_cnt_month": Y_test
    })
    path = f'{data_root}/{alg}_[{score:.5g}].csv'
    submission.to_csv(path, index=False)

    # save predictions for an ensemble
    #pickle.dump(Y_pred, open(f'{data_root}/xgb_train.pickle', 'wb'))
    #pickle.dump(Y_test, open(f'{data_root}/xgb_test.pickle', 'wb'))
input(f"......Save submit @{path}......")
