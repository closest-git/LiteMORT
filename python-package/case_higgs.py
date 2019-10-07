'''
    https://archive.ics.uci.edu/ml/datasets/HIGGS
This is a classification problem to distinguish between a signal process which produces Higgs bosons and a background process which does not.
The data has been produced using Monte Carlo simulations. The first 21 features (columns 2-22) are kinematic properties measured by the particle detectors in the accelerator. The last seven features are functions of the first 21 features; these are high-level features derived by physicists to help discriminate between the two classes. There is an interest in using deep learning methods to obviate the need for physicists to manually develop such features. Benchmark results using Bayesian Decision Trees from a standard physics package and 5-layer neural networks are presented in the original paper. The last 500,000 examples are used as a test set.

    https://github.com/Laurae2/boosting_tree_benchmarks/tree/master/data
    https://github.com/guolinke/boosting_tree_benchmarks/tree/master/data
    https://blog.bigml.com/2017/09/28/case-study-finding-higgs-bosons-with-deepnets/

    5/19/2019 需要确定是regression 或 binary classification
    8/23/2019   subsample subfeature 似乎都没用(2000000测试)
        lesome_rows=2000000 iter=2000 auc=0.83775(1,1) auc=0.83847(0.8,1);auc=0.83618(0.8,0.5)

'''
import lightgbm as lgb
import time
import sys
import os
import gc
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold, RepeatedKFold
import pickle
from litemort import *
#from LiteMORT_EDA import *

isMORT = len(sys.argv)>1 and sys.argv[1] == "mort"
#isMORT = True
model_type = 'mort' if isMORT else 'lgb'
#some_rows=      200000
#some_rows=      2000000
some_rows=      10500000
nTotal =        11000000
nLastForTest =    500000       #The last 500,000 examples are used as a test set.

#some_rows=None

def read_higgs_data(path):
    pkl_path = 'F:/Datasets/HIGGS_/higgs_data_{}.pickle'.format(some_rows)
    if os.path.isfile(pkl_path):
        print("====== Load pickle @{} ......".format(pkl_path))
        with open(pkl_path, "rb") as fp:
            [X, y, X_test,y_test] = pickle.load(fp)
    else:
        assert(some_rows<=nTotal-nLastForTest)
        print("====== Read last {} examples as training set ......".format(some_rows))
        df = pd.read_csv(path, nrows=some_rows,header=None)
        y=pd.Series(df.iloc[:,0])
        X=df.iloc[:,1:]
        print("====== Read last {} examples as testing set ......".format(nLastForTest))
        df = pd.read_csv(path, skiprows = nTotal-nLastForTest,nrows=nLastForTest, header=None)
        y_test = pd.Series(df.iloc[:, 0])
        X_test = df.iloc[:,1:]
        del df
        gc.collect()
        print("====== Save pickle @{} ......".format(pkl_path))
        with open(pkl_path, "wb") as fp:  # Pickling
            pickle.dump([X, y, X_test,y_test], fp)
    print("====== read_higgs_data X={}, y={}, X_test={} ...... OK".format(X.shape, y.shape, X_test.shape))
    return X,y,X_test

X,y,X_test = read_higgs_data("F:/Datasets/HIGGS_/HIGGS.csv")
#X = Unique_Expand(X)
#X_test = Unique_Expand(X_test)
num_rounds = 10001
params = {
        "objective": "binary",
        "metric": "auc",        #"binary_logloss"
        "adaptive":'weight',
            'max_bin': 256,
          'num_leaves': 64,
          'learning_rate': 0.1,
          'tree_learner': 'serial',
          'task': 'train',
          'is_training_metric': 'false',
          'min_data_in_leaf': 512,
          #'min_sum_hessian_in_leaf': 100,
          #'bagging_fraction': 1,#0.2,
          'subsample': 1,     'bagging_freq': 1,
            'feature_fraction': 1,
          #'ndcg_eval_at': [1, 3, 5, 10],
          #'sparse_threshold': 1.0,
            'n_estimators':num_rounds,
            'early_stopping_rounds': 500,
          #'device': 'cpu'
           #'device': 'gpu',
          #'gpu_platform_id': 0,
          #'gpu_device_id': 0
          }
n_fold = 5
folds = KFold(n_splits=n_fold, shuffle=True, random_state=11)
for fold_n, (train_index, valid_index) in enumerate(folds.split(X)):
    t0 = time.time()

    if type(X) == np.ndarray:
        X_train, X_valid = X[train_index], X[valid_index]
        y_train, y_valid = y[train_index], y[valid_index]
    else:
        X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
        y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]
    if False:
        mean = y_train.mean();
        d_train = pd.concat([y_train, X_train], ignore_index=True, axis=1)
        print("X_train={}, y_train={} d_train={}".format(X_train.shape, y_train.shape, d_train.shape))
        np.savetxt("D:/LightGBM-master/examples/regression/geo_test.csv", d_train, delimiter='\t')

    if model_type == 'mort':
        #model = LiteMORT(params).fit(X_train, y_train, eval_set=[(X_valid, y_valid)])
        model = LiteMORT(params).fit_1(X_train, y_train, eval_set=[(X_valid, y_valid)])
        #y_pred_valid = model.predict(X_valid)
        #y_pred = model.predict(X_test)

    if model_type == 'lgb':
        model = lgb.LGBMRegressor(**params, n_jobs=-1)
        model.fit(X_train, y_train,
                  eval_set=[(X_train, y_train), (X_valid, y_valid)], eval_metric='auc',verbose=5)
        model.booster_.save_model('geo_test_.model')
        #y_pred_valid = model.predict(X_valid)
        #y_pred = model.predict(X_test, num_iteration=model.best_iteration_)
    break

input("loss is {} time={:.3g} model={}...".format(0,time.time()-t0,model_type))
sys.exit(-1)

t0 = time.time()
gbm = lgb.train(params, train_set=dtrain, num_boost_round=10,
                valid_sets=None, valid_names=None,
                fobj=None, feval=None, init_model=None,
                feature_name='auto', categorical_feature='auto',
                early_stopping_rounds=None, evals_result=None,
                verbose_eval=True,
                keep_training_booster=False, callbacks=None)
t1 = time.time()

print('cpu version elapse time: {}'.format(t1 - t0))
