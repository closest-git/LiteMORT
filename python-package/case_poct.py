import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, KFold, RepeatedKFold
from sklearn.metrics import mean_absolute_error,mean_squared_error
from sklearn.metrics import roc_auc_score, roc_curve,auc
import time
import numpy as np
from litemort import *
isMORT=False
import matplotlib.pyplot as plt
import pandas as pd
import gc
import seaborn as sns
import pickle

def ROC_plot(features,X_,y_, pred_,title):
    fpr_, tpr_, thresholds = roc_curve(y_, pred_)
    optimal_idx = np.argmax(tpr_ - fpr_)
#https://stackoverflow.com/questions/28719067/roc-curve-and-cut-off-point-python
    optimal_threshold = thresholds[optimal_idx]
    auc_ = auc(fpr_, tpr_)
    title = "{} auc=".format(title)
    print("{} auc={} OT={:.4g}".format(title, auc_,optimal_threshold))
    plt.plot(fpr_, tpr_, label="{}:{:.4g}".format(title, auc_))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('SMPLEs={} Features={} OT={:.4g}'.format(X_.shape[0],len(features),optimal_threshold))
    plt.legend(loc='best')
    plt.savefig("./_auc_[{}].jpg".format(features))
    plt.show()
    return auc_,optimal_threshold

def runLgb(X, y, test=None, num_rounds=10000, max_depth=-1, eta=0.01, subsample=0.8,
           colsample=0.8, min_child_weight=1, early_stopping_rounds=50, seeds_val=2017):
    plot_feature_importance = True
    features = list(X.columns)
    print("X={} y={}".format(X.shape,y.shape))
    params = {'task': 'train',
             'min_data_in_leaf': 32,
             'boosting_type': 'gbdt',
             'objective': 'binary',
             'learning_rate': eta,
             # 'metric': {'multi_logloss'},
             'metric': 'auc',
             'max_depth': max_depth,
             # 'min_child_weight':min_child_weight,
             'bagging_fraction': subsample,
             'feature_fraction': colsample,
             'bagging_seed': seeds_val,
             'num_iterations': num_rounds,
             'num_leaves': 32,
             'lambda_l1': 1.0,
             'verbose': 0,
             'nthread': -1}
    n_fold = 5
    folds = KFold(n_splits=n_fold, shuffle=True, random_state=11)
    y_pred=np.zeros(y.shape[0])
    feature_importance = None
    if not isMORT:
        feature_importance = pd.DataFrame()
    for fold_n, (train_index, valid_index) in enumerate(folds.split(X)):
        t0 = time.time()

        if type(X) == np.ndarray:
            X_train, X_valid = X[train_index], X[valid_index]
            y_train, y_valid = y[train_index], y[valid_index]
        else:
            X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
            y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

        if isMORT:
            model = LiteMORT(params).fit(X_train, y_train, eval_set=[(X_valid, y_valid)])
            pass
        else:
            lgtrain = lgb.Dataset(X_train, y_train)
            lgval = lgb.Dataset(X_valid, y_valid)
            model = lgb.train(params, lgtrain, num_rounds, valid_sets=lgval,
                              early_stopping_rounds=early_stopping_rounds, verbose_eval=100)
            fold_importance = pd.DataFrame()
            fold_importance["feature"] = X.columns
            fold_importance["importance"] = model.feature_importance()
            fold_importance["fold"] = fold_n + 1
            feature_importance = pd.concat([feature_importance, fold_importance], axis=0)
        pred_val = model.predict(X_valid)
        y_pred[valid_index] = pred_val
        model.save_model(f'model_lgb_poct_{fold_n}_.txt')
        if test is not None:
            pred_test = model.predict(test, num_iteration=model.best_iteration)
        else:
            pred_test = None
    auc = roc_auc_score(y, y_pred)
    if feature_importance is not None:
        feature_importance["importance"] /= n_fold
        if plot_feature_importance:
            cols = feature_importance[["feature", "importance"]].groupby("feature").mean().sort_values(
                by="importance", ascending=False)[:].index
            best_features = feature_importance.loc[feature_importance.feature.isin(cols)]
            plt.figure(figsize=(5, 3));
            sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
            plt.xlabel("importance of each feature")
            plt.title('AUC={:.3f} ({}-folds)'.format(auc,n_fold))
            plt.savefig("./_importance_[{}].jpg".format(features))
            plt.show()

    ROC_plot(features,X, y, y_pred, "")

    print("CV score: {:<8.5f}".format(auc))
    return auc

pkl_path=f"E:/POCTx/poct_InHospital.pkl"
with open(pkl_path, "rb") as fp:  # Pickling
    X = pickle.load(fp)
    y = pickle.load(fp)
score = runLgb(X, y)