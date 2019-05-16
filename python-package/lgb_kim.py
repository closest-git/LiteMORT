#https://www.kaggle.com/chocozzz/santander-lightgbm-baseline-lb-0-899

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from sklearn.model_selection import KFold, StratifiedKFold
import warnings
import gc
import time
import sys
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')
from sklearn import metrics
from litemort import *

isMORT = len(sys.argv)>1 and sys.argv[1] == "mort"
isMORT = True
#some_rows=10000
some_rows=None
pick_samples=None
#pick_samples=1500
early_stop=50      #0.898->0.899
max_bin = 256      #119增大到1280
lr=0.02             #0.02   0.01提升一点点   0.89826-》0.89842
nFold,nLeaves = 5,10
n_round=35000
min_child=32         #11(0.89830)-> 32(0.89882)
x_feat,x_sub=1,0.15
x_feat,x_sub=0.15,0.3
#x_feat,x_sub=1,1;       nLeaves=2;n_round=5       #仅用于测试

print('argv={}\nsome_rows={} pick_samples={}'.format(sys.argv,some_rows,pick_samples))
plt.style.use('seaborn')
sns.set(font_scale=1)
pd.set_option('display.max_columns', 500)

train_df = pd.read_csv("G:/kaggle/Santander/input/train.csv",nrows=some_rows)
test_df = pd.read_csv("G:/kaggle/Santander/input/test.csv",nrows=some_rows)

features = [c for c in train_df.columns if c not in ['ID_code', 'target']]
target = train_df['target']
cat_cols=[];
if True:
    #see_all_2(train_df, test_df, features, target, [0, 256, 1024])
    #plot_binary_dist(train_df, test_df, ['var_81','var_68','var_139'],bins=1024)      #','var_139','var_12','var_53','var_110'
    #[train_df,test_df],features=feat_extend([train_df,test_df],features)
    print("")
else:
    #[train_df,test_df],features=feat_extend([train_df,test_df],features)
    may_cols=['var_68','var_6','var_108','var_13','var_33','var_146','var_21','var_80','var_139','var_81']
    may_cols=['var_68']     #var_68 is date?
    #may_cols = features
    #[train_df,test_df],features,cat_cols=df2category_hisogram([train_df,test_df],features,may_cols)
    #train_df,test_df,features,cat_cols = df2category_rf(train_df,target,test_df,features)

from sklearn.metrics import roc_auc_score, roc_curve
#Target Encoding
TE_folds, TE_inner_folds=10,5
if True:
    for var_name in cat_cols:
        #train_df, test_df, feat_T = TE_cross(5, 2, train_df, 'target', test_df, var_name)
        train_df, test_df, feat_T = TE_expm(train_df, 'target', test_df, var_name)
        features.append(feat_T)


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2019)
oof = np.zeros(len(train_df))
predictions = np.zeros(len(test_df))
feature_importance_df = pd.DataFrame()

start = time.time()

param = {
        'num_leaves': nLeaves,
        'num_round':n_round,
        #'num_leaves': 32,
        #'max_bin': 119,
        'max_bin': max_bin,
        'min_data_in_leaf': min_child,
        'learning_rate': lr,
        #'learning_rate': 0.5,
        'min_sum_hessian_in_leaf': 0.00245,
        'bagging_fraction': x_sub,
        'bagging_freq': 5,
        'feature_fraction': x_feat,
        'lambda_l1': 4.972,
        'lambda_l2': 2.276,
        'min_gain_to_split': 0.65,
        'max_depth': 14,
        'save_binary': True,
        'seed': 1337,
        'feature_fraction_seed': 1337,
        'bagging_seed': 1337,
        'drop_seed': 1337,
        'data_random_seed': 1337,
        'objective': 'binary',
        'boosting_type': 'gbdt',
        'verbose': 1,
        'metric': 'auc',
        #'metric': 'auc',
        'is_unbalance': True,
        'boost_from_average': False,
    }
N_min = 100     #   N_min 越大，regularization效果越强 smoothing term, minimum sample size, if sample size is less than N_min, add up to N_min
#

alg="LGBMRegressor"
print("LightGBM training... train={} test={} \nparam={}".format(train_df.shape,test_df.shape,param))
features_0=features.copy()
for fold_, (trn_idx, val_idx) in enumerate(skf.split(train_df.values, target.values)):
    t0=time.time()
    print("fold n°{}".format(fold_))
    num_round = 10000
    features = features_0.copy()
    X_train = train_df.iloc[trn_idx][features].astype(np.float32)
    Y_train = target.iloc[trn_idx].astype(np.double)
    X_test = train_df.iloc[val_idx][features].astype(np.float32)
    Y_test = target.iloc[val_idx].astype(np.double)
    if isMORT:
        #mort = LiteMORT(param).fit(X_train, Y_train, eval_set=[(X_test, Y_test)])
        mort = LiteMORT(param).fit_1(X_train, Y_train, eval_set=[(X_test, Y_test)])
        oof[val_idx] = mort.predict_raw(X_test)
        fold_score = roc_auc_score(Y_test, oof[val_idx])
        #print("\nFold ", fold_, " score: ", fold_score)
        predictions += mort.predict_raw(test_df[features]) / 5
    else:
        if alg=="LGBMRegressor":
            dev=X_train
            val=X_test
            target_col='target_col';  dev[target_col]=Y_train
            clf = lgb.LGBMRegressor(num_boost_round=num_round, early_stopping_rounds=early_stop,**param)
            if True:
                print("features={} X_train={} Y_train={} X_test={} Y_test={} ".format( len(features),
                    X_train.shape,Y_train.shape,X_test.shape,Y_test.shape))
                clf.fit(X_train[features], Y_train, eval_set=[(X_train[features], Y_train),(X_test[features], Y_test)],eval_metric="auc",categorical_feature=cat_cols, verbose=1000)
                feat_importance = clf.feature_importances_
                best_iteration = clf.best_iteration_
                if best_iteration is None:
                    best_iteration = -1
                oof[val_idx] = clf.predict(val[features],num_iteration=best_iteration)
            else:
                gLR = GBDT_LR(clf)
                gLR.fit(X_train, Y_train, eval_set=[(X_test, Y_test)],eval_metric="auc", verbose=1000)
                feat_importance = gLR.feature_importance()
                best_iteration = -1
                clf=gLR
                oof[val_idx] = clf.predict(train_df.iloc[val_idx][features], y_=target.iloc[val_idx],
                                           num_iteration=best_iteration)

        else:       #lambda ranker
            gbr = lgb.LGBMRanker()
            gbr.fit(X_train, y_train, group=q_train, eval_set=[(X_test, y_test)],
                    eval_group=[q_test], eval_at=[1, 3], early_stopping_rounds=5, verbose=False,
                    callbacks=[lgb.reset_parameter(learning_rate=lambda x: 0.95 ** x * 0.1)])

        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = features
        fold_importance_df["importance"] = feat_importance
        fold_importance_df["fold"] = fold_ + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

        predictions += clf.predict(test_df[features], num_iteration=best_iteration) / 5
    print("fold n°{} time={} score={}".format(fold_,time.time()-t0,fold_score))
cv_score = roc_auc_score(target, oof)
print("CV score: {:<8.5f}".format(cv_score))

if feature_importance_df.size>0:
#if False:
    cols = (feature_importance_df[["feature", "importance"]]
            .groupby("feature")
            .mean()
            .sort_values(by="importance", ascending=False)[:32].index)
    best_features = feature_importance_df.loc[feature_importance_df.feature.isin(cols)]

    plt.figure(figsize=(14,26))
    sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance",ascending=False))
    plt.title('LightGBM Features (averaged over folds)')
    plt.tight_layout()
    plt.savefig('lgbm_importances.png')


input("Press Enter to continue...")