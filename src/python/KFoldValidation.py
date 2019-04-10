from sklearn.model_selection import GroupKFold
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import lightgbm as lgb
import time
pd.options.display.max_columns = 999
import warnings
warnings.simplefilter("ignore")
import os
import pickle
import gc
from LiteMORT import *
from sklearn.preprocessing import LabelEncoder


mort_params = {
    'feature_quanti':1024,   'feature_sample':0.9,'min_child_samples':30,  'subsample': 0.9,
     'NA':-1,        'normal':0,
    'histo_bin_map':1,      #0-quantile,1-frequency                                   #'histo_algorithm': 0,
    'node_task':0,              #0:split-X,     1:split-Y 2:REGRESS_X
    'k_fold':5,
    'learning_rate': 0.01,
    'n_estimators': 2000,    'num_leaves': 31,
    'min_split_gain': np.power(10.0, -4.9380),
    'reg_alpha': np.power(10.0, -3.2454),
    'reg_lambda': np.power(10.0, -4.8571),
    'min_child_weight': np.power(10.0, 2),
    "early_stopping_rounds": 50, "verbose": 0,
}

'''
    v0.1    cys
        12/10/2018
    缺省 StratifiedKFold
'''
class KFoldBoost():
    def __init__(self, data=None):
        self.fold_ids = []
        
        self.mort = None
        self.xtool = None       #lightgbm,xgboost,catboost, ....    third party 
        self.desc = ""        

    '''
        You can’t do stratified on multilabel. Just do kfold
        StratifiedKFold-Split其实只需要一个参数    Note that providing y is sufficient to generate the splits and hence np.zeros(n_samples) may be used as a placeholder for X instead of actual training data.

    '''
    def split_ids(self,X_,Y_,folds=None, random_state=42):
        self.nFold = mort_params['k_fold']
        self.folds = folds
        if self.folds is None:
            self.folds = StratifiedKFold(n_splits=self.nFold, shuffle=True, random_state=random_state)
            self.fold_ids = self.folds.split(X_,Y_)
        return

    def Init_MORT(self,user_train,user_test,features,isEDA=True):
        t0 = time.time()
        self.mort = LiteMORT()
        self.mort.init(mort_params)
        feat_dict={}
        for feat in features:
            feat_dict[feat]=0
        feat_dict['feature_1'] = 1
        self.mort.init_features(feat_dict)
        if mort_params['NA'] == -1:
            print("---- !!!No data imputation!!!----")
        else:
            if True:  # 奇怪的教训，会影响其它列,需要重写，暂时这样！！！
                user_train[features] = self.mort.Imputer(mort_params, user_train[features], None, np.float32)
                user_test[features] = self.mort.Imputer(mort_params, user_test[features], None, np.float32)
            else:
                user_train = OnMissingValue(user_train);
                user_test = OnMissingValue(user_test)
            # print("",user_train.iloc[3004,190])

        if isEDA:
            all_data = pd.concat([user_train[features], user_test[features]])
            print("====== all_data for EDA={}\n".format(all_data.shape))
            self.mort.EDA(mort_params,all_data, None,user_test.shape[0])
            #self.mort.EDA(mort_params, user_test[features], None, 0)       #直接传入这个分布，似乎没啥用     12/11/2018
            del all_data;
        gc.collect()
        self.desc = "MORT"
        print("======KFoldBoost::Init_MORT time={:.3f}".format(time.time()-t0))

    def Init_lgb_regress(self):
        self.xtool = lgb.LGBMRegressor(n_estimators=mort_params['n_estimators'], objective="regression", metric="rmse", num_leaves=mort_params['num_leaves'],#31,
                    min_child_samples=mort_params['min_child_samples'],#50,
                    learning_rate=mort_params['learning_rate'], subsample=mort_params['subsample'], feature_fraction=mort_params['feature_sample'],#0.5,
                    bagging_seed=2019,
            )#use_best_model=True
        self.desc="lgb_"+str(lgb.__version__)

    def score(data, y):
        return 0.0

    def validate(self, train,target, test, features, 
                 fit_params={"early_stopping_rounds": 50, "verbose": 100, "eval_metric": "rmse"}):
        self.predictions=0
        full_score = 0
        mort = self.mort
        xboost = self.xtool
        for fold_, (trn_idx, val_idx) in enumerate(self.fold_ids):
            print("fold {}:...".format(fold_))
            devel = train[features].iloc[trn_idx]
            y_devel = target.iloc[trn_idx]
            valid = train[features].iloc[val_idx]
            y_valid = target.iloc[val_idx]
            eval_set = [(valid, y_valid)]
            if mort is not None:
                #mort.EDA(mort_params, devel, None, 0)
                mort.fit(mort_params, devel, y_devel, valid, y_valid)
                # print("======LGB_{} mort.predict...... ".format(fold_))
                predict_valid = mort.predict(valid)
            else:
                xboost.FI = pd.DataFrame(index=features)
                xboost.fit(devel, y_devel, eval_set=eval_set, **fit_params)
                if len(xboost.feature_importances_) == len(features):  # some bugs in catboost?
                    xboost.FI['fold' + str(fold_)] = xboost.feature_importances_ / xboost.feature_importances_.sum()
                if False:
                    xboost.booster_.save_model('mode.txt', num_iteration=10)
                    graph = lgb.create_tree_digraph(xboost)
                    graph.render(view=True)
                predict_valid = xboost.predict(valid)
            fold_score = mean_squared_error(y_valid, predict_valid) ** 0.5
            print("\nFold ", fold_, " score: ",fold_score )
            full_score += fold_score / self.nFold

            if mort is not None:
                test_predictions = mort.predict(test[features])
            else:
                test_predictions = xboost.predict(test[features])
            self.predictions += test_predictions / self.nFold
            if mort is not None:
                mort.Clear()
            gc.collect()

        print("Final score: ", full_score)
        return full_score