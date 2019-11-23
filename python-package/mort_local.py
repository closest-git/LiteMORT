import litemort
from litemort import *
print(litemort.__version__)

early_stop = 20
verbose_eval = 5
metric = 'l2'
#num_rounds=1000, lr=0.05, bf=0.3
num_rounds = 1000;      lr = 0.05;          bf = 0.3
params = {'num_leaves': 31, 'n_estimators': num_rounds,
              'objective': 'regression',
              'max_bin': 256,
              #               'max_depth': -1,
              'learning_rate': lr,
              "boosting": "gbdt",
              "bagging_freq": 5,
              "bagging_fraction": bf,
              "feature_fraction": 0.9,  # STRANGE GBDT  why("bagging_freq": 5 "feature_fraction": 0.9)!!!
              "metric": metric, "verbose_eval": verbose_eval, 'n_jobs': 8, "elitism": 0,"debug":'1',
              "early_stopping_rounds": early_stop, "adaptive": 'weight1', 'verbose': 0, 'min_data_in_leaf': 20,
              #               "verbosity": -1,
              #               'reg_alpha': 0.1,
              #               'reg_lambda': 0.3
              }
mort=LiteMORT(params)