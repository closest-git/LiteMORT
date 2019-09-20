#https://www.kaggle.com/kyakovlev/ieee-simple-lgbm

# General imports
import numpy as np
import pandas as pd
import os, sys, gc, warnings, random, datetime
import time
import pickle
from sklearn import metrics
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder
from litemort import *
from tqdm import tqdm
import math
warnings.filterwarnings('ignore')

isMORT = len(sys.argv)>1 and sys.argv[1] == "mort"
isMORT = True
model='MORT' if isMORT else 'LGB'
NFOLDS = 8
#some_rows = 50000
some_rows = None
data_root = 'E:/Kaggle/ieee_fraud/input/'
#data_root = '../input/'
pkl_path = f'{data_root}/_kyakovlev_{some_rows}.pickle'

def M_PickSamples(pick_samples,df_train,df_test):
    nMost = min(df_train.shape[0], df_test.shape[0])
    random.seed(42)
    subset = random.sample(range(nMost), pick_samples)
    df_train = df_train.iloc[subset, :].reset_index(drop=True)
    df_test = df_test.iloc[subset, :].reset_index(drop=True)
    print('====== Mort_PickSamples ... df_train={} df_test={}'.format(df_train.shape, df_test.shape))
    return df_train,df_test

def seed_everything(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

import lightgbm as lgb


def make_predictions(tr_df, tt_df, features_columns, target, lgb_params, NFOLDS=2):
    print(f'train_df={tr_df.shape} test_df={tt_df.shape} \nlgb_params={lgb_params}')
    folds = KFold(n_splits=NFOLDS, shuffle=True, random_state=SEED)

    #X, y = tr_df[features_columns], tr_df[target]
    #P, P_y = tt_df[features_columns], tt_df[target]
    y, P_y = tr_df[target], tt_df[target]


    predictions = np.zeros(len(tt_df))

    for fold_, (trn_idx, val_idx) in enumerate(folds.split(tr_df[features_columns], y)):
        t0=time.time()
        print('Fold:', fold_)
        tr_x, tr_y = tr_df[features_columns].iloc[trn_idx, :], y[trn_idx]
        vl_x, vl_y = tr_df[features_columns].iloc[val_idx, :], y[val_idx]
        print(len(tr_x), len(vl_x))

        if isMORT:
            model = LiteMORT(lgb_params).fit(tr_x, tr_y, eval_set=[(vl_x, vl_y)])
            best_iter = 1000
            # pred_val = model.predict(vl_x)
            pred_raw = model.predict_raw(vl_x)
            # y_pred[val_idx] = pred_raw
            fold_score = metrics.roc_auc_score(vl_y, pred_raw)
            pp_p = model.predict_raw(tt_df[features_columns])
        else:
            tr_data = lgb.Dataset(tr_x, label=tr_y)
            if LOCAL_TEST:
                vl_data = lgb.Dataset(tt_df[features_columns], label=P_y)
            else:
                vl_data = lgb.Dataset(vl_x, label=vl_y)
            estimator = lgb.train(
                lgb_params,
                tr_data,
                valid_sets=[tr_data, vl_data],
                verbose_eval=200,
            )
            pred_raw = estimator.predict(vl_x)
            fold_score = metrics.roc_auc_score(vl_y, pred_raw)
            pp_p = estimator.predict(tt_df[features_columns])
            del tr_data, vl_data

        predictions += pp_p / NFOLDS

        if LOCAL_TEST:
            feature_imp = pd.DataFrame(sorted(zip(estimator.feature_importance(), X.columns)),
                                       columns=['Value', 'Feature'])
            print(feature_imp)
        print(f'Fold:{fold_} score={fold_score} time={time.time() - t0:.4g} tr_x={tr_x.shape} val_x={vl_x.shape}')
        del tr_x, tr_y, vl_x, vl_y
        gc.collect()
    tt_df = tt_df[['TransactionID', target]]
    tt_df['prediction'] = predictions
    gc.collect()

    return tt_df,fold_score

SEED = 42
seed_everything(SEED)
LOCAL_TEST = False
TARGET = 'isFraud'
START_DATE = datetime.datetime.strptime('2017-11-30', '%Y-%m-%d')
lgb_params = {
                    'objective':'binary',
                    'boosting_type':'gbdt',
                    'metric':'auc',
                    'n_jobs':-1,
                    'learning_rate':0.01,
                    "learning_schedule":'adaptive',
                    'num_leaves': 2**8,
                    'max_depth':-1,
                    'tree_learner':'serial',
                    'colsample_bytree': 0.7,
                    'subsample_freq':1,
                    'subsample':0.7,
                    'n_estimators':800,
                    'max_bin':255,
                    'verbose':666,
                    'seed': SEED,
                    'early_stopping_rounds':100,
                }

if os.path.isfile(pkl_path):
    print("====== Load pickle @{} ......".format(pkl_path))
    with open(pkl_path, "rb") as fp:
        [train_df, test_df, features_columns] = pickle.load(fp)
else:
    print('Load Data......')
    train_df = pd.read_pickle(f'{data_root}/ieee-fe-with-some-eda/train_df.pkl')

    if LOCAL_TEST:
        test_df = train_df[train_df['DT_M']==train_df['DT_M'].max()].reset_index(drop=True)
        train_df = train_df[train_df['DT_M']<(train_df['DT_M'].max()-1)].reset_index(drop=True)
    else:
        test_df = pd.read_pickle(f'{data_root}/ieee-fe-with-some-eda/test_df.pkl')

    remove_features = pd.read_pickle(f'{data_root}/ieee-fe-with-some-eda/remove_features.pkl')
    remove_features = list(remove_features['features_to_remove'].values)
    print('Load Data OK\nShape control:', train_df.shape, test_df.shape)

    features_columns = [col for col in list(train_df) if col not in remove_features]

    ########################### Final Minification
    print('reduce_mem_usage......')
    train_df = reduce_mem_usage(train_df)
    test_df  = reduce_mem_usage(test_df)
    print('reduce_mem_usage......OK!!!')
    if some_rows is not None:
        train_df,test_df = M_PickSamples(some_rows,train_df,test_df)
    with open(pkl_path, "wb") as fp:  # Pickling
        pickle.dump([train_df, test_df, features_columns], fp)
        print("====== Dump pickle @{} ......OK".format(pkl_path))


if LOCAL_TEST:
    lgb_params['learning_rate'] = 0.01
    lgb_params['n_estimators'] = 20000
    lgb_params['early_stopping_rounds'] = 100
    test_predictions = make_predictions(train_df, test_df, features_columns, TARGET, lgb_params)
    print(metrics.roc_auc_score(test_predictions[TARGET], test_predictions['prediction']))
else:
    lgb_params['learning_rate'] = 0.005
    lgb_params['n_estimators'] = 4800
    lgb_params['early_stopping_rounds'] = 100
    test_predictions,fold_score = make_predictions(train_df, test_df, features_columns, TARGET, lgb_params, NFOLDS=NFOLDS)
    test_predictions['isFraud'] = test_predictions['prediction']
    # test_predictions[['TransactionID', 'isFraud']].to_csv(f'submit_{some_rows}_{0.5}.csv', index=False,compression='gzip')
    path = f'E:/Kaggle/ieee_fraud/result/[{model}]_{some_rows}_{fold_score:.5f}_F{NFOLDS}_.csv'
    test_predictions[['TransactionID', 'isFraud']].to_csv(path, index=False)  # ,compression='gzip'
    print(f"test_predictions[['TransactionID', 'isFraud']] to_csv @{path}")
    input("Press Enter to exit...")