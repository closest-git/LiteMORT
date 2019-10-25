#https://www.kaggle.com/hmendonca/shapley-values-for-feature-selection-ashrae

import sys
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.metrics import log_loss, mean_squared_error
from litemort import *
from LiteMORT_hyppo import *
import datetime
import time
import random
import gc
import pandas as pd
import numpy as np
import lightgbm as lgb
#import shap as shap
import os
from tqdm import tqdm
unimportant_cols = ['wind_direction', 'wind_speed', 'sea_level_pressure']
target = 'meter_reading'

isMORT = len(sys.argv)>1 and sys.argv[1] == "mort"
isMORT = True
gbm='MORT' if isMORT else 'LGB'
folds = 4
seed = 42
some_rows = 5000000
#some_rows = None
path = '../input/ashrae-energy-prediction'
path = 'F:/Datasets/ashrae/'
pkl_path = f'{path}/_ashrae_{some_rows}_.pickle'
submit_path = f'{path}/[{gbm}]_{some_rows}__.csv.gz'
def load_data(source='train', path=path):    #''' load and merge all tables '''
    assert source in ['train', 'test']
    building = pd.read_csv(f'{path}/building_metadata.csv', dtype={'building_id':np.uint16, 'site_id':np.uint8})
    weather  = pd.read_csv(f'{path}/weather_{source}.csv', parse_dates=['timestamp'],
                                                           dtype={'site_id':np.uint8, 'air_temperature':np.float16,
                                                                  'cloud_coverage':np.float16, 'dew_temperature':np.float16,
                                                                  'precip_depth_1_hr':np.float16},
                                                           usecols=lambda c: c not in unimportant_cols)
    df = pd.read_csv(f'{path}/{source}.csv', dtype={'building_id':np.uint16, 'meter':np.uint8}, parse_dates=['timestamp'])
    df = df.merge(building, on='building_id', how='left')
    df = df.merge(weather, on=['site_id', 'timestamp'], how='left')
    return df

if os.path.isfile(pkl_path):
    print("====== Load pickle @{} ......".format(pkl_path))
    with open(pkl_path, "rb") as fp:
        [train, test,features,folds,tr_idxs,val_idxs] = pickle.load(fp)
    print(f"train={train.shape}, test={test.shape} folds={folds}")
else:   # load and display some samples
    train = load_data('train')
    test = load_data('test')
    if some_rows is not None:
        train,_ = Mort_PickSamples(some_rows,train,None)
        print('====== Some Samples ... data={}'.format(train.shape))
    print(train.sample(5),test.sample(5))
    print("Ratio of available data (not NAN's):")
    data_ratios = train.count()/len(train)
    print("Ratio of available data (not NAN's):")
    test.count()/len(test)
    print(train.loc[:, data_ratios < 1.0].mean())

    class ASHRAE3Preprocessor(object):
        @classmethod
        def fit(cls, df, data_ratios=data_ratios):
            cls.avgs = df.loc[:, data_ratios < 1.0].mean()
            cls.pu_le = LabelEncoder()
            cls.pu_le.fit(df["primary_use"])

        @classmethod
        def transform(cls, df):
            df = df.fillna(cls.avgs)  # refill NAN with averages
            df['primary_use'] = np.uint8(cls.pu_le.transform(df['primary_use']))  # encode labels
            # expand datetime into its components
            df['hour'] = np.uint8(df['timestamp'].dt.hour)
            df['day'] = np.uint8(df['timestamp'].dt.day)
            df['weekday'] = np.uint8(df['timestamp'].dt.weekday)
            df['month'] = np.uint8(df['timestamp'].dt.month)
            df['year'] = np.uint8(df['timestamp'].dt.year - 2000)
            # parse and cast columns to a smaller type
            df.rename(columns={"square_feet": "log_square_feet"}, inplace=True)
            df['log_square_feet'] = np.float16(np.log(df['log_square_feet']))
            df['year_built'] = np.uint8(df['year_built'] - 1900)
            df['floor_count'] = np.uint8(df['floor_count'])

            # remove redundant columns
            for col in df.columns:
                if col in ['timestamp', 'row_id']:
                    del df[col]

            # extract target column
            if 'meter_reading' in df.columns:
                df['meter_reading'] = np.log1p(df['meter_reading']).astype(np.float32)  # comp metric uses log errors
            return df


    ASHRAE3Preprocessor.fit(train)
    print('ASHRAE3Preprocessor.transform......')
    train = ASHRAE3Preprocessor.transform(train)
    print(train.sample(7))
    print(train.dtypes)
    features = [col for col in train.columns if col not in [target, 'year', 'month', 'day']]
    test = ASHRAE3Preprocessor.transform(test)
    print(test.sample(5))
    #train[features].sample(5)
    kf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    kf_split = enumerate(kf.split(train, train['building_id']))
    tr_idxs,val_idxs=[],[]
    for i, (tr_idx, val_idx) in kf_split:
        print(f"{i}-{tr_idx.shape}{val_idx.shape}")
        tr_idxs.append(tr_idx)
        val_idxs.append(val_idx)

    with open(pkl_path, "wb") as fp:  # Pickling
        pickle.dump([train, test,features,folds,tr_idxs,val_idxs], fp)
        print("====== Dump pickle @{} ......OK".format(pkl_path))
    input("......")

#kf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
models = []
shap_values = np.zeros(train[features].shape)
shap_sampling = 125000  # reduce compute cost
oof_pred = np.zeros(train.shape[0])  # out of fold predictions

params={    'application': 'regression',
    'n_estimators':200,'learning_rate':0.4,'feature_fraction':0.9,'subsample':0.25,  # batches of 25% of the data
    'subsample_freq':1,'num_leaves':20,'lambda_l1':1,'lambda_l2':1,'metric':'rmse','n_jobs':-1,
    "adaptive":'weight1','verbose': 666,
}
## stratify data by building_id
#for i, (tr_idx, val_idx) in tqdm(enumerate(kf.split(train, train['building_id'])), total=folds):
def fit_regressor(tr_idx, val_idx, i):  # memory closure
    t0 = time.time()
    tr_x, tr_y = train[features].iloc[tr_idx], train[target].iloc[tr_idx]
    vl_x, vl_y = train[features].iloc[val_idx], train[target].iloc[val_idx]
    print({'fold': i, 'train size': len(tr_x), 'eval size': len(vl_x)})
    if isMORT:
        clf = LiteMORT(params).fit(tr_x, tr_y, eval_set=[(vl_x, vl_y)])
        fold_importance = None
    else:
        tr_data = lgb.Dataset(tr_x, label=tr_y)
        vl_data = lgb.Dataset(vl_x, label=vl_y)
        clf = lgb.LGBMRegressor(**params)
        clf.fit(tr_x, tr_y,
                eval_set=[(vl_x, vl_y)],
                #                 early_stopping_rounds=50,
                verbose=200)
        # sample shapley values
        fold_importance = None  # shap.TreeExplainer(clf).shap_values(vl_x[:shap_sampling])
    # out of fold predictions
    valid_prediticion = clf.predict(vl_x, num_iteration=clf.best_iteration_)
    oof_loss = np.sqrt(mean_squared_error(vl_y, valid_prediticion))  # target is already in log scale
    print(f'Fold:{i} RMSLE: {oof_loss:.4f} time={time.time() - t0:.5g}')

    return clf, fold_importance, valid_prediticion

#for (tr_idx, val_idx) in kf.split(train, train['building_id']):
for i in range(folds):
    tr_idx, val_idx= tr_idxs[i], val_idxs[i]
    clf, shap_values[val_idx[:shap_sampling]], oof_pred[val_idx] = fit_regressor(tr_idx, val_idx,i)
    input(".............")
    i = i + 1
    models.append(clf)
gc.collect()
oof_loss = np.sqrt(mean_squared_error(train[target], oof_pred)) # target is already in log scale
print(f'OOF RMSLE: {oof_loss:.4f}')

if isMORT:
    pass
elif False:
    _ = lgb.plot_importance(models[0], importance_type='gain')
    shap.summary_plot(shap_values, train[features], plot_type="bar")
    ma_shap = pd.DataFrame(sorted(zip(abs(shap_values).mean(axis=0), features), reverse=True),
                           columns=['Mean Abs Shapley', 'Feature']).set_index('Feature')
    # fig, ax = plt.subplots(figsize=(2,6))
    # _ = sns.heatmap(ma_shap, annot=True, cmap='Blues', fmt='.06f')
    print(ma_shap)
    shap.force_plot(shap.TreeExplainer(models[0]).expected_value, shap_values[0,:], train[features].iloc[0,:], matplotlib=True)
# load and pre-process test data

def recover_timestamp(x):
    ''' reassemble timestamp using date components '''
    return datetime.datetime.strptime(f'{x.year}-{x.month}-{x.day} {x.hour}', '%y-%m-%d %H')

set_size = len(test)
iterations = 50
batch_size = set_size // iterations

print(set_size, iterations, batch_size)
assert set_size == iterations * batch_size
meter_reading = []
for i in tqdm(range(iterations)):
    pos = i*batch_size
    fold_preds = [np.expm1(model.predict(test[features].iloc[pos : pos+batch_size])) for model in models]
    meter_reading.extend(np.mean(fold_preds, axis=0))
print(len(meter_reading))
assert len(meter_reading) == set_size

submission = pd.read_csv(f'{path}/sample_submission.csv')
submission['meter_reading'] = np.clip(meter_reading, a_min=0, a_max=None)
submission.to_csv('submission.csv', index=False)
submit_path = f'{path}/[{gbm}]_{some_rows}__.csv.gz'
submission.to_csv(submit_path, index=False,float_format='%.4f',compression='gzip')
print(submission.head(9))