#https://www.kaggle.com/hmendonca/shapley-values-for-feature-selection-ashrae
#https://www.kaggle.com/robikscube/ashrae-energy-consumption-starter-kit-eda
'''
天气修正
    https://www.kaggle.com/c/ashrae-energy-prediction/discussion/114447#latest-658783
    https://www.kaggle.com/c/ashrae-energy-prediction/discussion/114483#latest-659028

CV
    https://www.kaggle.com/c/ashrae-energy-prediction/discussion/113644#latest-658931

'''

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
#unimportant_cols = ['wind_direction', 'wind_speed', 'sea_level_pressure']
target = 'meter_reading'

isMORT = len(sys.argv)>1 and sys.argv[1] == "mort"
#isMORT = True
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
                                                                  'precip_depth_1_hr':np.float16})
    df = pd.read_csv(f'{path}/{source}.csv', dtype={'building_id':np.uint16, 'meter':np.uint8}, parse_dates=['timestamp'])
    df = df.merge(building, on='building_id', how='left')
    df = df.merge(weather, on=['site_id', 'timestamp'], how='left')
    return df

def degToCompass(num):
    val=int((num/22.5)+.5)
    arr=[i for i in range(0,16)]
    return arr[(val % 16)]

def relative_humidity(Tc,Tdc):
    E = 6.11*10.0**(7.5*Tdc/(237.7+Tdc))
    Es = 6.11*10.0**(7.5*Tc/(237.7+Tc))
    RH = (E/Es)*100
    return RH

def average_imputation(df, column_name):
    t0=time.time()
    imputation = df.groupby(['timestamp'])[column_name].mean()

    df.loc[df[column_name].isnull(), column_name] = df[df[column_name].isnull()][[column_name]].apply(
        lambda x: imputation[df['timestamp'][x.index]].values)
    del imputation
    print(f"average_imputation@{column_name} time={time.time()-t0:.5g}")
    return df

class ASHRAE3Preprocessor(object):
    @classmethod
    def fit(cls, df, data_ratios):
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
        df['year'] = np.uint8(df['timestamp'].dt.year - 1900)
        # parse and cast columns to a smaller type
        df.rename(columns={"square_feet": "log_square_feet"}, inplace=True)
        df['log_square_feet'] = np.float16(np.log(df['log_square_feet']))
        df['year_built'] = np.uint8(df['year_built'] - 1900)
        df['floor_count'] = np.uint8(df['floor_count'])
        #df['humidity'] = relative_humidity(df.air_temperature, df.dew_temperature).astype(np.float16)
        df = average_imputation(df, 'wind_speed')
        df = average_imputation(df, 'wind_direction')
        beaufort = [(0, 0, 0.3), (1, 0.3, 1.6), (2, 1.6, 3.4), (3, 3.4, 5.5), (4, 5.5, 8), (5, 8, 10.8),
                    (6, 10.8, 13.9),
                    (7, 13.9, 17.2), (8, 17.2, 20.8), (9, 20.8, 24.5), (10, 24.5, 28.5), (11, 28.5, 33), (12, 33, 200)]
        for item in beaufort:
            df.loc[(df['wind_speed'] >= item[1]) & (df['wind_speed'] < item[2]), 'beaufort_scale'] = item[0]
        df['wind_direction'] = df['wind_direction'].apply(degToCompass)
        df['beaufort_scale'] = df['beaufort_scale'].astype(np.uint8)
        df["wind_direction"] = df['wind_direction'].astype(np.uint8)
        df["meter"] = df['meter'].astype(np.uint8)
        df["site_id"] = df['site_id'].astype(np.uint8)

        # remove redundant columns
        for col in df.columns:
            if col in ['row_id',"sea_level_pressure", "wind_speed"]:      #['timestamp', 'row_id']:
                del df[col]

        # extract target column
        if 'meter_reading' in df.columns:
            df['meter_reading'] = np.log1p(df['meter_reading']).astype(np.float32)  # comp metric uses log errors
        categoricals = ["site_id", "building_id", "primary_use", "hour", "weekday", "meter", "wind_direction"]
        numericals = ["square_feet", "year_built", "air_temperature", "cloud_coverage",
                      "dew_temperature", 'precip_depth_1_hr', 'floor_count', 'beaufort_scale']
        return df,categoricals,numericals

if os.path.isfile(pkl_path):
    print("====== Load pickle @{} ......".format(pkl_path))
    with open(pkl_path, "rb") as fp:
        [train,test,features,category_feat,split_idxs] = pickle.load(fp)
    print(f"train={train.shape}, nFold={len(split_idxs)},features={features}")
else:   # load and display some samples
    split_idxs=[]
    train = load_data('train')
    if some_rows is not None:
        train,_ = Mort_PickSamples(some_rows,train,None)
        print('====== Some Samples ... data={}'.format(train.shape))
    print(train.sample(5))

    data_ratios = train.count()/len(train)
    print("Ratio of available data (not NAN's):",data_ratios)
    print(train.loc[:, data_ratios < 1.0].mean())
    ASHRAE3Preprocessor.fit(train,data_ratios)
    print('ASHRAE3Preprocessor.transform......')
    train,category_feat,numerical_feat = ASHRAE3Preprocessor.transform(train)
    train = train.sort_index(by='timestamp').reset_index(drop=True)
    nSamp = len(train)
    if False:
        nTrain = int(nSamp * 0.5)
        tr_idx, val_idx=train.iloc[:nTrain].index.tolist(), train.iloc[nTrain:].index.tolist()
    else:
        mask = (train.month >=2) & (train.month <8)
        tr_idx = train[mask].index.tolist()
        val_idx = train[~mask].index.tolist()
    split_idxs.append((tr_idx, val_idx))
    split_idxs.append((val_idx,tr_idx))
    print(train.sample(7))
    print(train.dtypes)
    features = [col for col in train.columns if col not in [target, 'year', 'month', 'day','timestamp']]

    if False:
        test = load_data('test')
        print("Ratio of available data (not NAN's):",test.count() / len(test))
        test = ASHRAE3Preprocessor.transform(test)
        print(test.sample(5))
    else:
        test = None

    with open(pkl_path, "wb") as fp:  # Pickling
        pickle.dump([train,test,features,category_feat,split_idxs], fp)
        print("====== Dump pickle @{} ......OK".format(pkl_path))
        print(f"train={train.shape}, nFold={len(split_idxs)},features={features}")
        print(f"category_feat={category_feat}")

#kf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
models = []
shap_values = np.zeros(train[features].shape)
shap_sampling = 125000  # reduce compute cost
oof_pred = np.zeros(train.shape[0])  # out of fold predictions

params={ 'application': 'regression',
    'n_estimators':9000,'learning_rate':0.33,'feature_fraction':0.9,'subsample':0.25,  # batches of 25% of the data
    'subsample_freq':1,'num_leaves':20,'lambda_l1':0.5,'lambda_l2':0.5,'metric':'rmse','n_jobs':-1,
    "adaptive":'weight1','verbose': 666,
}
## stratify data by building_id
#for i, (tr_idx, val_idx) in tqdm(enumerate(kf.split(train, train['building_id'])), total=folds):
def fit_regressor(tr_idx, val_idx,features, i):  # memory closure
    t0 = time.time()
    if isMORT and False:
        features = features+['day']
        params['representive'] = {'day':10}

    tr_x, tr_y = train[features].iloc[tr_idx], train[target].iloc[tr_idx]
    vl_x, vl_y = train[features].iloc[val_idx], train[target].iloc[val_idx]
    print({'fold': i, 'train size': tr_x.shape, 'eval size': vl_x.shape})
    if isMORT:
        clf = LiteMORT(params).fit(tr_x, tr_y, eval_set=[(vl_x, vl_y)])
        fold_importance = None
    else:
        tr_data = lgb.Dataset(tr_x, label=tr_y)
        vl_data = lgb.Dataset(vl_x, label=vl_y)
        clf = lgb.LGBMRegressor(n_estimators=9000,
                                learning_rate=0.33,
                                feature_fraction=0.9,
                                subsample=0.25,  # batches of 25% of the data
                                subsample_freq=1,
                                num_leaves=20,
                                lambda_l1=0.5,  # regularisation
                                lambda_l2=0.5,
                                seed=i,   # seed diversification
                                metric='rmse')
        clf.fit(tr_x, tr_y,categorical_feature=category_feat,
                eval_set=[(vl_x, vl_y)],
                early_stopping_rounds=100,
                verbose=150)
        # sample shapley values
        fold_importance = None  # shap.TreeExplainer(clf).shap_values(vl_x[:shap_sampling])
    # out of fold predictions
    valid_prediticion = clf.predict(vl_x, num_iteration=clf.best_iteration_)
    oof_loss = np.sqrt(mean_squared_error(vl_y, valid_prediticion))  # target is already in log scale
    print(f'Fold:{i} RMSLE: {oof_loss:.4f} time={time.time() - t0:.5g}')

    return clf, fold_importance, oof_loss

#for (tr_idx, val_idx) in kf.split(train, train['building_id']):
scores=[]
for i,(tr_idx, val_idx) in enumerate(split_idxs):
    clf, shap_values[val_idx[:shap_sampling]], score = fit_regressor(tr_idx, val_idx,features,i)
    scores.append(score)
    i = i + 1
    models.append(clf)
gc.collect()
#oof_loss = np.sqrt(mean_squared_error(train[target], oof_pred)) # target is already in log scale
print(f'OOF RMSLE: {scores}')
input(".............")

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