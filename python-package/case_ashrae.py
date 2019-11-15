# https://www.kaggle.com/corochann/ashrae-training-lgbm-by-meter-type
import sys
import pickle
import seaborn as sns; sns.set()
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, GroupKFold,KFold
from sklearn.metrics import log_loss, mean_squared_error
from litemort import *
#from LiteMORT_hyppo import *
import datetime
import time
import random
import gc
import pandas as pd
import numpy as np
import lightgbm as lgb
# import shap as shap
import os
from tqdm import tqdm
data_root = 'F:/Datasets/ashrae/'
from pandas.api.types import is_datetime64_any_dtype as is_datetime
from pandas.api.types import is_categorical_dtype

isMORT = len(sys.argv)>1 and sys.argv[1] == "mort"
isMORT = True
isMerge = len(sys.argv)>1 and sys.argv[1] == "merge"
gbm='MORT' if isMORT else 'LGB'

print(f"====== MERGE={isMerge} gbm={gbm} ======\n\n")

def reduce_mem_usage(df, use_float16=False):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.
    """
    start_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        if is_datetime(df[col]) or is_categorical_dtype(df[col]):
            # skip datetime type or categorical type
            continue
        col_type = df[col].dtype

        if col_type != object:
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
                if use_float16 and c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df

def LoadBuilding(data_root):
    building_meta_df = pd.read_csv(f'{data_root}/building_metadata.csv')
    primary_use_list = building_meta_df['primary_use'].unique()
    primary_use_dict = {key: value for value, key in enumerate(primary_use_list)}
    print('primary_use_dict: ', primary_use_dict)
    building_meta_df['primary_use'] = building_meta_df['primary_use'].map(primary_use_dict)
    print(f"{building_meta_df.shape}\n{building_meta_df.head()}")
    return building_meta_df

class Whether(object):
    def __init__(self, source, data_root,params=None):
        self.source = source
        self.data_root = data_root
        self.pkl_path = f'{data_root}/_ashrae_whether_{source}_.pickle'

    def df(self):
        if os.path.isfile(self.pkl_path):
            print("====== Load pickle @{} ......".format(self.pkl_path))
            with open(self.pkl_path, "rb") as fp:
                [whether_df] = pickle.load(fp)
            return whether_df
        else:
            weather_train_df = pd.read_csv(f'{self.data_root}/weather_{self.source}.csv', parse_dates=['timestamp'],
                        dtype={'site_id': np.uint8, 'air_temperature': np.float16,
                               'cloud_coverage': np.float16, 'dew_temperature': np.float16,'precip_depth_1_hr': np.float16})
            print(f"{weather_train_df.shape}\n{weather_train_df.isna().sum()}")
            w_sum = weather_train_df.groupby('site_id').apply(lambda group: group.isna().sum())
            weather_train_df = weather_train_df.groupby('site_id').apply(
                lambda group: group.interpolate(limit_direction='both'))
            w_sum = weather_train_df.groupby('site_id').apply(lambda group: group.isna().sum())
            self.add_lag_feature(weather_train_df, window=3)
            self.add_lag_feature(weather_train_df, window=72)
            print(weather_train_df.head(), weather_train_df.columns)
            with open(self.pkl_path, "wb") as fp:
                pickle.dump([weather_train_df], fp)
            return weather_train_df

    def add_lag_feature(self,weather_df, window=3):
        group_df = weather_df.groupby('site_id')
        cols = ['air_temperature', 'cloud_coverage', 'dew_temperature', 'precip_depth_1_hr', 'sea_level_pressure',
                'wind_direction', 'wind_speed']
        rolled = group_df[cols].rolling(window=window, min_periods=0)
        lag_mean = rolled.mean().reset_index().astype(np.float16)
        lag_max = rolled.max().reset_index().astype(np.float16)
        lag_min = rolled.min().reset_index().astype(np.float16)
        lag_std = rolled.std().reset_index().astype(np.float16)
        for col in cols:
            weather_df[f'{col}_mean_lag{window}'] = lag_mean[col]
            weather_df[f'{col}_max_lag{window}'] = lag_max[col]
            weather_df[f'{col}_min_lag{window}'] = lag_min[col]
            weather_df[f'{col}_std_lag{window}'] = lag_std[col]

class COROchann(object):
    @classmethod
    def __init__(self, source,data_root,building_meta_df,weather_df):
        self.category_cols = ['building_id', 'site_id', 'primary_use']  # , 'meter'
        self.feature_cols = ['square_feet', 'year_built'] + [
            'hour', 'weekend',  # 'month' , 'dayofweek'
            'building_median'] + [
                                'air_temperature', 'cloud_coverage',
                                'dew_temperature', 'precip_depth_1_hr', 'sea_level_pressure',
                                'wind_direction', 'wind_speed', 'air_temperature_mean_lag72',
                                'air_temperature_max_lag72', 'air_temperature_min_lag72',
                                'air_temperature_std_lag72', 'cloud_coverage_mean_lag72',
                                'dew_temperature_mean_lag72', 'precip_depth_1_hr_mean_lag72',
                                'sea_level_pressure_mean_lag72', 'wind_direction_mean_lag72',
                                'wind_speed_mean_lag72', 'air_temperature_mean_lag3',
                                'air_temperature_max_lag3',
                                'air_temperature_min_lag3', 'cloud_coverage_mean_lag3',
                                'dew_temperature_mean_lag3',
                                'precip_depth_1_hr_mean_lag3', 'sea_level_pressure_mean_lag3',
                                'wind_direction_mean_lag3', 'wind_speed_mean_lag3']

        self.source = source
        self.data_root = data_root
        self.building_meta_df = building_meta_df
        self.weather_df = weather_df
        #self.some_rows = 5000
        self.some_rows = None
        self.df_base = self.Load_Processing()
        self.df_base_shape = self.df_base.shape

    def fit(cls, df):
        pass

    def data_X_y(self,target_meter):
        self.building_meta_df.drop('floor_count', axis=1, inplace=True)
        self.building_meta_df['primary_use'] = self.building_meta_df.primary_use.astype('category')
        feat_v0 = self.feature_cols + self.category_cols
        feat_infos = {"categorical": self.category_cols}
        train_df = self.df_base
        print(f"{self.source}_X_y@{target_meter} df_base={train_df.shape}......")
        pkl_path = f'{data_root}/_ashrae_{self.source}_T{target_meter}_{self.some_rows}_M[{isMerge}]_.pickle'
        if isMerge:
            feat_v0 = feat_v0 + ['timestamp']
            #self.weather_df = self.weather_df[:1000]
            feat_v1 = list(set(feat_v0).intersection(set(list(self.weather_df.columns))))
            #feat_v1 = ['site_id','timestamp','precip_depth_1_hr']       #测试需要
            self.weather_df = self.weather_df[feat_v1]
            self.merge_infos = [
                {'on': ['site_id', 'timestamp'], 'dataset': self.weather_df, "desc": "weather"},
                {'on': ['building_id'], 'dataset': self.building_meta_df, "desc": "building","feat_info": feat_infos},
            ]
        else:
            self.merge_infos = []

        if False:#os.path.isfile(pkl_path):
            print("====== Load pickle @{} ......".format(pkl_path))
            with open(pkl_path, "rb") as fp:
                [X_train, y_train] = pickle.load(fp)
        else:
            target_train_df = train_df[train_df['meter'] == target_meter]
            print(f"target@{target_meter}={target_train_df.shape}")
            building_site=self.building_meta_df[['building_id','site_id']]
            self.building_meta_df.drop(['site_id'], axis=1, inplace=True)
            target_train_df = target_train_df.merge(building_site, on='building_id', how='left')    #add 'site_id'
            #target_train_df = target_train_df.merge(self.weather_df, on=['site_id', 'timestamp'], how='left')

            if isMerge:
                pass#
            else:
                target_train_df = target_train_df.merge(self.building_meta_df, on='building_id', how='left')
                target_train_df = target_train_df.merge(self.weather_df, on=['site_id', 'timestamp'], how='left')
            feat_v1 = list(set(feat_v0).intersection(set(list(target_train_df.columns))))
            X_train = target_train_df[feat_v1]
            print(f"data_X__@{target_meter}={X_train.shape}\toriginal={target_train_df.shape}\tmerge={isMerge}")
            if (self.source == "train"):
                y_train = target_train_df['meter_reading_log1p'].values
            else:
                y_train=None
            del target_train_df
            gc.collect()
            with open(pkl_path, "wb") as fp:
                pickle.dump([X_train, y_train], fp)
        return X_train, y_train

    @classmethod
    def Load_Processing(self):
        pkl_path = f'{self.data_root}/_ashrae_{self.source}_{self.some_rows}_.pickle'
        build_group_pkl = f'{self.data_root}/_ashrae_build_group_{self.some_rows}_.pickle'
        if os.path.isfile(pkl_path):
            print("====== Load pickle @{} ......".format(pkl_path))
            with open(pkl_path, "rb") as fp:
                [df] = pickle.load(fp)
        else:
            df = pd.read_csv(f'{self.data_root}/{self.source}.csv', dtype={'building_id': np.uint16, 'meter': np.uint8},
                             parse_dates=['timestamp'])
            if self.source=="train":
                df = df.query('not (building_id <= 104 & meter == 0 & timestamp <= "2016-05-20")')
                if self.some_rows is not None:
                    df, _ = Mort_PickSamples(self.some_rows, df, None)
                    print(f'====== Some Samples@{self.source} ... data={df.shape}')
            #df['date'] = df['timestamp'].dt.date
            df["hour"] = np.uint8(df["timestamp"].dt.hour)
            df["weekend"] = np.uint8(df["timestamp"].dt.weekday)    #cys
            df["month"] = np.uint8(df["timestamp"].dt.month)
            df["dayofweek"] = np.uint8(df["timestamp"].dt.dayofweek)
            if self.source == "train":
                df['meter_reading_log1p'] = np.log1p(df['meter_reading'])
                df_group = df.groupby('building_id')['meter_reading_log1p']
                building_mean = df_group.mean().astype(np.float16)
                building_median = df_group.median().astype(np.float16)
                building_min = df_group.min().astype(np.float16)
                building_max = df_group.max().astype(np.float16)
                building_std = df_group.std().astype(np.float16)
                with open(build_group_pkl, "wb") as fp:
                    pickle.dump([building_mean,building_median,building_min,building_max,building_std], fp)
                print(building_mean.head())
            else:
                with open(build_group_pkl, "rb") as fp:
                    [building_mean,building_median,building_min,building_max,building_std]=pickle.load(fp)

            df['building_mean'] = df['building_id'].map(building_mean)
            df['building_median'] = df['building_id'].map(building_median)
            df['building_min'] = df['building_id'].map(building_min)
            df['building_max'] = df['building_id'].map(building_max)
            df['building_std'] = df['building_id'].map(building_std)
            print(df.head())
            with open(pkl_path, "wb") as fp:
                pickle.dump([df], fp)
        return df

building_meta_df=LoadBuilding(data_root)
weather_train_df = Whether('train', data_root).df()
print(weather_train_df.head())
weather_test_df = Whether('test', data_root).df()
print(weather_test_df.head())
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
              "bagging_freq": 1,
              "bagging_fraction": bf,
              "feature_fraction": 1,  # STRANGE GBDT  why("bagging_freq": 5 "feature_fraction": 0.9)!!!
              "metric": metric, "verbose_eval": verbose_eval, 'n_jobs': 8, "elitism": 0,"debug":'1',
              "early_stopping_rounds": early_stop, "adaptive": 'weight1', 'verbose': 666, 'min_data_in_leaf': 20,
              #               "verbosity": -1,
              #               'reg_alpha': 0.1,
              #               'reg_lambda': 0.3
              }

def fit_regressor(train, val,target_meter,fold, some_params, devices=(-1,), merge_info=None, cat_features=None):
    t0=time.time()
    X_train, y_train = train
    X_valid, y_valid = val

    device = devices[0]
    if device == -1:        # use cpu
        pass
    else:        # use gpu
        print(f'using gpu device_id {device}...')
        params.update({'device': 'gpu', 'gpu_device_id': device})


    if False:
        col_y = pd.DataFrame(y_train)
        col_X = X_train.reset_index(drop=True)
        d_train = pd.concat([col_y, col_X], ignore_index=True, axis=1)
        np.savetxt("E:/2/LightGBM-master/examples/regression/case_cys_.csv", d_train, delimiter='\t')
        print("X_train={}, y_train={} d_train={}".format(col_X.shape, col_y.shape, d_train.shape))

    if isMORT:
        params['verbose']=667
        merge_datas=[]
        model = LiteMORT(some_params,merge_infos=merge_info)   # all train,eval,predict would use same merge infomation
        model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], categorical_feature=cat_features)
        fold_importance = None
        log = ""
    else:
        d_train = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_features)
        d_valid = lgb.Dataset(X_valid, label=y_valid, categorical_feature=cat_features)
        watchlist = [d_train, d_valid]
        print('training LGB: parmas=',params)
        model = lgb.train(some_params,
                          train_set=d_train,
                          num_boost_round=num_rounds,
                          valid_sets=watchlist,
                          verbose_eval=verbose_eval,
                          early_stopping_rounds=early_stop)
        print('best_score', model.best_score)
        log = {'train/mae': model.best_score['training'][metric], 'valid/mae': model.best_score['valid_1'][metric]}
    # predictions
    y_pred_valid = model.predict(X_valid, num_iteration=model.best_iteration)
    oof_loss = mean_squared_error(y_valid, y_pred_valid)  # target is already in log scale
    print(f'METER:{target_meter} Fold:{fold} MSE: {oof_loss:.4f} time={time.time() - t0:.5g}', flush=True)
    #input("......")
    os._exit(-200)      #
    return model, y_pred_valid, log

folds = 5
seed = 666
shuffle = False
kf = KFold(n_splits=folds, shuffle=shuffle, random_state=seed)
#kf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)

cat_features=None
meter_models=[]
train_datas = COROchann("train",data_root,building_meta_df,weather_train_df)
losses=[]
feat_fix = ['building_id','building_median','hour','weekend','site_id', 'square_feet','primary_use','air_temperature', 'year_built']
def GetSplit_idxs(kf,X_train, y_train):
    split_idxs=[]
    for train_idx, valid_idx in kf.split(X_train, y_train):
        #train_idx=list(train_idx)
        #alid_idx=list(valid_idx)
        split_idxs.append((train_idx, valid_idx))
    return split_idxs

for target_meter in range(4):
    X_train, y_train = train_datas.data_X_y(target_meter)
    split_ids=GetSplit_idxs(kf,X_train, y_train)
    #X_train = X_train[feat_fix]
    y_valid_pred_total = np.zeros(X_train.shape[0])
    gc.collect()
    print(f'target_meter={target_meter} X_train={X_train.shape}')
    cat_features = train_datas.category_cols
    # cat_features = ['building_id']
    # [X_train.columns.get_loc(cat_col) for cat_col in train_datas.category_cols]
    print('cat_features', cat_features)
    if False :
        feat_select = X_train.columns
        feat_select = list(set(feat_select) - set(feat_fix))
        params['split_idxs'] = split_ids
        params['early_stopping_rounds'] = 50        #不宜太大，掉到坑里
        params['category_features'] = cat_features
        MORT_feat_select_(X_train, y_train, feat_fix, feat_select,params,nMostSelect=(int)(len(feat_select)/2))
        input("......MORT_feat_search......")
        sys.exit(-100)
    feat_useful_ =feat_fix+ ['dew_temperature_mean_lag72', 'wind_direction_mean_lag72', 'cloud_coverage_mean_lag72',
                    'precip_depth_1_hr_mean_lag72', 'air_temperature_std_lag72', 'cloud_coverage_mean_lag3',
                    'air_temperature_min_lag72', 'wind_direction', 'wind_speed_mean_lag72', 'air_temperature_max_lag3',
                    'wind_speed_mean_lag3', 'dew_temperature']
    #X_train = X_train[feat_useful_]        #0.4499->0.4496 略有提高

    t0=time.time()
    fold = 0
    models_ = []
    for train_idx, valid_idx in kf.split(X_train, y_train):
    #for (train_idx, valid_idx) in kf.split(X_train, X_train['building_id']):
        train_data = X_train.iloc[train_idx, :], y_train[train_idx]
        valid_data = X_train.iloc[valid_idx, :], y_train[valid_idx]
        params['seed'] = seed
        print(f'fold={fold} train={train_data[0].shape},valid={valid_data[0].shape}')
        #     model, y_pred_valid, log = fit_cb(train_data, valid_data, cat_features=cat_features, devices=[0,])
        model, y_pred_valid, log = fit_regressor(train_data, valid_data,target_meter,fold,some_params=params,merge_info=train_datas.merge_infos, cat_features=cat_features)
        y_valid_pred_total[valid_idx] = y_pred_valid
        models_.append(model)
        gc.collect()
        fold=fold+1
        #break
    meter_loss = mean_squared_error(y_train, y_valid_pred_total)
    print(f'======METER:{target_meter} MSE: {meter_loss:.4f} time={time.time() - t0:.5g}\n')
    losses.append(meter_loss)
    meter_models.append(models_)
    sns.distplot(y_train)
    del X_train, y_train
    gc.collect()
    #break

def pred(X_test, models, batch_size=1000000):
    iterations = (X_test.shape[0] + batch_size -1) // batch_size
    print('iterations', iterations)

    y_test_pred_total = np.zeros(X_test.shape[0])
    for i, model in enumerate(models):
        print(f'predicting {i}-th model')
        for k in tqdm(range(iterations)):
            y_pred_test = model.predict(X_test[k*batch_size:(k+1)*batch_size], num_iteration=model.best_iteration)
            y_test_pred_total[k*batch_size:(k+1)*batch_size] += y_pred_test

    y_test_pred_total /= len(models)
    return y_test_pred_total

test_datas = COROchann("test",data_root,building_meta_df,weather_test_df)
test_df = test_datas.df_base
sample_submission = pd.read_csv(os.path.join(data_root, 'sample_submission.csv'))
reduce_mem_usage(sample_submission)
for target_meter in range(4):
    X_test,_ = test_datas.data_X_y(target_meter)
    gc.collect()
    y_test0 = pred(X_test, meter_models[target_meter])
    sns.distplot(y_test0)
    sample_submission.loc[test_df['meter'] == target_meter, 'meter_reading'] = np.expm1(y_test0)
    del X_test
    gc.collect()

submit_path = f'{data_root}/[{gbm}]_[{losses}].csv.gz'
sample_submission.to_csv(submit_path, index=False, float_format='%.4f',compression='gzip')
print(sample_submission.head())