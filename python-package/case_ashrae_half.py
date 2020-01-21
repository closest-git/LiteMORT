#https://www.kaggle.com/rohanrao/ashrae-divide-and-conquer
#https://www.kaggle.com/isaienkov/lightgbm-fe-1-19
'''
    [3,18,72]=0.937 [3,18]=0.939 [18]=0.946min_data_in_leaf [18,72] =0.933      [3,72] =0.938       [72] =0.933 [72,168]=0.939
    min_data_in_leaf:   20=0.933    200=0.932       2000=0.930      10000=0.9289    20000=0.930
'''
import gc
import os
import sys
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from pandas.api.types import is_datetime64_any_dtype as is_datetime
from pandas.api.types import is_categorical_dtype
import pickle
import litemort
from litemort import *
print(litemort.__version__)

data_root = 'F:/Datasets/ashrae/'
isMORT = len(sys.argv)>1 and sys.argv[1] == "mort"
#isMORT = False
isMerge = False #len(sys.argv)>1 and sys.argv[1] == "merge"
gbm='MORT' if isMORT else 'LGB'
use_building_id=False
leak_some_rows=2000000
#some_rows = 500
some_rows = None
lag_hours=[72]            #[3,18,72]内存溢出
use_building_id=True

def reduce_mem_usage(df, use_float16=False):
    """
    Iterate through all the columns of a dataframe and modify the data type to reduce memory usage.
    """

    start_mem = df.memory_usage().sum() / 1024 ** 2
    print("Memory usage of dataframe is {:.2f} MB".format(start_mem))

    for col in df.columns:
        if is_datetime(df[col]) or is_categorical_dtype(df[col]):
            continue
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
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
            df[col] = df[col].astype("category")

    end_mem = df.memory_usage().sum() / 1024 ** 2
    print("Memory usage after optimization is: {:.2f} MB".format(end_mem))
    print("Decreased by {:.1f}%".format(100 * (start_mem - end_mem) / start_mem))

    return df

def create_lag_features(df, window):
    """
    Creating lag-based features looking back in time.
    """

    feature_cols = ["air_temperature", "cloud_coverage", "dew_temperature", "precip_depth_1_hr", 'sea_level_pressure','wind_direction', 'wind_speed']
    df_site = df.groupby("site_id")

    df_rolled = df_site[feature_cols].rolling(window=window, min_periods=0)

    df_mean = df_rolled.mean().reset_index().astype(np.float16)
    df_median = df_rolled.median().reset_index().astype(np.float16)
    df_min = df_rolled.min().reset_index().astype(np.float16)
    df_max = df_rolled.max().reset_index().astype(np.float16)
#https://stackoverflow.com/questions/50306914/pandas-groupby-agg-std-nan
#pd.DataFrame.std assumes 1 degree of freedom by default, also known as sample standard deviation. This results in NaN results for groups with one number.
    df_std = df_rolled.std().reset_index().astype(np.float16).fillna(0)
    df_skew = df_rolled.skew().reset_index().astype(np.float16).fillna(0)

    for feature in feature_cols:
        df[f"{feature}_mean_lag{window}"] = df_mean[feature]
        df[f"{feature}_median_lag{window}"] = df_median[feature]
        df[f"{feature}_min_lag{window}"] = df_min[feature]
        df[f"{feature}_max_lag{window}"] = df_max[feature]
        df[f"{feature}_std_lag{window}"] = df_std[feature]
        df[f"{feature}_skew_lag{window}"] = df_skew[feature]
        # df[f"{feature}_skew_lag{window}"] = df_skew[feature]

    return df

def get_leak_():
    pkl_path = f'{data_root}/leak_.pickle'
    print("====== Load pickle @{} ......".format(pkl_path))
    if os.path.isfile(pkl_path):
        with open(pkl_path, "rb") as fp:
            [leak,leak_building_ids] = pickle.load(fp)
    else:
        leak0 = pd.read_csv(f"{data_root}/site0.csv")
        print(f"leak0={leak0.shape},mem={sys.getsizeof(leak0) / 1.0e6:.3f}")
        leak1 = pd.read_csv(f"{data_root}/site1.csv")
        leak2 = pd.read_csv(f"{data_root}/site2.csv")
        leak4 = pd.read_csv(f"{data_root}/site4.csv")
        leak15 = pd.read_csv(f"{data_root}/site15.csv")
        leak = pd.concat([leak0, leak1, leak2, leak4, leak15])
        leak["log_meter_reading"] = np.log1p(leak.meter_reading_scraped)
        leak = leak[leak.log_meter_reading.notnull()]
        if leak_some_rows is not None:
            leak, _ = Mort_PickSamples(leak_some_rows, leak, None)

        na_count = leak.isna().sum()
        del leak0, leak1, leak2, leak4, leak15
        gc.collect()
        leak_building_ids = leak.building_id.unique()
        print(f"leak={leak.shape} nNA={na_count} leak_buildings={len(leak_building_ids)}:\t{leak_building_ids}")

        test = pd.read_csv(path_test)
        shape_0=test.shape
        test = test[test.building_id.isin(leak_building_ids)]
        leak = leak.merge(test, on=["building_id", "meter", "timestamp"],how = "left")
        leak = reduce_mem_usage(leak, use_float16=True)
        print(f"test0={shape_0} test@leak={test.shape} leak={leak.shape},mem={sys.getsizeof(leak)/1.0e6:.3f} ")
        leak.drop(["row_id"],axis=1, inplace=True)
        del test
        gc.collect()
        with open(pkl_path, "wb") as fp:
            pickle.dump([leak,leak_building_ids], fp, protocol=4)

    return leak,leak_building_ids

path_data = data_root   #"/kaggle/input/ashrae-energy-prediction/"
path_train = path_data + "train.csv"
path_test = path_data + "test.csv"
path_building = path_data + "building_metadata.csv"
path_weather_train = path_data + "weather_train.csv"
path_weather_test = path_data + "weather_test.csv"
myfavouritenumber = 13
seed = myfavouritenumber
pkl_path = f'{data_root}/cys_{"merge"if isMerge else ""}_{lag_hours}_.pickle'
df_building_meter=None
df_building_meter_hour=None

def EDA_(df_,weather_,building,isTrain=False,isLogMeter=False):
    global df_building_meter,df_building_meter_hour
    df_ = df_.merge(building, on="building_id",how = "left")
    #print(f"{df_.shape}")
    # df_train = df_train.merge(weather_train, on=["site_id", "timestamp"], how="left")
    if isTrain:
        df_ = df_[~((df_.site_id == 0) & (df_.meter == 0) & (df_.building_id <= 104) & (
                    df_.timestamp < "2016-05-21"))]

    df_.reset_index(drop=True, inplace=True)
    #print(f"{df_.shape}")
    df_.timestamp = pd.to_datetime(df_.timestamp, format='%Y-%m-%d %H:%M:%S')
    if isLogMeter:
        df_["log_meter_reading"] = np.log1p(df_.meter_reading)
    # Feature Engineering: Time
    df_["hour"] = df_.timestamp.dt.hour
    df_["weekday"] = df_.timestamp.dt.weekday
    print(f"{df_.shape}")
    print("EDA_ Aggregation@building_meter@building_meter_hour...")
    if isTrain:
        df_building_meter = df_.groupby(["building_id", "meter"]). \
            agg(mean_building_meter=("log_meter_reading", "mean"),
                median_building_meter=("log_meter_reading", "median")).reset_index()
        df_building_meter_hour = df_.groupby(["building_id", "meter", "hour"]). \
            agg(mean_building_meter=("log_meter_reading", "mean"),
                median_building_meter=("log_meter_reading", "median")).reset_index()

    df_ = df_.merge(df_building_meter, on=["building_id", "meter"], how="left")
    df_ = df_.merge(df_building_meter_hour, on=["building_id", "meter", "hour"], how="left")
    print(f"{df_.shape}")
    if weather_ is not None:
        if isMerge:
            pass
        else:
            df_ = df_.merge(weather_, on=["site_id", "timestamp"], how="left")
    gc.collect()

    df_ = reduce_mem_usage(df_)
    print(f"{df_.shape}")
    # df_train = df_train.reindex(sorted(df_train.columns), axis=1)
    # df_test = df_test.reindex(sorted(df_test.columns), axis=1)
    return df_

if os.path.isfile(pkl_path):
    print("====== Load pickle @{} ......".format(pkl_path))
    with open(pkl_path, "rb") as fp:
        if isMerge:
            [df_train, df_leak_test,weather_test,weather_train] = pickle.load(fp)
            merge_infos = [{'on': ['site_id', 'timestamp'], 'dataset': weather_train, "desc": "weather"}]
        else:
            merge_infos = None
            [df_train, df_leak_test,df_test_,weather_test] = pickle.load(fp)
else:
    df_leak_test,leak_building_ids = get_leak_()
    df_leak_test.rename(columns={'meter_reading_scraped': 'meter_reading'}, inplace=True)
    df_train = pd.read_csv(path_train)
    #df_train = df_train[df_train.building_id.isin(leak_building_ids)]
    print(f"df_train={df_train.shape} cols={df_train.columns}")
    print(f"df_leak_test={df_leak_test.shape} cols={df_leak_test.columns}")

    building = pd.read_csv(path_building)
    le = LabelEncoder()
    building.primary_use = le.fit_transform(building.primary_use)
    weather_train = pd.read_csv(path_weather_train)
    weather_test = pd.read_csv(path_weather_test)
    #weather_train.drop(["sea_level_pressure", "wind_direction", "wind_speed"], axis=1, inplace=True)
    #weather_test.drop(["sea_level_pressure", "wind_direction", "wind_speed"], axis=1, inplace=True)
    weather_train = weather_train.groupby("site_id").apply(lambda group: group.interpolate(limit_direction="both"))
    weather_test = weather_test.groupby("site_id").apply(lambda group: group.interpolate(limit_direction="both"))
    print(f"df_train={df_train.shape} df_leak_test={df_leak_test.shape} weather_train={weather_train.shape} weather_test={weather_test.shape}")

    #df_train = reduce_mem_usage(df_train, use_float16=True)
    #df_test = reduce_mem_usage(df_test, use_float16=True)
    weather_train.timestamp = pd.to_datetime(weather_train.timestamp, format='%Y-%m-%d %H:%M:%S')
    weather_test.timestamp = pd.to_datetime(weather_test.timestamp, format='%Y-%m-%d %H:%M:%S')
    weather_train = reduce_mem_usage(weather_train, use_float16=True)
    weather_test = reduce_mem_usage(weather_test, use_float16=True)

    for h in lag_hours:
        weather_train = create_lag_features(weather_train, h)
        #weather_train.drop(["air_temperature", "cloud_coverage", "dew_temperature", "precip_depth_1_hr"], axis=1, inplace=True)
        weather_test = create_lag_features(weather_test, h)
        #weather_test.drop(["air_temperature", "cloud_coverage", "dew_temperature", "precip_depth_1_hr"], axis=1,inplace=True)

    df_train = EDA_(df_train, weather_train, building, isTrain=True, isLogMeter=True)
    df_test_ = pd.read_csv(path_test)
    print(df_test_.shape)
    df_test_ = EDA_(df_test_, None, building, isTrain=False, isLogMeter=False)
    print(df_test_.head(100));      print(df_test_.tail(100))
    df_leak_test = EDA_(df_leak_test, weather_test, building, isTrain=False,isLogMeter=True)

    del building, le;    del weather_train;
    gc.collect()
    print(f"df_train={df_train.shape} cols={df_train.columns}")
    print(f"df_leak_test={df_leak_test.shape} cols={df_leak_test.columns} NAN={df_leak_test.isna().sum()}")
    print(f"df_test_={df_test_.shape} cols={df_test_.columns} NAN={df_test_.isna().sum()}")
    with open(pkl_path, "wb") as fp:
        if isMerge:     #df_train=1172.185  df_test=2960.530 weather_test=18.021 weather_train=9.085 leak=253.434
            print(f"df_train={sys.getsizeof(df_train) / 1.0e6:.3f}  df_leak_test={sys.getsizeof(df_leak_test) / 1.0e6:.3f} weather_test={sys.getsizeof(weather_test) / 1.0e6:.3f}"
                  f"\tweather_train={sys.getsizeof(weather_train) / 1.0e6:.3f}")
            pickle.dump([df_train, df_leak_test, weather_test, weather_train], fp)
        else:
            pickle.dump([df_train, df_leak_test,df_test_,weather_test], fp, protocol=4)
    #sys.exit(-1)

categorical_features = ["primary_use","meter", "weekday","hour"]
if isMerge:
    all_features = [col for col in df_train.columns if col not in ["building_id","meter_reading", "log_meter_reading"]]
else:
    all_features = [col for col in df_train.columns if col not in ["building_id","timestamp", "site_id", "meter_reading", "log_meter_reading"]]

cv = 5
models = []
cv_scores = {"site_id": [], "cv_score": []}

early_stop = 21
verbose_eval = 20
metric = 'l2'
num_rounds=999; lr=0.049; bf=0.51;ff=0.81;nLeaf=41
#num_rounds = 30;    lr = 0.05;  verbose_eval=1;     bf=1;   ff=1;nLeaf=41

params = {"objective": "regression", "metric": "rmse",
          "num_leaves": nLeaf, "learning_rate": 0.049,'n_estimators': num_rounds,
          "bagging_freq": 1, "bagging_fraction": bf, "feature_fraction": ff,'min_data_in_leaf': 10000,
            'verbose_eval': verbose_eval,"early_stopping_rounds": early_stop,'n_jobs': 8, "elitism": 0
          }

kf = KFold(n_splits=cv, random_state=seed)

#X_train = df_train;
if some_rows is not None:
    df_train, _ = Mort_PickSamples(some_rows, df_train, None)
    df_leak_test, _ = Mort_PickSamples(some_rows, df_leak_test, None)
X_leak = df_leak_test;  y_leak = X_leak.log_meter_reading
Y_train = df_train.log_meter_reading
#X_train = X_train[all_features]
X_leak = X_leak[all_features]
y_pred_train_site = np.zeros(df_train.shape[0])
score = 0

for fold, (train_index, valid_index) in enumerate(kf.split(df_train, Y_train)):
    X_train, X_valid = df_train[all_features].loc[train_index], X_leak
    y_train, y_valid = Y_train.iloc[train_index], y_leak
    if fold==0:
        print(f"X_train={X_train.shape} cols={X_train.columns}")
        print(f"X_valid={X_leak.shape} cols={X_valid.columns}")

    if isMORT:
        params['verbose'] = 667 if fold == 0 else 1
        merge_datas = []
        model = LiteMORT(params,merge_infos=merge_infos)
        model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], categorical_feature=categorical_features)
    else:
        dtrain = lgb.Dataset(X_train, label=y_train, categorical_feature=categorical_features)
        dvalid = lgb.Dataset(X_valid, label=y_valid, categorical_feature=categorical_features)
        watchlist = [dtrain, dvalid]
        model = lgb.train(params, train_set=dtrain, num_boost_round=num_rounds, valid_sets=watchlist, verbose_eval=verbose_eval,
                              early_stopping_rounds=early_stop)
    models.append(model)
    y_pred_valid = model.predict(X_valid, num_iteration=model.best_iteration)
    #y_pred_train_site[valid_index] = y_pred_valid

    rmse = np.sqrt(mean_squared_error(y_valid, y_pred_valid))
    print(", Fold:", fold + 1, ", RMSE:", rmse)
    score += rmse / cv
    del  X_train, y_train
    gc.collect()
    #input("......")
del  X_leak, y_leak
gc.collect()
cv_scores["cv_score"].append(score)

del df_train,df_leak_test
gc.collect()

merge_test_info = [{'on': ['site_id', 'timestamp'], 'dataset': weather_test, "desc": "weather"}]
submit = pd.read_csv(os.path.join(data_root, 'sample_submission.csv'))

def pred(X_test, models, batch_size=1000000):
    iterations = (X_test.shape[0] + batch_size -1) // batch_size
    print('iterations', iterations)

    y_test_pred_total = np.zeros(X_test.shape[0])
    for i, model in enumerate(models):
        print(f'predicting {i}-th model')
        if isMORT and isMerge:
            y_pred_test = model.predict(X_test, num_iteration=model.best_iteration)
            y_test_pred_total += y_pred_test
        else:
            for k in tqdm(range(iterations)):
            #for k in range(1):
                y_pred_test = model.predict(X_test[k*batch_size:(k+1)*batch_size], num_iteration=model.best_iteration)
                #print(y_pred_test[:100]);            input("pred ......")
                y_test_pred_total[k*batch_size:(k+1)*batch_size] += y_pred_test

    y_test_pred_total /= len(models)
    return y_test_pred_total

if isMerge:
    pass
else:
    print("Before Merge::df_test_={df_test_.shape}")
    df_test_ = df_test_.merge(weather_test, on=["site_id", "timestamp"], how="left")

submit.row_id = df_test_.row_id
df_test_ = df_test_[all_features]
print(f"df_test_={df_test_.shape} cols={df_test_.columns} NAN={df_test_.isna().sum()} submit={submit.shape}")
print(f"df_test_={df_test_.head()} {df_test_.tail()}")

assert(submit.shape[0]==df_test_.shape[0])
y_pred_ = pred(df_test_,models)

submit.meter_reading = np.clip(np.expm1(y_pred_), 0, a_max=None)
print(submit.head(100),submit.tail(100))
submit.to_csv("submission_cys_noleak.csv.gz", index=False, float_format='%.4f',compression='gzip')

if False:
    submit = submit.merge(leak[["row_id", "meter_reading_scraped"]], on=["row_id"], how="left")
    submit.loc[submit.meter_reading_scraped.notnull(), "meter_reading"] = submit.loc[submit.meter_reading_scraped.notnull(), "meter_reading_scraped"]
    submit.drop(["meter_reading_scraped"], axis=1, inplace=True)

    submit.to_csv("submission.csv.gz", index=False, float_format='%.4f',compression='gzip')