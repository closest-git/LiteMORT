#https://www.kaggle.com/rohanrao/ashrae-divide-and-conquer
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
isMORT = False
isMerge = True #len(sys.argv)>1 and sys.argv[1] == "merge"
gbm='MORT' if isMORT else 'LGB'
use_ucf=False
some_rows = 500000
some_rows = None

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

    feature_cols = ["air_temperature", "cloud_coverage", "dew_temperature", "precip_depth_1_hr"]
    df_site = df.groupby("site_id")

    df_rolled = df_site[feature_cols].rolling(window=window, min_periods=0)

    df_mean = df_rolled.mean().reset_index().astype(np.float16)
    df_median = df_rolled.median().reset_index().astype(np.float16)
    df_min = df_rolled.min().reset_index().astype(np.float16)
    df_max = df_rolled.max().reset_index().astype(np.float16)
    df_std = df_rolled.std().reset_index().astype(np.float16)
    df_skew = df_rolled.skew().reset_index().astype(np.float16)

    for feature in feature_cols:
        df[f"{feature}_mean_lag{window}"] = df_mean[feature]
        df[f"{feature}_median_lag{window}"] = df_median[feature]
        df[f"{feature}_min_lag{window}"] = df_min[feature]
        df[f"{feature}_max_lag{window}"] = df_max[feature]
        df[f"{feature}_std_lag{window}"] = df_std[feature]
        df[f"{feature}_skew_lag{window}"] = df_skew[feature]
        # df[f"{feature}_skew_lag{window}"] = df_skew[feature]

    return df

def get_leak():
    ## adding leak
    leak0 = pd.read_csv(f"{data_root}/site0.csv")
    leak1 = pd.read_csv(f"{data_root}/site1.csv")
    leak2 = pd.read_csv(f"{data_root}/site2.csv")
    leak4 = pd.read_csv(f"{data_root}/site4.csv")
    leak15 = pd.read_csv(f"{data_root}/site15.csv")
    leak = pd.concat([leak0, leak1, leak2, leak4, leak15])
    del leak0, leak1, leak2, leak4, leak15
    gc.collect()
    print(f"leak={leak.shape} ")

    test = pd.read_csv(path_test)
    test = test[test.building_id.isin(leak.building_id.unique())]
    leak = leak.merge(test, on=["building_id", "meter", "timestamp"])
    print(f"test={test.shape} leak={leak.shape} ")
    del test
    gc.collect()
    return leak

path_data = data_root   #"/kaggle/input/ashrae-energy-prediction/"
path_train = path_data + "train.csv"
path_test = path_data + "test.csv"
path_building = path_data + "building_metadata.csv"
path_weather_train = path_data + "weather_train.csv"
path_weather_test = path_data + "weather_test.csv"
myfavouritenumber = 13
seed = myfavouritenumber
pkl_path = f'{data_root}/site_divide_{some_rows}_.pickle'

if os.path.isfile(pkl_path):
    print("====== Load pickle @{} ......".format(pkl_path))
    with open(pkl_path, "rb") as fp:
        [df_train, df_test,weather_test,leak] = pickle.load(fp)
else:
    leak = get_leak()
    df_train = pd.read_csv(path_train)
    df_test = pd.read_csv(path_test)

    building = pd.read_csv(path_building)
    le = LabelEncoder()
    building.primary_use = le.fit_transform(building.primary_use)
    weather_train = pd.read_csv(path_weather_train)
    weather_test = pd.read_csv(path_weather_test)
    weather_train.drop(["sea_level_pressure", "wind_direction", "wind_speed"], axis=1, inplace=True)
    weather_test.drop(["sea_level_pressure", "wind_direction", "wind_speed"], axis=1, inplace=True)
    weather_train = weather_train.groupby("site_id").apply(lambda group: group.interpolate(limit_direction="both"))
    weather_test = weather_test.groupby("site_id").apply(lambda group: group.interpolate(limit_direction="both"))
    print(f"df_train={df_train.shape} df_test={df_test.shape} weather_train={weather_train.shape} weather_test={weather_test.shape}")

    df_train = df_train.merge(building, on="building_id")
    df_train = df_train.merge(weather_train, on=["site_id", "timestamp"], how="left")
    df_train = df_train[~((df_train.site_id==0) & (df_train.meter==0) & (df_train.building_id <= 104) & (df_train.timestamp < "2016-05-21"))]

    df_train.reset_index(drop=True, inplace=True)
    df_train.timestamp = pd.to_datetime(df_train.timestamp, format='%Y-%m-%d %H:%M:%S')
    df_train["log_meter_reading"] = np.log1p(df_train.meter_reading)

    df_test = df_test.merge(building, on="building_id")
    df_test = df_test.merge(weather_test, on=["site_id", "timestamp"], how="left")
    df_test.reset_index(drop=True, inplace=True)
    df_test.timestamp = pd.to_datetime(df_test.timestamp, format='%Y-%m-%d %H:%M:%S')

    del building, le
    gc.collect()

    df_train = reduce_mem_usage(df_train, use_float16=True)
    df_test = reduce_mem_usage(df_test, use_float16=True)

    weather_train.timestamp = pd.to_datetime(weather_train.timestamp, format='%Y-%m-%d %H:%M:%S')
    weather_test.timestamp = pd.to_datetime(weather_test.timestamp, format='%Y-%m-%d %H:%M:%S')
    weather_train = reduce_mem_usage(weather_train, use_float16=True)
    weather_test = reduce_mem_usage(weather_test, use_float16=True)
    #Feature Engineering: Time
    df_train["hour"] = df_train.timestamp.dt.hour
    df_train["weekday"] = df_train.timestamp.dt.weekday
    df_test["hour"] = df_test.timestamp.dt.hour
    df_test["weekday"] = df_test.timestamp.dt.weekday

    print("Aggregation...")
    df_building_meter = df_train.groupby(["building_id", "meter"]).\
        agg(mean_building_meter=("log_meter_reading", "mean"), median_building_meter=("log_meter_reading", "median")).reset_index()
    df_train = df_train.merge(df_building_meter, on=["building_id", "meter"])
    df_test = df_test.merge(df_building_meter, on=["building_id", "meter"])

    df_building_meter_hour = df_train.groupby(["building_id", "meter", "hour"]).\
        agg(mean_building_meter=("log_meter_reading", "mean"),median_building_meter=("log_meter_reading", "median")).reset_index()

    df_train = df_train.merge(df_building_meter_hour, on=["building_id", "meter", "hour"])
    df_test = df_test.merge(df_building_meter_hour, on=["building_id", "meter", "hour"])
    print(f"Aggregation... df_train={df_train.shape} df_test={df_test.shape}")

    weather_train = create_lag_features(weather_train, 18)
    weather_train.drop(["air_temperature", "cloud_coverage", "dew_temperature", "precip_depth_1_hr"], axis=1, inplace=True)

    df_train = df_train.merge(weather_train, on=["site_id", "timestamp"], how="left")
    del weather_train
    gc.collect()

    with open(pkl_path, "wb") as fp:
        pickle.dump([df_train, df_test,weather_test,leak], fp)
    sys.exit(-1)

categorical_features = ["building_id","primary_use","meter", "weekday","hour"]
all_features = [col for col in df_train.columns if col not in ["timestamp", "site_id", "meter_reading", "log_meter_reading"]]

cv = 2
models = {}
cv_scores = {"site_id": [], "cv_score": []}

early_stop = 21
verbose_eval = 101
metric = 'l2'
num_rounds=999; lr=0.05; bf=0.3;
#num_rounds = 10;      lr = 0.05;          bf = 0.3

params = {"objective": "regression", "metric": "rmse",
          "num_leaves": 41, "learning_rate": 0.049,'n_estimators': num_rounds,
          "bagging_freq": 5, "bagging_fraction": 0.51, "feature_fraction": 0.81,
            'verbose_eval': verbose_eval,"early_stopping_rounds": early_stop,'n_jobs': 8, "elitism": 0
          }
for site_id in tqdm(range(16), desc="site_id"):
    print(cv, "fold CV for site_id:", site_id)
    kf = KFold(n_splits=cv, random_state=seed)
    models[site_id] = []

    X_train_site = df_train[df_train.site_id == site_id].reset_index(drop=True)
    y_train_site = X_train_site.log_meter_reading
    y_pred_train_site = np.zeros(X_train_site.shape[0])

    score = 0

    for fold, (train_index, valid_index) in enumerate(kf.split(X_train_site, y_train_site)):
        X_train, X_valid = X_train_site.loc[train_index, all_features], X_train_site.loc[valid_index, all_features]
        y_train, y_valid = y_train_site.iloc[train_index], y_train_site.iloc[valid_index]

        if isMORT:
            params['verbose'] = 667 if site_id==0 and fold == 0 else 0
            merge_datas = []
            model = LiteMORT(params,merge_infos=None)
            model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], categorical_feature=categorical_features)
        else:
            dtrain = lgb.Dataset(X_train, label=y_train, categorical_feature=categorical_features)
            dvalid = lgb.Dataset(X_valid, label=y_valid, categorical_feature=categorical_features)
            watchlist = [dtrain, dvalid]
            model = lgb.train(params, train_set=dtrain, num_boost_round=num_rounds, valid_sets=watchlist, verbose_eval=verbose_eval,
                                  early_stopping_rounds=early_stop)
        models[site_id].append(model)
        y_pred_valid = model.predict(X_valid, num_iteration=model.best_iteration)
        y_pred_train_site[valid_index] = y_pred_valid

        rmse = np.sqrt(mean_squared_error(y_valid, y_pred_valid))
        print("Site Id:", site_id, ", Fold:", fold + 1, ", RMSE:", rmse)
        score += rmse / cv

        gc.collect()

    cv_scores["site_id"].append(site_id)
    cv_scores["cv_score"].append(score)

    print("\nSite Id:", site_id, ", CV RMSE:", np.sqrt(mean_squared_error(y_train_site, y_pred_train_site)), "\n")

weather_test = create_lag_features(weather_test, 18)
weather_test.drop(["air_temperature", "cloud_coverage", "dew_temperature", "precip_depth_1_hr"], axis=1, inplace=True)

df_test_sites = []

for site_id in tqdm(range(16), desc="site_id"):
    print("Preparing test data for site_id", site_id)

    X_test_site = df_test[df_test.site_id == site_id]
    weather_test_site = weather_test[weather_test.site_id == site_id]

    X_test_site = X_test_site.merge(weather_test_site, on=["site_id", "timestamp"], how="left")

    row_ids_site = X_test_site.row_id

    X_test_site = X_test_site[all_features]
    y_pred_test_site = np.zeros(X_test_site.shape[0])

    print("Scoring for site_id", site_id)
    for fold in range(cv):
        model_lgb = models[site_id][fold]
        y_pred_test_site += model_lgb.predict(X_test_site, num_iteration=model_lgb.best_iteration) / cv
        gc.collect()

    df_test_site = pd.DataFrame({"row_id": row_ids_site, "meter_reading": y_pred_test_site})
    df_test_sites.append(df_test_site)

    print("Scoring for site_id", site_id, "completed\n")
    gc.collect()

submit = pd.concat(df_test_sites)
submit.meter_reading = np.clip(np.expm1(submit.meter_reading), 0, a_max=None)
#submit.to_csv("submission_noleak.csv", index=False)

del df_test_sites;        gc.collect()

submit = submit.merge(leak[["row_id", "meter_reading_scraped"]], on=["row_id"], how="left")
submit.loc[submit.meter_reading_scraped.notnull(), "meter_reading"] = submit.loc[submit.meter_reading_scraped.notnull(), "meter_reading_scraped"]
submit.drop(["meter_reading_scraped"], axis=1, inplace=True)

submit.to_csv("submission.csv.gz", index=False, float_format='%.4f',compression='gzip')
print(submit.head(100),submit.tail(100))