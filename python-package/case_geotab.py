#https://www.kaggle.com/dcaichara/feature-engineering-and-lightgbm

import pandas as pd
pd.set_option('display.max_columns', 999)
import numpy as np
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import math
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')
import os
import sys
import pickle
from litemort import *
from LiteMORT_hyppo import *
import time
import random
import gc
isMORT = len(sys.argv)>1 and sys.argv[1] == "mort"
isMORT = True
gbm='MORT' if isMORT else 'LGB'
some_rows = 5000
#some_rows = None
#data_root = '../input/'
data_root = "F:/Datasets/geotab"
pkl_path = f'{data_root}/_geotab___.pickle'

if os.path.isfile(pkl_path):
    print("====== Load pickle @{} ......".format(pkl_path))
    with open(pkl_path, "rb") as fp:
        [train, test,final_features, all_target] = pickle.load(fp)
    final_features =  ['CenterDistance_Intersection_mean', 'EntryType_2', 'EntryType_1_FE',
                                           'ExitHeading_Intersection_std' ]+final_features
else:
    print('Loading trian set...')
    train = pd.read_csv(f'{data_root}/train.csv')
    print('Loading test set...')
    test = pd.read_csv(f'{data_root}/test.csv')
    print('We have {} rows and {} columns in our train set'.format(train.shape[0], train.shape[1]))
    print('We have {} rows and {} columns in our test set'.format(test.shape[0], test.shape[1]))
    train = train[['TotalTimeStopped_p80', 'IntersectionId', 'Latitude', 'Longitude', 'EntryStreetName',
                            'ExitStreetName', 'EntryHeading', 'ExitHeading', 'Hour', 'Weekend',
                            'Month', 'City']]
    # let's select the target variable we are going to use for feature selection
    target = train['TotalTimeStopped_p80']
    train.drop('TotalTimeStopped_p80', axis = 1, inplace = True)


    def prepro(train, test):
        # Road Mapping
        road_encoding = {'Street': 'Street', 'St': 'Street', 'Avenue': 'Avenue', 'Ave': 'Avenue',
                         'Boulevard': 'Boulevard', 'Road': 'Road', 'Drive': 'Drive', 'Lane': 'Lane',
                         'Tunnel': 'Tunnel', 'Highway': 'Highway', 'Way': 'Way', 'Parkway': 'Parkway',
                         'Parking': 'Parking', 'Oval': 'Oval', 'Square': 'Square', 'Place': 'Place',
                         'Bridge': 'Bridge'}

        def encode(x):
            if pd.isna(x):
                return 'Street'
            for road in road_encoding.keys():
                if road in x:
                    return road_encoding[road]

        for par in [train, test]:
            par['EntryType'] = par['EntryStreetName'].apply(encode)
            par['ExitType'] = par['ExitStreetName'].apply(encode)
            par['EntryType_1'] = par['EntryStreetName'].str.split().str.get(0)
            par['ExitType_1'] = par['ExitStreetName'].str.split().str.get(0)
            par['EntryType_2'] = par['EntryStreetName'].str.split().str.get(1)
            par['ExitType_2'] = par['ExitStreetName'].str.split().str.get(1)
            par.loc[par['EntryType_1'].isin(
                par['EntryType_1'].value_counts()[par['EntryType_1'].value_counts() <= 500].index), 'EntryType_1'] = 'Other'
            par.loc[par['ExitType_1'].isin(
                par['ExitType_1'].value_counts()[par['ExitType_1'].value_counts() <= 500].index), 'ExitType_1'] = 'Other'
            par.loc[par['EntryType_2'].isin(
                par['EntryType_2'].value_counts()[par['EntryType_2'].value_counts() <= 500].index), 'EntryType_2'] = 'Other'
            par.loc[par['ExitType_2'].isin(
                par['ExitType_2'].value_counts()[par['ExitType_2'].value_counts() <= 500].index), 'ExitType_2'] = 'Other'
            par['EntryType_1'].fillna('Other', inplace=True)
            par['ExitType_1'].fillna('Other', inplace=True)
            par['EntryType_2'].fillna('Other', inplace=True)
            par['ExitType_2'].fillna('Other', inplace=True)

        # The cardinal directions can be expressed using the equation: θ/π
        # Where  θ  is the angle between the direction we want to encode and the north compass direction, measured clockwise.
        directions = {'N': 0, 'NE': 1 / 4, 'E': 1 / 2, 'SE': 3 / 4, 'S': 1, 'SW': 5 / 4, 'W': 3 / 2, 'NW': 7 / 4}
        for par in [train, test]:
            par['EntryHeading'] = par['EntryHeading'].map(directions)
            par['ExitHeading'] = par['ExitHeading'].map(directions)

        # EntryStreetName == ExitStreetName ?
        # EntryHeading == ExitHeading ?
        for par in [train, test]:
            par["same_street_exact"] = (par["EntryStreetName"] == par["ExitStreetName"]).astype(int)
            par["same_heading_exact"] = (par["EntryHeading"] == par["ExitHeading"]).astype(int)

        # We have some intersection id that are in more than one city, it is a good idea to feature cross them
        for par in [train, test]:
            par['Intersection'] = par['IntersectionId'].astype(str) + '_' + par['City'].astype(str)

        for par in [train, test]:
            # Concatenating the city and month into one variable
            par['city_month'] = par["City"].astype(str) + par["Month"].astype(str)

        # Add climate data
        monthly_av = {'Atlanta1': 43, 'Atlanta5': 69, 'Atlanta6': 76, 'Atlanta7': 79, 'Atlanta8': 78,
                      'Atlanta9': 73, 'Atlanta10': 62, 'Atlanta11': 53, 'Atlanta12': 45, 'Boston1': 30,
                      'Boston5': 59, 'Boston6': 68, 'Boston7': 74, 'Boston8': 73, 'Boston9': 66,
                      'Boston10': 55, 'Boston11': 45, 'Boston12': 35, 'Chicago1': 27, 'Chicago5': 60,
                      'Chicago6': 70, 'Chicago7': 76, 'Chicago8': 76, 'Chicago9': 68,
                      'Chicago10': 56, 'Chicago11': 45, 'Chicago12': 32, 'Philadelphia1': 35,
                      'Philadelphia5': 66, 'Philadelphia6': 76, 'Philadelphia7': 81,
                      'Philadelphia8': 79, 'Philadelphia9': 72, 'Philadelphia10': 60,
                      'Philadelphia11': 49, 'Philadelphia12': 40}

        monthly_rainfall = {'Atlanta1': 5.02, 'Atlanta5': 3.95, 'Atlanta6': 3.63, 'Atlanta7': 5.12,
                            'Atlanta8': 3.67, 'Atlanta9': 4.09, 'Atlanta10': 3.11, 'Atlanta11': 4.10,
                            'Atlanta12': 3.82, 'Boston1': 3.92, 'Boston5': 3.24, 'Boston6': 3.22,
                            'Boston7': 3.06, 'Boston8': 3.37, 'Boston9': 3.47, 'Boston10': 3.79,
                            'Boston11': 3.98, 'Boston12': 3.73, 'Chicago1': 1.75, 'Chicago5': 3.38,
                            'Chicago6': 3.63, 'Chicago7': 3.51, 'Chicago8': 4.62, 'Chicago9': 3.27,
                            'Chicago10': 2.71, 'Chicago11': 3.01, 'Chicago12': 2.43,
                            'Philadelphia1': 3.52, 'Philadelphia5': 3.88, 'Philadelphia6': 3.29,
                            'Philadelphia7': 4.39, 'Philadelphia8': 3.82, 'Philadelphia9': 3.88,
                            'Philadelphia10': 2.75, 'Philadelphia11': 3.16, 'Philadelphia12': 3.31}

        monthly_snowfall = {'Atlanta1': 0.6, 'Atlanta5': 0, 'Atlanta6': 0, 'Atlanta7': 0,
                            'Atlanta8': 0, 'Atlanta9': 0, 'Atlanta10': 0, 'Atlanta11': 0,
                            'Atlanta12': 0.2, 'Boston1': 12.9, 'Boston5': 0, 'Boston6': 0,
                            'Boston7': 0, 'Boston8': 0, 'Boston9': 0, 'Boston10': 0, 'Boston11': 1.3,
                            'Boston12': 9.0, 'Chicago1': 11.5, 'Chicago5': 0, 'Chicago6': 0,
                            'Chicago7': 0, 'Chicago8': 0, 'Chicago9': 0, 'Chicago10': 0,
                            'Chicago11': 1.3, 'Chicago12': 8.7, 'Philadelphia1': 6.5,
                            'Philadelphia5': 0, 'Philadelphia6': 0, 'Philadelphia7': 0,
                            'Philadelphia8': 0, 'Philadelphia9': 0, 'Philadelphia10': 0,
                            'Philadelphia11': 0.3, 'Philadelphia12': 3.4}

        monthly_daylight = {'Atlanta1': 10, 'Atlanta5': 14, 'Atlanta6': 14, 'Atlanta7': 14,
                            'Atlanta8': 13, 'Atlanta9': 12, 'Atlanta10': 11, 'Atlanta11': 10,
                            'Atlanta12': 10, 'Boston1': 9, 'Boston5': 15, 'Boston6': 15,
                            'Boston7': 15, 'Boston8': 14, 'Boston9': 12, 'Boston10': 11,
                            'Boston11': 10, 'Boston12': 9, 'Chicago1': 10, 'Chicago5': 15,
                            'Chicago6': 15, 'Chicago7': 15, 'Chicago8': 14, 'Chicago9': 12,
                            'Chicago10': 11, 'Chicago11': 10, 'Chicago12': 9, 'Philadelphia1': 10,
                            'Philadelphia5': 14, 'Philadelphia6': 15, 'Philadelphia7': 15,
                            'Philadelphia8': 14, 'Philadelphia9': 12, 'Philadelphia10': 11,
                            'Philadelphia11': 10, 'Philadelphia12': 9}

        monthly_sunshine = {'Atlanta1': 5.3, 'Atlanta5': 9.3, 'Atlanta6': 9.5, 'Atlanta7': 8.8, 'Atlanta8': 8.3,
                            'Atlanta9': 7.6,
                            'Atlanta10': 7.7, 'Atlanta11': 6.2, 'Atlanta12': 5.3, 'Boston1': 5.3, 'Boston5': 8.6,
                            'Boston6': 9.6,
                            'Boston7': 9.7, 'Boston8': 8.9, 'Boston9': 7.9, 'Boston10': 6.7, 'Boston11': 4.8,
                            'Boston12': 4.6,
                            'Chicago1': 4.4, 'Chicago5': 9.1, 'Chicago6': 10.4, 'Chicago7': 10.3, 'Chicago8': 9.1,
                            'Chicago9': 7.6,
                            'Chicago10': 6.2, 'Chicago11': 3.6, 'Chicago12': 3.4, 'Philadelphia1': 5.0,
                            'Philadelphia5': 7.9,
                            'Philadelphia6': 9.0, 'Philadelphia7': 8.9, 'Philadelphia8': 8.4, 'Philadelphia9': 7.9,
                            'Philadelphia10': 6.6, 'Philadelphia11': 5.2, 'Philadelphia12': 4.4}

        for par in [train, test]:
            # Creating a new column by mapping the city_month variable to it's corresponding average monthly temperature
            par["average_temp"] = par['city_month'].map(monthly_av)
            # Creating a new column by mapping the city_month variable to it's corresponding average monthly rainfall
            par["average_rainfall"] = par['city_month'].map(monthly_rainfall)
            # Creating a new column by mapping the city_month variable to it's corresponding average monthly snowfall
            par['average_snowfall'] = par['city_month'].map(monthly_snowfall)
            # Creating a new column by mapping the city_month variable to it's corresponding average monthly daylight
            par["average_daylight"] = par['city_month'].map(monthly_daylight)
            # Creating a new column by mapping the city_month variable to it's corresponding average monthly sunshine
            par["average_sunshine"] = par['city_month'].map(monthly_sunshine)

        for par in [train, test]:
            # drop city month
            par.drop('city_month', axis=1, inplace=True)
            # Add feature is day
            par['is_day'] = par['Hour'].apply(lambda x: 1 if 5 < x < 20 else 0)

            # distance from the center of the city

        def add_distance(df):
            df_center = pd.DataFrame({"Atlanta": [33.753746, -84.386330],
                                      "Boston": [42.361145, -71.057083],
                                      "Chicago": [41.881832, -87.623177],
                                      "Philadelphia": [39.952583, -75.165222]})
            df["CenterDistance"] = df.apply(lambda row: math.sqrt((df_center[row.City][0] - row.Latitude) ** 2 +
                                                                  (df_center[row.City][1] - row.Longitude) ** 2), axis=1)

        add_distance(train)
        add_distance(test)

        # frequency encode
        def encode_FE(df1, df2, cols):
            for col in cols:
                df = pd.concat([df1[col], df2[col]])
                vc = df.value_counts(dropna=True, normalize=True).to_dict()
                nm = col + '_FE'
                df1[nm] = df1[col].map(vc)
                df1[nm] = df1[nm].astype('float32')
                df2[nm] = df2[col].map(vc)
                df2[nm] = df2[nm].astype('float32')
                print(nm, ', ', end='')

        # combine features
        def encode_CB(col1, col2, df1=train, df2=test):
            nm = col1 + '_' + col2
            df1[nm] = df1[col1].astype(str) + '_' + df1[col2].astype(str)
            df2[nm] = df2[col1].astype(str) + '_' + df2[col2].astype(str)
            print(nm, ', ', end='')

        # group aggregations nunique
        def encode_AG2(main_columns, agg_col, train_df=train, test_df=test):
            for main_column in main_columns:
                for col in agg_col:
                    comb = pd.concat([train_df[[col] + [main_column]], test_df[[col] + [main_column]]], axis=0)
                    mp = comb.groupby(col)[main_column].agg(['nunique'])['nunique'].to_dict()
                    train_df[col + '_' + main_column + '_ct'] = train_df[col].map(mp).astype('float32')
                    test_df[col + '_' + main_column + '_ct'] = test_df[col].map(mp).astype('float32')
                    print(col + '_' + main_column + '_ct, ', end='')

        def encode_AG(main_columns, agg_col, aggregations=['mean'], train_df=train, test_df=test, fillna=True, usena=False):
            # aggregation of main agg_cols
            for main_column in main_columns:
                for col in agg_col:
                    for agg_type in aggregations:
                        new_col_name = main_column + '_' + col + '_' + agg_type
                        temp_df = pd.concat([train_df[[col, main_column]], test_df[[col, main_column]]])
                        if usena: temp_df.loc[temp_df[main_column] == -1, main_column] = np.nan
                        temp_df = temp_df.groupby([col])[main_column].agg([agg_type]).reset_index().rename(
                            columns={agg_type: new_col_name})

                        temp_df.index = list(temp_df[col])
                        temp_df = temp_df[new_col_name].to_dict()

                        train_df[new_col_name] = train_df[col].map(temp_df).astype('float32')
                        test_df[new_col_name] = test_df[col].map(temp_df).astype('float32')

                        if fillna:
                            train_df[new_col_name].fillna(-1, inplace=True)
                            test_df[new_col_name].fillna(-1, inplace=True)

                        print("'" + new_col_name + "'", ', ', end='')

        # Frequency encode
        encode_FE(train, test,
                  ['Hour', 'Month', 'EntryType', 'ExitType', 'EntryType_1', 'EntryType_2', 'ExitType_1', 'ExitType_2',
                   'Intersection', 'City'])

        # Agreggations of main columns
        encode_AG(['Longitude', 'Latitude', 'CenterDistance', 'EntryHeading', 'ExitHeading'],
                  ['Hour', 'Weekend', 'Month', 'Intersection'], ['mean', 'std'])

        # bucketize lat and lon
        temp_df = pd.concat([train[['Latitude', 'Longitude']], test[['Latitude', 'Longitude']]]).reset_index(drop=True)
        temp_df['Latitude_B'] = pd.cut(temp_df['Latitude'], 30)
        temp_df['Longitude_B'] = pd.cut(temp_df['Longitude'], 30)

        # feature cross lat and lon
        temp_df['Latitude_B_Longitude_B'] = temp_df['Latitude_B'].astype(str) + '_' + temp_df['Longitude_B'].astype(str)
        train['Latitude_B'] = temp_df.loc[:(train.shape[0]), 'Latitude_B']
        test['Latitude_B'] = temp_df.loc[(train.shape[0]):, 'Latitude_B']
        train['Longitude_B'] = temp_df.loc[:(train.shape[0]), 'Longitude_B']
        test['Longitude_B'] = temp_df.loc[(train.shape[0]):, 'Longitude_B']
        train['Latitude_B_Longitude_B'] = temp_df.loc[:(train.shape[0]), 'Latitude_B_Longitude_B']
        test['Latitude_B_Longitude_B'] = temp_df.loc[(train.shape[0]):, 'Latitude_B_Longitude_B']

        # feature crosses
        encode_CB('Hour', 'Month')

        # group aggregations nunique
        encode_AG2(['Intersection', 'Latitude_B_Longitude_B'], ['Hour', 'Month'])

        # label encode
        for i, f in enumerate(train.columns):
            if (np.str(train[f].dtype) == 'category') | (train[f].dtype == 'object'):
                df_comb = pd.concat([train[f], test[f]], axis=0)
                df_comb, _ = df_comb.factorize(sort=True)
                if df_comb.max() > 32000: print(f, 'needs int32')
                train[f] = df_comb[:len(train)].astype('int16')
                test[f] = df_comb[len(train):].astype('int16')

        print('After preprocessing we have {} columns'.format(train.shape[1]))
    prepro(train, test)

    usefull_columns = ['same_heading_exact', 'Intersection', 'is_day', 'CenterDistance', 'Intersection_FE', 'Longitude_Weekend_mean', 'CenterDistance_Month_std'] # this columns were picked with forward feature selection (run previous cell)
    final_features = usefull_columns + ['IntersectionId', 'Latitude', 'Longitude', 'EntryStreetName','ExitStreetName',
                                        'EntryHeading', 'ExitHeading', 'Hour', 'Weekend', 'Month', 'City', 'EntryType', 'ExitType']
    print('Our usefull features found with forward feature selection are {}'.format(final_features))

    print('-'*50)
    print('We have selected {} features'.format(len(final_features)))

    train_target = pd.read_csv(f'{data_root}/train.csv')
    target1 = train_target['TotalTimeStopped_p20']
    target2 = train_target['TotalTimeStopped_p50']
    target3 = train_target['TotalTimeStopped_p80']
    target4 = train_target['DistanceToFirstStop_p20']
    target5 = train_target['DistanceToFirstStop_p50']
    target6 = train_target['DistanceToFirstStop_p80']
    all_target = [target1, target2, target3, target4, target5, target6]
    with open(pkl_path, "wb") as fp:  # Pickling
        pickle.dump([train, test,final_features, all_target], fp)
        print("====== Dump pickle @{} ......OK".format(pkl_path))
    input("......")

if some_rows is not None:
    nMost=train.shape[0]
    random.seed(42)
    subset = random.sample(range(nMost), some_rows)
    train = train.iloc[subset, :].reset_index(drop=True)
    all_=[]
    for target in all_target:
        target = target.iloc[subset].reset_index(drop=True)
        all_.append(target)
    all_target=all_
    gc.collect()
    print('====== Some Samples ... data={}'.format(train.shape))

if False:
    print(f"train={train.shape} test={test.shape}\n final_features={final_features}")
    feat_fix = ['IntersectionId', 'Latitude', 'Longitude', 'EntryStreetName','ExitStreetName', 'EntryHeading',
         'ExitHeading', 'Hour', 'Weekend', 'Month', 'City', 'EntryType', 'ExitType']
    feat_select = train.columns
    feat_select = list(set(feat_select)-set(feat_fix))
    MORT_feat_select_(train,all_target[2],feat_fix,feat_select,n_init=5, n_iter=12)
    input("......MORT_feat_search......")

param = {'application': 'regression','n_estimators':100000,'early_stopping_rounds':100,
         'learning_rate': 0.05,
         'metric': 'rmse',
         'seed': 42,
         'bagging_fraction': 0.7,
         'feature_fraction': 0.9,
         'lambda_l1': 0.0,
         'lambda_l2': 5.0,
         'max_depth': 30,
         'min_child_weight': 50.0,
         'min_split_gain': 0.1,
         'num_leaves': 230}
param_mort = {'objective': 'regression','num_leaves': 512,   'n_estimators':100000,'early_stopping_rounds':100,
     'feature_fraction': 0.9,     'bagging_fraction': 1,
    "adaptive":'weight1',   #无效，晕
    'max_bin': 512,
    #'cascade':'lasso', 64.409->64.392 作用不大
    #"learning_schedule":"adaptive",
     'max_depth': 30,
     'min_split_gain': 0.1,
     'min_child_weight': 20.0,
    #'min_data_in_leaf': 10,
     'learning_rate': 0.05,
     'boosting_type': 'gbdt',     'verbose': 666,     'metric': {'rmse'}}

def run_lgb_f(train, test,all_target):
    # get prediction dictonary were we are going to store predictions
    all_preds = {0 : [], 1 : [], 2 : [], 3 : [], 4 : [], 5 : []}
    # get a list with all the target variables
    #all_target = [target1, target2, target3, target4, target5, target6]
    nfold = 5
    kf = KFold(n_splits=nfold, random_state=228, shuffle=True)
    for i in range(len(all_preds)):
        #if i<len(all_preds)-1:            continue
        print('Training and predicting for target {}'.format(i+1))
        t0 = time.time()
        oof = np.zeros(len(train))
        all_preds[i] = np.zeros(len(test))
        n = 1
        for train_index, valid_index in kf.split(all_target[i]):
            print("fold {}".format(n))
            y_train, y_valid = all_target[i][train_index], all_target[i][valid_index]
            if isMORT:
                mort = LiteMORT(param_mort).fit(train.iloc[train_index], y_train,
                                           eval_set=[(train.iloc[valid_index], y_valid)] )
                oof[valid_index] = mort.predict(train.iloc[valid_index])
                all_preds[i] += mort.predict(test) / nfold
            else:
                xg_train = lgb.Dataset(train.iloc[train_index],
                                       label=all_target[i][train_index]
                                       )
                xg_valid = lgb.Dataset(train.iloc[valid_index],
                                       label=all_target[i][valid_index]
                                       )

                clf = lgb.train(param, xg_train, 100000, valid_sets=[xg_train, xg_valid],
                                verbose_eval=500, early_stopping_rounds=100)
                oof[valid_index] = clf.predict(train.iloc[valid_index], num_iteration=clf.best_iteration)

                all_preds[i] += clf.predict(test, num_iteration=clf.best_iteration) / nfold
            score = np.sqrt(mean_squared_error(oof[valid_index], y_valid))
            print(f"------{n}:\tRMSE: {score:0.4f} time={time.time() - t0:.4g}")
            n = n + 1
        fold_score = np.sqrt(mean_squared_error(all_target[i], oof))
        print("\n\nTARGET_{} CV RMSE: {:0.4f} time={:.4g}".format(i, fold_score,time.time() - t0))
        #print("\n\nCV RMSE: {:<0.4f}".format(np.sqrt(mean_squared_error(all_target[i], oof))))
    return all_preds,fold_score

if False:       #hyparam_search
    def MortOnParam(num_leaves, feature_fraction, bagging_fraction, max_depth, learning_rate, min_data_in_leaf,max_bin):
        param_mort['verbose']=0
        param_mort['early_stopping_rounds']=1   #Oscillate
        param_mort["num_leaves"] = int(round(num_leaves))
        param_mort['feature_fraction'] = max(min(feature_fraction, 1), 0)
        param_mort['bagging_fraction'] = max(min(bagging_fraction, 1), 0)
        param_mort['max_depth'] = int(round(max_depth))
        param_mort['learning_rate'] = learning_rate
        param_mort['min_data_in_leaf'] = int(round(min_data_in_leaf))
        param_mort['max_bin'] = int(round(max_bin))

        _,fold_score = run_lgb_f(train[final_features], test[final_features],all_target)
        return -fold_score

    pds = {'num_leaves': (230, 230),
           'feature_fraction': (1, 1),
           'bagging_fraction': (1, 1),
           'max_depth': (10,50),
           'learning_rate': (0.05, 0.05),
           'min_data_in_leaf': (20, 20),
           'max_bin': (512, 512),
           }
    hyparam_search(MortOnParam,pds,n_init=5, n_iter=12)
    input("......")

nfold=5
all_preds,fold_score = run_lgb_f(train[final_features], test[final_features],all_target)
#input("......")

submission = pd.read_csv(f'{data_root}/sample_submission.csv')
data2 = pd.DataFrame(all_preds).stack()
data2 = pd.DataFrame(data2)
submission['Target'] = data2[0].values
submission.to_csv('lgbm_baseline_fs_bopt.csv', index=False)
path = f'{data_root}/[{gbm}]_{some_rows}_{fold_score:.5f}_F{nfold}_.csv.gz'
submission.to_csv(path, index=False,float_format='%.4f',compression='gzip' )
input("......")