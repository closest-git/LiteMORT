#https://www.kaggle.com/dcaichara/feature-engineering-and-lightgbm

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
import math
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split,KFold
from sklearn.preprocessing import StandardScaler
#from bayes_opt import BayesianOptimization
import lightgbm as lgb
from litemort import *
import os, sys
import pickle
import time

isMORT = len(sys.argv)>1 and sys.argv[1] == "mort"
isMORT = True
model='MORT' if isMORT else 'LGB'
#some_rows = 5000
some_rows = None
#data_root = '../input/'
data_root = "F:/Datasets/geotab"
pkl_path = f'{data_root}/_kyakovlev_{some_rows}.pickle'


if os.path.isfile(pkl_path):
    print("====== Load pickle @{} ......".format(pkl_path))
    with open(pkl_path, "rb") as fp:
        [train,test, cat_feat, all_target] = pickle.load(fp)
else:
    train = pd.read_csv(f'{data_root}/train.csv')
    test = pd.read_csv(f'{data_root}/test.csv')

    print(train["City"].unique())
    print(test["City"].unique())
    road_encoding = {
    "Street":0, "St":0, "Avenue":1, "Ave":1, "Boulevard":2, "Road":3, "Drive":4, "Lane":5, "Tunnel":6,
     "Highway":7, "Way":8, "Parkway":9, "Parking":10, "Oval":11, "Square":12, "Place":13, "Bridge":14}

    def encode(x):
        if pd.isna(x):
            return 0
        for road in road_encoding.keys():
            if road in x:
                return road_encoding[road]
        return 0

    train['EntryType'] = train['EntryStreetName'].apply(encode)
    train['ExitType'] = train['ExitStreetName'].apply(encode)
    test['EntryType'] = test['EntryStreetName'].apply(encode)
    test['ExitType'] = test['ExitStreetName'].apply(encode)

    directions = {
        'N': 0,    'NE': 1/4,    'E': 1/2,    'SE': 3/4,    'S': 1,    'SW': 5/4,    'W': 3/2,    'NW': 7/4
    }

    train['EntryHeading'] = train['EntryHeading'].map(directions)
    train['ExitHeading'] = train['ExitHeading'].map(directions)
    test['EntryHeading'] = test['EntryHeading'].map(directions)
    test['ExitHeading'] = test['ExitHeading'].map(directions)
    train['diffHeading'] = train['EntryHeading']-train['ExitHeading']
    test['diffHeading'] = test['EntryHeading']-test['ExitHeading']
    train["same_street_exact"] = (train["EntryStreetName"] ==  train["ExitStreetName"]).astype(int)
    test["same_street_exact"] = (test["EntryStreetName"] ==  test["ExitStreetName"]).astype(int)
    train["Intersection"] = train["IntersectionId"].astype(str) + train["City"]
    test["Intersection"] = test["IntersectionId"].astype(str) + test["City"]
    encoder = LabelEncoder()
    encoder.fit(pd.concat([train["Intersection"],test["Intersection"]]).drop_duplicates().values)
    train["Intersection"] = encoder.transform(train["Intersection"])
    test["Intersection"] = encoder.transform(test["Intersection"])
    monthly_av = {'Atlanta1': 43, 'Atlanta5': 69, 'Atlanta6': 76, 'Atlanta7': 79, 'Atlanta8': 78, 'Atlanta9': 73,
                  'Atlanta10': 62, 'Atlanta11': 53, 'Atlanta12': 45, 'Boston1': 30, 'Boston5': 59, 'Boston6': 68,
                  'Boston7': 74, 'Boston8': 73, 'Boston9': 66, 'Boston10': 55,'Boston11': 45, 'Boston12': 35,
                  'Chicago1': 27, 'Chicago5': 60, 'Chicago6': 70, 'Chicago7': 76, 'Chicago8': 76, 'Chicago9': 68,
                  'Chicago10': 56,  'Chicago11': 45, 'Chicago12': 32, 'Philadelphia1': 35, 'Philadelphia5': 66,
                  'Philadelphia6': 76, 'Philadelphia7': 81, 'Philadelphia8': 79, 'Philadelphia9': 72, 'Philadelphia10': 60,
                  'Philadelphia11': 49, 'Philadelphia12': 40}
    # Concatenating the city and month into one variable
    train['city_month'] = train["City"] + train["Month"].astype(str)
    test['city_month'] = test["City"] + test["Month"].astype(str)
    monthly_rainfall = {'Atlanta1': 5.02, 'Atlanta5': 3.95, 'Atlanta6': 3.63, 'Atlanta7': 5.12, 'Atlanta8': 3.67, 'Atlanta9': 4.09,
                  'Atlanta10': 3.11, 'Atlanta11': 4.10, 'Atlanta12': 3.82, 'Boston1': 3.92, 'Boston5': 3.24, 'Boston6': 3.22,
                  'Boston7': 3.06, 'Boston8': 3.37, 'Boston9': 3.47, 'Boston10': 3.79,'Boston11': 3.98, 'Boston12': 3.73,
                  'Chicago1': 1.75, 'Chicago5': 3.38, 'Chicago6': 3.63, 'Chicago7': 3.51, 'Chicago8': 4.62, 'Chicago9': 3.27,
                  'Chicago10': 2.71,  'Chicago11': 3.01, 'Chicago12': 2.43, 'Philadelphia1': 3.52, 'Philadelphia5': 3.88,
                  'Philadelphia6': 3.29, 'Philadelphia7': 4.39, 'Philadelphia8': 3.82, 'Philadelphia9':3.88 , 'Philadelphia10': 2.75,
                  'Philadelphia11': 3.16, 'Philadelphia12': 3.31}
    # Creating a new column by mapping the city_month variable to it's corresponding average monthly rainfall
    train["average_rainfall"] = train['city_month'].map(monthly_rainfall)
    test["average_rainfall"] = test['city_month'].map(monthly_rainfall)
    monthly_snowfall = {'Atlanta1': 0.6, 'Atlanta5': 0, 'Atlanta6': 0, 'Atlanta7': 0, 'Atlanta8': 0, 'Atlanta9': 0,
                  'Atlanta10': 0, 'Atlanta11': 0, 'Atlanta12': 0.2, 'Boston1': 12.9, 'Boston5': 0, 'Boston6': 0,
                  'Boston7': 0, 'Boston8': 0, 'Boston9': 0, 'Boston10': 0,'Boston11': 1.3, 'Boston12': 9.0,
                  'Chicago1': 11.5, 'Chicago5': 0, 'Chicago6': 0, 'Chicago7': 0, 'Chicago8': 0, 'Chicago9': 0,
                  'Chicago10': 0,  'Chicago11': 1.3, 'Chicago12': 8.7, 'Philadelphia1': 6.5, 'Philadelphia5': 0,
                  'Philadelphia6': 0, 'Philadelphia7': 0, 'Philadelphia8': 0, 'Philadelphia9':0 , 'Philadelphia10': 0,
                  'Philadelphia11': 0.3, 'Philadelphia12': 3.4}

    # Creating a new column by mapping the city_month variable to it's corresponding average monthly snowfall
    train["average_snowfall"] = train['city_month'].map(monthly_snowfall)
    test["average_snowfall"] = test['city_month'].map(monthly_snowfall)
    monthly_daylight = {'Atlanta1': 10, 'Atlanta5': 14, 'Atlanta6': 14, 'Atlanta7': 14, 'Atlanta8': 13, 'Atlanta9': 12,
                  'Atlanta10': 11, 'Atlanta11': 10, 'Atlanta12': 10, 'Boston1': 9, 'Boston5': 15, 'Boston6': 15,
                  'Boston7': 15, 'Boston8': 14, 'Boston9': 12, 'Boston10': 11,'Boston11': 10, 'Boston12': 9,
                  'Chicago1': 10, 'Chicago5': 15, 'Chicago6': 15, 'Chicago7': 15, 'Chicago8': 14, 'Chicago9': 12,
                  'Chicago10': 11,  'Chicago11': 10, 'Chicago12': 9, 'Philadelphia1': 10, 'Philadelphia5': 14,
                  'Philadelphia6': 15, 'Philadelphia7': 15, 'Philadelphia8': 14, 'Philadelphia9':12 , 'Philadelphia10': 11,
                  'Philadelphia11': 10, 'Philadelphia12': 9}

    # Creating a new column by mapping the city_month variable to it's corresponding average monthly daylight
    train["average_daylight"] = train['city_month'].map(monthly_daylight)
    test["average_daylight"] = test['city_month'].map(monthly_daylight)
    monthly_sunsine = {'Atlanta1': 5.3, 'Atlanta5': 9.3, 'Atlanta6': 9.5, 'Atlanta7': 8.8, 'Atlanta8': 8.3, 'Atlanta9': 7.6,
                  'Atlanta10': 7.7, 'Atlanta11': 6.2, 'Atlanta12': 5.3, 'Boston1': 5.3, 'Boston5': 8.6, 'Boston6': 9.6,
                  'Boston7': 9.7, 'Boston8': 8.9, 'Boston9': 7.9, 'Boston10': 6.7,'Boston11': 4.8, 'Boston12': 4.6,
                  'Chicago1': 4.4, 'Chicago5': 9.1, 'Chicago6': 10.4, 'Chicago7': 10.3, 'Chicago8': 9.1, 'Chicago9': 7.6,
                  'Chicago10': 6.2,  'Chicago11': 3.6, 'Chicago12': 3.4, 'Philadelphia1': 5.0, 'Philadelphia5': 7.9,
                  'Philadelphia6': 9.0, 'Philadelphia7': 8.9, 'Philadelphia8': 8.4, 'Philadelphia9':7.9 , 'Philadelphia10': 6.6,
                  'Philadelphia11': 5.2, 'Philadelphia12': 4.4}

    # Creating a new column by mapping the city_month variable to it's corresponding average monthly sunsine
    train["average_sunsine"] = train['city_month'].map(monthly_sunsine)
    test["average_sunsine"] = test['city_month'].map(monthly_sunsine)

    train.drop('city_month', axis=1, inplace=True)
    test.drop('city_month', axis=1, inplace=True)

    train['is_day'] = train['Hour'].apply(lambda x: 1 if 5 < x < 20 else 0)
    test['is_day'] = test['Hour'].apply(lambda x: 1 if 5 < x < 20 else 0)


    def add_distance(df):
        df_center = pd.DataFrame({"Atlanta": [33.753746, -84.386330],
                                  "Boston": [42.361145, -71.057083],
                                  "Chicago": [41.881832, -87.623177],
                                  "Philadelphia": [39.952583, -75.165222]})

        df["CenterDistance"] = df.apply(lambda row: math.sqrt((df_center[row.City][0] - row.Latitude) ** 2 +
                                                              (df_center[row.City][1] - row.Longitude) ** 2), axis=1)


    add_distance(train)
    add_distance(test)

    train = pd.concat([train,pd.get_dummies(train["City"],dummy_na=False, drop_first=False)],axis=1).drop(["City"],axis=1)
    test = pd.concat([test,pd.get_dummies(test["City"],dummy_na=False, drop_first=False)],axis=1).drop(["City"],axis=1)
    # scale Log and lat columns
    scaler = StandardScaler()
    for col in ['Latitude','Longitude']:
        scaler.fit(train[col].values.reshape(-1, 1))
        train[col] = scaler.transform(train[col].values.reshape(-1, 1))
        test[col] = scaler.transform(test[col].values.reshape(-1, 1))
    print(train.shape,test.shape)

    train_road_id = train['RowId']
    test_road_id = test['RowId']
    preds = train.iloc[:,12:27]
    train.drop(['RowId', 'Path','EntryStreetName','ExitStreetName'],axis=1, inplace=True)
    test.drop(['RowId', 'Path','EntryStreetName','ExitStreetName'],axis=1, inplace=True)

    train.drop(preds.columns.tolist(), axis=1, inplace =True)
    target1 = preds['TotalTimeStopped_p20']
    target2 = preds['TotalTimeStopped_p50']
    target3 = preds['TotalTimeStopped_p80']
    target4 = preds['DistanceToFirstStop_p20']
    target5 = preds['DistanceToFirstStop_p50']
    target6 = preds['DistanceToFirstStop_p80']
    print(train.columns)

    cat_feat = ['IntersectionId','Hour', 'Weekend','Month', 'same_street_exact', 'Intersection',
           'Atlanta', 'Boston', 'Chicago', 'Philadelphia', 'EntryType', 'ExitType']

    all_preds ={0:[],1:[],2:[],3:[],4:[],5:[]}
    all_target = [target1, target2, target3, target4, target5, target6]
    with open(pkl_path, "wb") as fp:  # Pickling
        pickle.dump([train,test, cat_feat, all_target], fp)
        print("====== Dump pickle @{} ......OK".format(pkl_path))
    exit(100)

all_preds ={0:[],1:[],2:[],3:[],4:[],5:[]}
target1=all_target[0]
dtrain = lgb.Dataset(data=train, label=target1)

param={'num_leaves': 230,   'n_estimators':10000,'early_stopping_rounds':200,
     'feature_fraction': 0.9,
     'bagging_fraction': 1,
       "adaptive":'weight1',   #无效，晕
    'max_bin': 512,
    #"learning_schedule":"adaptive",
     'max_depth': 19,
     'lambda_l1': 0.0,
     'lambda_l2': 0.0,
     'min_split_gain': 0.1,
     'min_child_weight': 20.0,
    #'min_data_in_leaf': 10,
     'learning_rate': 0.05,
     'objective': 'regression',
     'boosting_type': 'gbdt',
     'verbose': 666,
     'metric': {'rmse'}
}

nfold = 5
kf = KFold(n_splits=nfold, random_state =127, shuffle =True)
for i in range(len(all_preds)):
    t0=time.time()
    print('Training and predicting for target {}'.format(i+1))
    oof = np.zeros(len(train))
    all_preds[i] = np.zeros(len(test))
    n =1
    for train_index, valid_index in kf.split(all_target[i]):
        print("TARGET_{} fold {}".format(i,n))
        y_train,y_valid = all_target[i][train_index],all_target[i][valid_index]
        if isMORT:
            mort = LiteMORT(param).fit_1(train.iloc[train_index], y_train, eval_set=[(train.iloc[valid_index], y_valid)],discrete_feature=['Intersection','IntersectionId'])
            oof[valid_index] = mort.predict(train.iloc[valid_index])
            all_preds[i] += mort.predict(test) / nfold
        else:
            xg_train = lgb.Dataset(train.iloc[train_index],
                                   label=all_target[i][train_index]
                                   )
            xg_valid = lgb.Dataset(train.iloc[valid_index],
                                   label=all_target[i][valid_index]
                                   )

            clf = lgb.train(param, xg_train, 10000, valid_sets = [xg_valid],categorical_feature=cat_feat
                            , verbose_eval=100, early_stopping_rounds = 200)
            oof[valid_index] = clf.predict(train.iloc[valid_index], num_iteration=clf.best_iteration)

            all_preds[i] += clf.predict(test, num_iteration=clf.best_iteration) / nfold
        n = n + 1
        #break
    fold_score = np.sqrt(mean_squared_error(all_target[i], oof))
    print("\n\nTARGET_{} CV RMSE: {:<0.4f} time={:.4g}".format(i,np.sqrt(mean_squared_error(all_target[i], oof)),time.time()-t0))
path=""
if some_rows is None:
    # test_predictions[['TransactionID', 'isFraud']].to_csv(f'submit_{some_rows}_{0.5}.csv', index=False,compression='gzip')
    path = f'{data_root}/[{model}]_{some_rows}_{fold_score:.5f}_F{nfold}_.csv.gz'
    data2 = pd.DataFrame(all_preds).stack()
    data2 = pd.DataFrame(data2)
    submission = pd.read_csv(f'{data_root}/sample_submission.csv')
    submission['Target'] = data2[0].values
    submission.to_csv(path, index=False,float_format='%.4f',compression='gzip' )
input(f"......Save submit @{path}......")