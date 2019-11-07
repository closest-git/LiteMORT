#hyperparameter optimization
from bayes_opt import BayesianOptimization
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error
from litemort import *
import numpy as np

def hyparam_search(func_core,pds,n_init=5, n_iter=12):
    optimizer = BayesianOptimization(func_core, pds, random_state=7)
    optimizer.maximize(init_points=n_init, n_iter=n_iter)
    print(optimizer.max)
    for i, res in enumerate(optimizer.res):
        print("Iteration {}: \n\t{}".format(i, res))
    input(f"......BayesianOptimization is OK......")
    return optimizer.max

def _feat_select_core_(**kwargs):
    print(kwargs)
    feats=[]
    nFeat = len(kwargs)
    no=0
    feat_factor = np.zeros(nFeat)
    for k,v in kwargs.items():
        feats.append(k)
        feat_factor[no]=v;  no=no+1
    train_tmp = train_data[feats]
    print('Training with {} features'.format(train_tmp.shape[1]))
    x_train, x_val, y_train, y_val = train_test_split(train_tmp, train_target, test_size = 0.2, random_state = 42)
    feat_factor=feat_factor.astype(np.float32)
    param_mort["feat_factor"]=feat_factor
    mort = LiteMORT(param_mort).fit(x_train, y_train,eval_set=[(x_val, y_val)])
    eval_pred = mort.predict(x_val)
    score = np.sqrt(mean_squared_error(eval_pred, y_val))
    return -score

#10/20/2019 实测效果较差  BayesianOptimization并不适合非常多的参数
def MORT_feat_bayesian_search(train,target,feat_fix,feat_select,n_init=5, n_iter=12):
    global train_data,train_target
    train_data,train_target = train,target
    print(f"train={train_data.shape} target={train_target.shape}")
    feat_useful=[]
    pds = {}
    for feat in feat_fix:
        pds[feat]=(1, 1)
    for feat in feat_select:
        pds[feat]=(0, 1)
    optimizer = BayesianOptimization(_feat_select_core_, pds, random_state=42)
    optimizer.maximize(init_points=n_init, n_iter=n_iter)
    print(optimizer.max)
    for i, res in enumerate(optimizer.res):
        print("Iteration {}: \n\t{}".format(i, res))
    input(f"......BayesianOptimization is OK......")
    return feat_useful

def MORT_feat_select_(dataX,dataY,feat_fix,feat_select,select_params,nMostSelect=10):
    nFeat=len(feat_fix)+len(feat_select)
    feats = []
    no = 0
    feat_factor = np.zeros(nFeat)
    for feat in feat_fix:
        feats.append(feat)
        feat_factor[no] = 1
        no = no + 1
    for feat in feat_select:
        feats.append(feat)
        feat_factor[no] = 0
        no = no + 1
    data_tmp = dataX[feats]

    #select_params['learning_rate'] = select_params['learning_rate']*2
    #select_params['early_stopping_rounds'] = 100
    #select_params['verbose'] = 0

    print(f"======MORT_feat_select_ nFix={len(feat_fix)} nSelect={len(feat_select)} ")
    feat_useful_ = []
    if True:
        for loop in range(nMostSelect):
            select_params["feat_factor"] = feat_factor
            if('split_idxs' in select_params):
                assert(len(select_params['split_idxs'])>0)
                tr_idx, val_idx=select_params['split_idxs'][0]
                y_train = dataY[tr_idx]
                x_train =data_tmp.iloc[tr_idx, :]
                x_val, y_val = data_tmp.iloc[val_idx, :], dataY[val_idx]
            else:
                x_train, x_val, y_train, y_val = train_test_split(data_tmp, dataY, test_size=0.2, random_state=42)
            cat_features = select_params['category_features'] if 'category_features' in select_params else None
            mort = LiteMORT(select_params).fit(x_train, y_train, eval_set=[(x_val, y_val)], categorical_feature=cat_features)
            feat_factor_1 = mort.params.feat_factor
            rank = np.argsort(feat_factor_1)[::-1]
            nAdd=0
            for no in rank:
                if feat_factor[no]==1:
                    continue
                if feat_factor_1[no] > 0:
                    feat_useful_.append(feats[no]);     nAdd=nAdd+1
                    print(f"___MORT_feat_select___@{loop}:\t{feats[no]}={feat_factor_1[no]:.5g}" )
                    feat_factor[no]=1
            if nAdd==0:
                print(f"___MORT_feat_select___@{loop} break out")
            print(f"___MORT_feat_select___@{loop} feat_useful_={feat_useful_}")
        input(f"......MORT_feat_select_ is OK......")

    else:       #original forward feature selection
        x_train, x_val, y_train, y_val = train_test_split(dataX[feat_fix], dataY, test_size = 0.2, random_state = 42)
        mort = LiteMORT(param_mort).fit(x_train, y_train, eval_set=[(x_val, y_val)])
        predictions = mort.predict(x_val)
        rmse_score = np.sqrt(mean_squared_error(y_val, predictions))
        print("RMSE baseline val score: ", rmse_score)
        best_score = rmse_score
        train_columns = list(dataX.columns[13:])
        for num, i in enumerate(train_columns):
            train_tmp = dataX[feat_fix + feat_useful_ + [i]]
            x_train, x_val, y_train, y_val = train_test_split(train_tmp, dataY, test_size=0.2, random_state=42)
            mort = LiteMORT(param_mort).fit(x_train, y_train, eval_set=[(x_val, y_val)])
            predictions = mort.predict(x_val)
            rmse_score = np.sqrt(mean_squared_error(y_val, predictions))
            percent = (best_score-rmse_score) / best_score*100.0;
            if rmse_score < best_score:
                print(f'------ \"{i}\" is usefull {percent:.3g}% [{best_score:.7g}=>{rmse_score:.7g}]------')
                best_score = rmse_score
                feat_useful_.append(i)
            else:
                pass    #rint('Column {} is not usefull'.format(i))
    print(feat_useful_)
    return feat_useful_