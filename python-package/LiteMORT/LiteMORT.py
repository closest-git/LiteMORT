import gc
import numpy as np
import pandas as pd
from ctypes import *
pd.options.display.max_columns = 999
pd.set_option('expand_frame_repr', False)

'''
    sklearn style
'''

class LiteMORT_params(object):
    def OnArgs(self,dict_param):
        self.isOK = False
        if 'metric' in dict_param:
            self.metric = dict_param['metric']
        if 'num_leaves' in dict_param:
            self.num_leaves = dict_param['num_leaves']
        if 'max_depth' in dict_param:
            self.max_depth = dict_param['max_depth']
        if 'learning_rate' in dict_param:
            self.learning_rate = dict_param['learning_rate']
        if 'bagging_fraction' in dict_param:
            self.subsample = dict_param['bagging_fraction']
        if 'feature_fraction' in dict_param:
            self.feature_sample = dict_param['feature_fraction']
        if 'max_bin' in dict_param:
            self.feature_quanti = dict_param['max_bin']
        if 'min_data_in_leaf' in dict_param:
            self.min_child_samples = dict_param['min_data_in_leaf']
        if 'boost_from_average' in dict_param:
            self.boost_from_average = dict_param['boost_from_average']
        if 'verbosity' in dict_param:
            self.verbose = dict_param['verbosity']
        if 'early_stop' in dict_param:
            self.early_stopping_rounds = dict_param['early_stop']
        if 'num_boost_round' in dict_param:
            self.n_estimators = dict_param['num_boost_round']


    def __init__(self,objective,fold=5,lr=0.01,round=1000,early_stop=50,subsample=1,feature_sample=1,leaves=32,
                 max_bin=256,metric='mse',min_child=20,argv=None):
        #objective='outlier'
        self.isOK = False
        self.env = 'default'
        self.use_gpu = True
        self.version = 'v1'
        self.feature_quanti=max_bin
        self.feature_sample = feature_sample
        self.min_child_samples = min_child
        self.subsample = subsample
        self.NA = -1
        self.normal = 0
        self.histo_bin_map = 1       #1,      #0-quantile,1-frequency 3 dcrimini on Y
        self.node_task=0         #0:histo(X) split(X),     1:histo(X) split(Y) 2:REGRESS_X
        self.objective=objective
        self.metric=metric
        self.k_fold = fold
        self.learning_rate = lr
        self.n_estimators = round
        self.num_leaves = leaves
        self.early_stopping_rounds = early_stop
        self.verbose = 1
        #self.boost_from_average = 0,

        self.OnArgs(argv)

def fancy_impute():
    # https://www.kaggle.com/athi94/investigating-imputation-methods
    # user_train = KNN(k=1).fit_transform(user_train);        user_test = KNN(k=1).fit_transform(user_test)
    # user_train = SimpleFill.fit_transform(user_train)
    # user_test = SimpleFill.fit_transform(user_test)
    return

# https://www.kaggle.com/ashishpatel26/1-23-pb-first-try-to-think
def clearRare(train,test,columnname, limit=1000):
    # you may search for rare categories in train, train&test, or just test
    # vc = pd.concat([train[columnname], test[columnname]], sort=False).value_counts()
    vc = test[columnname].value_counts()

    common = vc > limit
    common = set(common.index[common].values)
    print("Set", sum(vc <= limit), columnname, "categories to 'other';", end=" ")

    train.loc[train[columnname].map(lambda x: x not in common), columnname] = 'other'
    test.loc[test[columnname].map(lambda x: x not in common), columnname] = 'other'
    print("now there are", train[columnname].nunique(), "categories in train")

#https://www.kaggle.com/karkun/sergey-ivanov-msu-mmp
def replace_atypical_categories(df_train, df_test, columnname, pct = .01, base_df = "test"):
    """ Replace all categories in a categorical variable whenever the number of
    observations in the test or train data set is lower than pct percetage of the
    total number of observations.  The replaced categories are assigned to "other" category.
    """
    if base_df == "test":
        limit  = len(df_test) *pct
        vc = df_test[columnname].value_counts()
    else:
        limit  = len(df_train) *pct
        vc = df_train[columnname].value_counts()

    common = vc > limit
    common = set(common.index[common].values)
    print("Set", sum(vc < limit), columnname, "categories to 'other';", end=" ")

    df_train.loc[df_train[columnname].map(lambda x: x not in common), columnname] = 'other'
    df_test.loc[df_test[columnname].map(lambda x: x not in common), columnname] = 'other'
    print("now there are", df_train[columnname].nunique(), "categories in train")
    return df_train, df_test

#https://www.kaggle.com/maheshdadhich/strength-of-visualization-python-visuals-tutorial
#https://www.kaggle.com/maheshdadhich/strength-of-visualization-python-visuals-tutorial
#https://stackoverflow.com/questions/18689235/numpy-array-replace-nan-values-with-average-of-columns
# For numerical values you should go with mean, and if there are some outliers try median (since it is much less sensitive to them).
# Imputer过于简单，以后转为由C库处理
def OnMissingValue(df):
    if False:
        imputer = Imputer(missing_values="NaN", strategy="mean")
        imputed_DF = pd.DataFrame(imputer.fit_transform(df))
        #imputed_DF = pd.DataFrame(fill_NaN.fit_transform(DF))
        imputed_DF.columns = df.columns
        imputed_DF.index = df.index
        df = imputed_DF
        #import missingno as msno
        #msno.matrix(census_data)
    df.fillna(0, inplace=True)
    return df
    #
    mA = df.values
    col_mean = np.nanmean(mA, axis=0)
    print(col_mean)
    # Find indicies that you need to replace
    inds = np.where(np.isnan(mA))
    # Place column means in the indices. Align the arrays using take
    mA[inds] = np.take(col_mean, inds[1])

    gc.collect()
    #print(" user_train.fillna={}".format(user_train.shape))
    # user_train.dropna(inplace=True);       print(" user_train.dropna={}".format(user_train.shape))
    return df

class M_argument(Structure):
    _fields_ = [    ('Keys',c_char_p),
                    ('Values',c_float),
                    ('text', c_char_p),
               ]

class LiteMORT(object):
    def __init__(self, params,dll_path='F:/Project/LiteMORT/LiteMORT.dll', fix_seed=None):
        # dll_path = 'F:/Project/LiteMORT/LiteMORT.dll'
        self.dll_path = dll_path
        if False:
            arr_path = "../input/df_ndarray.csv"
            np.savetxt(arr_path, data, delimiter="", fmt='%12g', )
            print("====== arr_file@{} size={} dll={}".format(arr_path, data.shape, dll_path))
        self.dll = cdll.LoadLibrary(dll_path)

        self.mort_init = self.dll.LiteMORT_init
        self.mort_init.argtypes = [POINTER(M_argument), c_int,c_size_t]

        #self.mort_set_feat = self.dll.LiteMORT_set_feat
        #self.mort_set_feat.argtypes = [POINTER(M_argument), c_int, c_size_t]

        self.mort_fit = self.dll.LiteMORT_fit
        self.mort_fit.argtypes = [POINTER(c_float), POINTER(c_double), c_size_t, c_size_t,
                                  POINTER(c_float),POINTER(c_double),c_size_t, c_size_t]
        self.mort_fit.restype = None

        self.mort_predcit = self.dll.LiteMORT_predict
        self.mort_predcit.argtypes = [POINTER(c_float), POINTER(c_double), c_size_t, c_size_t, c_size_t]

        self.mort_eda = self.dll.LiteMORT_EDA
        self.mort_eda.argtypes = [POINTER(c_float), POINTER(c_double), c_size_t, c_size_t,c_size_t, POINTER(M_argument), c_int,c_size_t]

        self.mort_imputer_f = self.dll.LiteMORT_Imputer_f
        self.mort_imputer_f.argtypes = [POINTER(c_float), POINTER(c_double), c_size_t, c_size_t, c_size_t]

        self.mort_imputer_d = self.dll.LiteMORT_Imputer_d
        self.mort_imputer_d.argtypes = [POINTER(c_double), POINTER(c_double), c_size_t, c_size_t, c_size_t]

        self.mort_clear = self.dll.LiteMORT_clear

        self.init(params)

    #  注意 Y_t与y_train不一样
    def Y_t(self, y_train, np_type):
        # print(type(y_train))
        if type(y_train) is pd.Series:
            np_target = y_train.values.astype(np_type)
        elif isinstance(y_train, pd.DataFrame):
            np_target = y_train.values.astype(np_type)
        else:
            np_target = y_train.astype(np_type)
        return np_target

    def X_t(self, X_train_0, target_type):
        # mort_fit需要feature优先
        if isinstance(X_train_0, pd.DataFrame):
            np_mat = X_train_0.values
        else:
            np_mat = X_train_0

        if np.isfortran(np_mat):
            np_fortran = np_mat
        else:
            print("X_t[{}] asfortranarray".format(np_mat.shape));
            np_fortran = np.asfortranarray(np_mat)
        # Transpose just changes the strides, it doesn't touch the actual array
        if np_fortran.dtype != target_type:
            print("X_t[{}] astype {}=>{}".format(np_fortran.shape,np_fortran.dtype,target_type));
            np_out = np_fortran.astype(target_type)  # .transpose()
            #del np_fortran;     gc.collect()
        else:
            np_out = np_fortran
        gc.collect()
        return np_out

    def init(self, params, flag=0x0):
        if not isinstance(params, LiteMORT_params):
            if isinstance(params,dict):
                pass
            else:
                return
            self.objective = params["objective"]
            self.params = LiteMORT_params(self.objective,argv=params)
        else:
            self.params = params

        ca_list = []
        for k, v in self.params.__dict__.items():
            ca = M_argument()
            ca.Keys = k.encode('utf8')  # Python 3 strings are Unicode, char* needs a byte string
            if isinstance(v, str):
                ca.text = v.encode('utf8')
            elif isinstance(v, bool):
                ca.Values = (c_float)(v==True)
            else:
                ca.Values = (c_float)(v)  # Interface unclear, how would target function know how many floats?

            # ca.Index = v[3]
            ca_list.append(ca)
        ca_array = (M_argument * len(ca_list))(*ca_list)
        self.mort_init(ca_array, len(ca_array),0)

    '''
            # v0.2
            # v0.3
                feat_dict   cys@1/10/2019
    '''
    def fit(self,X_train_0, y_train, X_test, y_test, feat_dict=None, params=None,flag=0x0):
        gc.collect()

        nFeat = X_train_0.shape[1]
        if feat_dict is not None:
            print("feat_dict={}".format(feat_dict))

        print("====== LiteMORT_fit X_train_0={} y_train={}......".format(X_train_0.shape, y_train.shape))
        train_y = self.Y_t(y_train, np.float64)
        eval_y = self.Y_t(y_test, np.float64)
        train_X = self.X_t(X_train_0, np.float32)
        eval_X = self.X_t(X_test, np.float32)
        nTrain, nFeat, nTest = train_X.shape[0], train_X.shape[1], eval_X.shape[0]

        self.mort_fit(
                      train_X.ctypes.data_as(POINTER(c_float)), train_y.ctypes.data_as(POINTER(c_double)), nFeat, nTrain,
                      eval_X.ctypes.data_as(POINTER(c_float)), eval_y.ctypes.data_as(POINTER(c_double)), nTest,
                      0)  # 1 : classification
        if not(train_X is X_train_0):
            del train_X;     gc.collect()
        if not(eval_X is X_test):
            del eval_X;     gc.collect()
        if not(eval_y is y_test):
            del eval_y;     gc.collect()
        if not(train_y is y_train):
            del train_y;     gc.collect()


    def predict(self, X_, flag=0x0):
        # print("====== LiteMORT_predict X_={} ......".format(X_.shape))
        dim, nFeat = X_.shape[0], X_.shape[1];
        Y_ = np.zeros(dim, dtype=np.float64)
        tY = Y_ #self.Y_t(Y_, np.float64)
        tX = self.X_t(X_, np.float32)
        self.mort_predcit(tX.ctypes.data_as(POINTER(c_float)), tY.ctypes.data_as(POINTER(c_double)), nFeat, dim, 0)
        if not(tX is X_):
            del tX;     gc.collect()
        return Y_

    def EDA(self, params,dataX_, dataY_,nValid,desc_list, flag=0x0):
        # print("====== LiteMORT_EDA X_={} ......".format(X_.shape))
        nSamp, nFeat = dataX_.shape[0], dataX_.shape[1];
        ca_array,nFeat_desc = None,len(desc_list)
        if nFeat_desc>0:
            assert(nFeat_desc==nFeat)
            ca_array = (M_argument * len(desc_list))(*desc_list)
        #nValid, nFeat = validX_.shape[0], trainX_.shape[1];
        if dataY_ is None:
            dataY_ = np.zeros(nSamp, dtype=np.float64)
        dataX = self.X_t(dataX_, np.float32)
        #validX = self.X_t(validX_, np.float32)
        self.mort_eda(dataX.ctypes.data_as(POINTER(c_float)),dataY_.ctypes.data_as(POINTER(c_double)) ,nFeat, nSamp,nValid,
                      ca_array,nFeat_desc,  0)
        return

    #奇怪的教训，会影响其它列,需要重写，暂时这样！！！
    def Imputer(self, params,X_, Y_,np_float, flag=0x0):
        # print("====== LiteMORT_EDA X_={} ......".format(X_.shape))
        dim, nFeat = X_.shape[0], X_.shape[1];
        if Y_ is None:
            Y_ = np.zeros(dim, dtype=np.float64)
        #print("head={}\ntail={}".format(X_.head(),X_.tail()))
        tX = self.X_t(X_, np_float)
        #tX = self.X_t(tX, np_float)
        if np_float==np.float32:
            self.mort_imputer_f(tX.ctypes.data_as(POINTER(c_float)),Y_.ctypes.data_as(POINTER(c_double)) , nFeat, dim, 0)
        elif np_float==np.float64:
            self.mort_imputer_d(tX.ctypes.data_as(POINTER(c_double)), Y_.ctypes.data_as(POINTER(c_double)), nFeat, dim, 0)
        else:
            assert(0)
        imputed_DF = pd.DataFrame(tX)
        imputed_DF.columns = X_.columns;        imputed_DF.index = X_.index
        X_ = imputed_DF
        #print("head={}\ntail={}".format(X_.head(), X_.tail()))
        return X_

    def Clear(self):
        self.mort_clear();

if __name__ == "__main__":

    df=pd.read_csv('D:\\LightGBM-master\\examples\\regression\\test_000.txt',header=None,sep='\t' )
    df =df.astype(np.float32)
    cols = df.columns
    train_X = df[cols[1:]]
    train_Y = df[cols[0]]
    if False:
        for i in range(len(y_devel)):
            y_devel[i] = 0  # 测试各种情况
    valid_X, valid_Y=train_X,train_Y
    eval_set = [(valid_X, valid_Y)]  # [(valid, y_valid)]

    mort_params = {
        'histo_bins': 0, 'feature_quanti': 1024, 'feature_sample': 1, 'min_child_samples': 1, 'subsample': 1,
         'NA': -1, 'normal': 0,
        'histo_bin_map':1,    #0-quantile,1-frequency                                   #'histo_algorithm': 0,
        'k_fold': 5,
        'learning_rate': 0.03,
        'n_estimators': 1, 'num_leaves': 31,
        "early_stopping_rounds": 50, "verbose": 100,
    }
    mort = LiteMORT()
    mort.init(mort_params)
    if mort_params['NA'] == -1:
        print("---- !!!No data imputation!!!----")
    else:
        if True:  # 奇怪的教训，会影响其它列,需要重写，暂时这样！！！
            user_train[features] = mort.Imputer(mort_params, user_train[features], None, np.float32)
            user_test[features] = mort.Imputer(mort_params, user_test[features], None, np.float32)
        else:
            user_train = OnMissingValue(user_train);
            user_test = OnMissingValue(user_test)
    if True:   # EDA algorithm
        all_data = train_X
        print("====== all_data for EDA={}\n".format(all_data.shape))
        mort.EDA(mort_params,train_X, None,0)
        del all_data;
    gc.collect()
    mort.fit(mort_params, train_X, train_Y, valid_X, valid_Y)
