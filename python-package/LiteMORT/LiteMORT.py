import gc
import numpy as np
import pandas as pd
from ctypes import *
from .libpath import find_lib_path
from .LiteMORT_preprocess import *
from .LiteMORT_problems import *
from sklearn import preprocessing
import os
import warnings

'''
BUG
    1 predict和predict_raw 参见case_poct.py
    
'''
class MORT_exception(Exception):
    """Error throwed by LiteMORT"""
    pass
def _check_call(ret,normal_value=0x0):
    """Check the return value of C API call
    Parameters
    ----------
    ret : int
        return value from API calls
    """
    if ret != normal_value:
        raise MORT_exception("_check_call failed at {}-{}".format(ret,normal_value))
'''
    sklearn style
'''

class LiteMORT_params(object):
    def alias_param(self,key,default_value,dict_param,alia_list):
        for alias in alia_list:
            if alias not in dict_param:
                continue
            if alias!=key:
                warnings.warn("Found `{}`(alias of {}) in params. Will use it instead of argument".format(alias,key))
            value = (type(default_value))(dict_param[alias])
            return value
        return default_value

    def OnArgs(self,dict_param):
        self.isOK = False
        if 'metric' in dict_param:
            if(dict_param['metric'] in ['regression_l2','mean_squared_error', 'mse', 'l2','l2_root','root_mean_squared_error', 'rmse'] ):
                self.metric = 'mse'
            elif (dict_param['metric'] in ['regression_l1','mean_absolute_error', 'mae', 'l1']):
                self.metric = 'mae'
            else:
                self.metric = dict_param['metric']
        if 'num_leaves' in dict_param:
            self.num_leaves = dict_param['num_leaves']
        if 'max_depth' in dict_param:
            self.max_depth = dict_param['max_depth']
        if 'learning_rate' in dict_param:
            self.learning_rate = dict_param['learning_rate']
        if 'bagging_fraction' in dict_param:
            self.subsample = dict_param['bagging_fraction']
        #if 'subsample' in dict_param:
        #    self.subsample = dict_param['subsample']
        self.subsample = self.alias_param('subsample',1.0,dict_param,['subsample','bagging_fraction','sub_row'])
        #if 'feature_fraction' in dict_param:
        #    self.feature_sample = dict_param['feature_fraction']
        self.feature_sample = self.alias_param('sub_feature', 1.0, dict_param, ['feature_sample', 'feature_fraction', 'sub_feature', 'colsample_bytree'])
        if 'max_bin' in dict_param:
            self.feature_quanti = dict_param['max_bin']
        if 'min_data_in_leaf' in dict_param:
            self.min_child_samples = dict_param['min_data_in_leaf']
        if 'boost_from_average' in dict_param:
            self.boost_from_average = dict_param['boost_from_average']
        if 'iter_refine' in dict_param:
            self.iter_refine = dict_param['iter_refine']
        if 'verbosity' in dict_param:
            self.verbose = dict_param['verbosity']
        #if 'early_stop' in dict_param:
        #    self.early_stopping_rounds = dict_param['early_stop']
        self.early_stopping_rounds = self.alias_param('early_stop',50,dict_param,['early_stop',"early_stopping_round", "early_stopping_rounds", "early_stopping"])
        self.n_estimators = self.alias_param('num_trees',self.n_estimators,dict_param,
            ["num_boost_round", "num_iterations", "num_iteration", "num_tree", "num_trees", "num_round", "num_rounds", "n_estimators"] )


    def __init__(self,objective,fold=5,lr=0.1,round=50,early_stop=50,subsample=1,feature_sample=1,leaves=31,
                 max_bin=256,metric='mse',min_child=20,max_depth=-1,subsample_for_bin=200000,argv=None):
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

class M_argument(Structure):
    _fields_ = [    ('Keys',c_char_p),
                    ('Values',c_float),
                    ('text', c_char_p),
               ]


class LiteMORT(object):
    #problem = None
    #preprocess = None
    #hEDA=None

    def load_dll(self):
        lib_path = find_lib_path()
        if len(lib_path) == 0:
            return None
        # lib_path.append( 'F:/Project/LiteMORT/LiteMORT.dll' )

        self.dll_path = lib_path[0]
        self.dll = cdll.LoadLibrary(self.dll_path)
        print("======Load LiteMORT library @{}".format(self.dll_path))
        self.mort_init = self.dll.LiteMORT_init
        self.mort_init.argtypes = [POINTER(M_argument), c_int, c_size_t]
        self.mort_init.restype = c_void_p

        # self.mort_set_feat = self.dll.LiteMORT_set_feat
        # self.mort_set_feat.argtypes = [POINTER(M_argument), c_int, c_size_t]

        self.mort_fit = self.dll.LiteMORT_fit
        self.mort_fit.argtypes = [c_void_p,POINTER(c_float), POINTER(c_double), c_size_t, c_size_t,
                                  POINTER(c_float), POINTER(c_double), c_size_t, c_size_t]
        self.mort_fit.restype = None

        self.mort_fit_1 = self.dll.LiteMORT_fit_1
        self.mort_fit_1.argtypes = [c_void_p, POINTER(M_COLUMN), POINTER(M_COLUMN), c_size_t,c_size_t,
                                  POINTER(M_COLUMN), POINTER(M_COLUMN), c_size_t, c_size_t]
        self.mort_fit_1.restype = None

        self.mort_predcit = self.dll.LiteMORT_predict
        self.mort_predcit.argtypes = [c_void_p,POINTER(c_float), POINTER(c_double), c_size_t, c_size_t, c_size_t]

        self.mort_eda = self.dll.LiteMORT_EDA
        self.mort_eda.argtypes = [c_void_p,POINTER(c_float), POINTER(c_double), c_size_t, c_size_t, c_size_t,
                                  POINTER(M_argument), c_int, c_size_t]
        #self.mort_eda.restype = c_void_p

        self.mort_imputer_f = self.dll.LiteMORT_Imputer_f
        self.mort_imputer_f.argtypes = [POINTER(c_float), POINTER(c_double), c_size_t, c_size_t, c_size_t]

        self.mort_imputer_d = self.dll.LiteMORT_Imputer_d
        self.mort_imputer_d.argtypes = [POINTER(c_double), POINTER(c_double), c_size_t, c_size_t, c_size_t]

        self.mort_clear = self.dll.LiteMORT_clear
        self.mort_clear.argtypes = [c_void_p]

    def __init__(self, params,fix_seed=None):
        self.problem = None
        self.preprocess = None
        self.hLIB = None
        self.load_dll()
        self._n_classes = None
        self.init(params)
        if self.params.objective == "binary":
            self.problem = Mort_BinaryClass()
        elif self.params.objective == "regression":
            self.problem = Mort_Regressor()

    def __del__(self):
        try:
            print("LiteMORT::__del__...".format())
            if self.hLIB is not None:
                self.mort_clear(self.hLIB)
            self.hLIB = None
        except AttributeError:
            pass

    # For DataFrames of mixed type calling .values converts multiple blocks (for each dtype) into one numpy array of a all-encompassing dtype.
    # This conversion can be slow for large frames!!!
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

        if self.params.objective=="binary":
            self._n_classes = 2
        else:
            pass

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
        self.hLIB = self.mort_init(ca_array, len(ca_array),0)


    def EDA(self,flag=0x0):
        '''

        :param flag:
        :return:
        '''
        return
        nFeat,nSamp,no=self.preprocess.nFeature,self.preprocess.nSample,0
        X, y = self.preprocess.train_X, self.preprocess.train_y
        categorical_feature = self.preprocess.categorical_feature
        ca_list = []
        for feat in self.preprocess.features:
            ca = M_COLUMN()
            ca.name = str(feat).encode('utf8')
            ca.values = (c_float)(0)
            if (categorical_feature is not None) and (feat in categorical_feature or no in categorical_feature):
                # ca.Values = (c_float)(1)
                ca.text = 'category'.encode('utf8')
            ca_list.append(ca)
            no=no+1
        self.column_array = (M_COLUMN * len(ca_list))(*ca_list)
        #self.EDA_000(self.mort_params, all_data, None, user_test.shape[0], ca_list)
        if False:
            self.mort_eda(self.hLIB,X,y, nFeat, nSamp,0,self.column_array, len(ca_list), 0)

    def fit_index(self, X, y,train_index, eval_set=None, feat_dict=None, categorical_feature=None, params=None, flag=0x0):
        gc.collect()
        self.preprocess = Mort_Preprocess(X, y, categorical_feature=categorical_feature)

        if (eval_set is not None and len(eval_set) > 0):
            eval_index = eval_set[0]
        pass
        print("====== LiteMORT_fit_index X_={} y_={}......".format(X.shape, y.shape))

        self.EDA(flag)

        self.mort_fit_index(self.hLIB, self.preprocess.train_X, self.preprocess.train_y, nFeat, nTrain,
                      self.preprocess.eval_X, self.preprocess.eval_y, nTest, 0)  # 1 : classification
        return self

    def fit(self, X_train_0, y_train, eval_set=None, feat_dict=None, categorical_feature=None, params=None, flag=0x0):
        gc.collect()
        # self.preprocess.fit(X_train_0, y_train)
        # self.preprocess.transform(eval_set)
        self.preprocess = Mort_Preprocess(X_train_0, y_train, categorical_feature=categorical_feature)

        if (eval_set is not None and len(eval_set) > 0):
            X_test, y_test = eval_set[0]
        #a0=X_train_0['id_trade'].min();       a1=X_train_0['id_trade'].max();
        print("====== LiteMORT_fit X_train_0={} y_train={}......".format(X_train_0.shape, y_train.shape))
        train_y = self.problem.OnY(y_train, np.float64)
        # train_y = self.Y_t(y_train, np.float64)
        train_X = self.X_t(X_train_0, np.float32)
        #b0 = train_X[:,3].min();        b1 = train_X[:,3].max();
        self.preprocess.train_X = train_X.ctypes.data_as(POINTER(c_float))
        self.preprocess.train_y = train_y.ctypes.data_as(POINTER(c_double))
        nTrain, nFeat, nTest = train_X.shape[0], train_X.shape[1], 0
        eval_X, eval_y = None, None
        if eval_set is not None:
            eval_y0 = self.problem.OnY(y_test, np.float64)
            eval_X0 = self.X_t(X_test, np.float32)
            nTest = eval_X0.shape[0]
            self.preprocess.eval_X, self.preprocess.eval_y = eval_X0.ctypes.data_as(
                POINTER(c_float)), eval_y0.ctypes.data_as(POINTER(c_double))
        self.EDA(flag)

        self.mort_fit(self.hLIB, self.preprocess.train_X, self.preprocess.train_y, nFeat, nTrain,
                      self.preprocess.eval_X, self.preprocess.eval_y, nTest, 0)  # 1 : classification
        if not (train_X is X_train_0):
            del train_X;
            gc.collect()
        if not (train_y is y_train):
            del train_y;
            gc.collect()
        if eval_X is not None and not (eval_X is X_test):
            del eval_X;
            gc.collect()
        if eval_y is not None and not (eval_y is y_test):
            del eval_y;
            gc.collect()

        return self
    '''
            # v0.2
            # v0.3
                feat_dict   cys@1/10/2019
    '''
    def fit_1(self,X_train_0, y_train,eval_set=None,  feat_dict=None,categorical_feature=None, params=None,flag=0x0):
        print("====== LiteMORT_fit X_train_0={} y_train={}......".format(X_train_0.shape, y_train.shape))
        gc.collect()
        self.train_set = Mort_Preprocess( X_train_0,y_train,categorical_feature=categorical_feature,cXcY=True)
        if(eval_set is not None and len(eval_set)>0):
            X_test, y_test=eval_set[0]
            self.eval_set = Mort_Preprocess(X_test, y_test, categorical_feature=categorical_feature,cXcY=True)
        #self.EDA(flag)

        nTrain, nFeat, nTest = self.train_set.nSample,self.train_set.nFeature, self.eval_set.nSample
        #self.mort_fit(self.hLIB,self.preprocess.train_X,self.preprocess.train_y, nFeat, nTrain,self.preprocess.eval_X, self.preprocess.eval_y, nTest,0)  # 1 : classification
        self.mort_fit_1(self.hLIB,self.train_set.cX,self.train_set.cY, nFeat,nTrain,self.eval_set.cX, self.eval_set.cY, nTest,0)  # 1 : classification
        return self


    def predict(self, X_,pred_leaf=False, pred_contrib=False,raw_score=False, flag=0x0):
        """Predict class or regression target for X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        Returns
        -------
        y : array, shape (n_samples,)
            The predicted values.
        """
        # print("====== LiteMORT_predict X_={} ......".format(X_.shape))
        Y_ = self.predict_raw(X_,pred_leaf, pred_contrib,raw_score, flag)
        Y_ = self.problem.OnResult(Y_,pred_leaf,pred_contrib,raw_score)
        return Y_

    def predict_raw(self, X_,pred_leaf=False, pred_contrib=False,raw_score=False,flag=0x0):
        """Predict class probabilities for X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        Raises
        ------
        AttributeError
            If the ``loss`` does not support probabilities.

        Returns
        -------
        p : array, shape (n_samples, n_classes)
            The class probabilities of the input samples. The order of the
            classes corresponds to that in the attribute `classes_`.
        """
        dim, nFeat = X_.shape[0], X_.shape[1];
        Y_ = np.zeros(dim, dtype=np.float64)
        tY = Y_  # self.Y_t(Y_, np.float64)
        tX = self.X_t(X_, np.float32)
        self.mort_predcit(self.hLIB, tX.ctypes.data_as(POINTER(c_float)), tY.ctypes.data_as(POINTER(c_double)), nFeat,dim, 0)
        if not (tX is X_):
            del tX;
        gc.collect()
        return Y_

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


