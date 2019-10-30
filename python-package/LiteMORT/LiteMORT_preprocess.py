import pandas as pd
import numpy as np
import gc
from ctypes import *
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
import random

class M_COLUMN(Structure):
    _fields_ = [    ('name',c_char_p),
                    ('data',c_void_p),
                    ('dtype', c_char_p),
                    ('type_x', c_char_p),
                    ('v_min', c_double),
                    ('v_max', c_double),
                    ('representive', c_float),
               ]

def Mort_PickSamples(pick_samples,df_train,df_test):
    nTrain = df_train.shape[0]
    random.seed(42)
    subset = random.sample(range(nTrain), pick_samples)
    df_train = df_train.iloc[subset, :].reset_index(drop=True)
    print('====== Mort_PickSamples ... df_train={}'.format(df_train.shape))
    if df_test is not None:
        nTest =df_test.shape[0]
        subset = random.sample(range(nTest), pick_samples)
        df_test = df_test.iloc[subset, :].reset_index(drop=True)
        print('====== Mort_PickSamples ... df_test={}'.format(df_test.shape))
    return df_train,df_test

class Mort_Preprocess(object):
    #nFeature,nSample=0,0
    #features = []
    #categorical_feature=[]
    #train_X,    train_y=None,None
    #eval_X,     eval_y = None, None
    def column_info(self,feat,X,categorical_feature=None,discrete_feature=None):
        col = M_COLUMN()
        col.name = str(feat).encode('utf8')
        col.data = None
        col.dtype = None
        col.representive = np.float32(0)
        x_info,type_x = '',''
        if isinstance(X, pd.DataFrame):
            narr = None
            isCat =(categorical_feature is not None) and (feat in categorical_feature)
            dtype = X[feat].dtype
            if isCat or dtype.name == 'category':
                x_info = 'category'
                type_x = '*'
            isDiscrete = (discrete_feature is not None) and (feat in discrete_feature)
            if isDiscrete:
                x_info = 'discrete'
                type_x = '#'
            if X[feat].dtype.name == 'category':
                narr = X[feat].cat.codes.values
            elif is_numeric_dtype(X[feat]):
                narr = X[feat].values
            else:
                pass
        elif isinstance(X, pd.Series):
            type_x='S'
            x_info = 'Series'
            narr = X.values
        elif isinstance(X, np.ndarray):
            narr = X
        else:
            pass
        if narr is not None:
            col.type_x=str(type_x).encode('utf8')
            col.v_min=narr.min();      col.v_max=narr.max()
            col.data = narr.ctypes.data_as(c_void_p)
            col.dtype = str(narr.dtype).encode('utf8')
            #print("\"{}\":\t{}\ttype={},data={},name={}".format(feat, x_info, col.dtype, col.data, col.name))
        return col

    def __init__(self,X,y,params,features=None,categorical_feature=None,discrete_feature=None,cXcY=False,  **kwargs):
        '''
        :param X:
        :param y:
        :param features:
        :param categorical_feature:
        :param kwargs:
        '''

        if not (isinstance(X, pd.DataFrame) or isinstance(X, np.ndarray)):
            raise NotImplementedError("Mort_Preprocess failed to init @{}".format(X))

        self.nSample,self.nFeature = X.shape[0],X.shape[1]
        self.categorical_feature=categorical_feature
        self.col_X=[]
        if features is None:
            if isinstance(X, pd.DataFrame):
                self.features = X.columns
            else:
                pass
        else:
            self.features = features
        if cXcY:       #v0.2
            for feat in self.features:
                col = self.column_info(feat,X,categorical_feature,discrete_feature)
                if 'representive' in params.__dict__ and feat in params.representive:
                    col.representive = params.representive[feat]
                if col.data is not None:
                    self.col_X.append(col)
            col=self.column_info('target',y,categorical_feature,discrete_feature)
            if col.data is None:
                raise( "Mort_Preprocess: col_Y is NONE!!! " )
            self.col_Y=[col]
            self.cX = (M_COLUMN * len(self.col_X))(*self.col_X)
            self.cY = (M_COLUMN * len(self.col_Y))(*self.col_Y)
        return    #please implement this

    def OrdinalEncode_(X,X_test,features=None):
        encoding_dict = dict()
        if features is None:
            features = X.columns
        for col in features:
            values = X[col].value_counts().index.tolist()
            # create a dictionary of values and corresponding number {value, number}
            dict_values = {value: count for value, count in zip(values, range(1, len(values) + 1))}
            # save the values to encode in the dictionary
            encoding_dict[col] = dict_values
            # replace the values with the corresponding number from the dictionary
            X[col] = X[col].map(lambda x: dict_values.get(x))
            X_test[col] = X_test[col].map(lambda x: dict_values.get(x))
        gc.collect()
        return X,X_test

    def fit(self,):
        """
        需要精心设计诶     Each entry in that list must be either 'numerical' or 'categorical'
        """
        pass  # please implement this

    def transform(self,):
        """
        Each entry in that list must be either 'numerical' or 'categorical'
        """
        pass