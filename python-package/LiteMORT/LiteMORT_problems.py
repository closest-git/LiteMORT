import gc
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge,ElasticNet
from .compat import (_MortModelBase,_MortClassifierBase,_MortRegressorBase)


class Mort_Problems(_MortModelBase):
    def __init__(self,  **kwargs):
        pass

    def get_params(self, deep=True):
        params = super(_MortModelBase, self).get_params(deep=deep)
        params.update(self._other_params)
        return params

    # minor change to support `**kwargs`
    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
            if hasattr(self, '_' + key):
                setattr(self, '_' + key, value)
            self._other_params[key] = value
        return self

    #  注意 Y_t与y_train不一样
    def OnY(self, y_train, np_type):
        # print(type(y_train))
        if type(y_train) is pd.Series:
            np_target = y_train.values.astype(np_type)
        elif isinstance(y_train, pd.DataFrame):
            np_target = y_train.values.astype(np_type)
        else:
            np_target = y_train.astype(np_type)
        return np_target

    def BeforeFit(self,train_set,eval_set):
        return False,None,None

    def AfterPredict(self, X_, Y_):
        return Y_


    def OnResult(self,result_,pred_leaf=False, pred_contrib=False,raw_score=False):
        return result_

class Mort_BinaryClass(Mort_Problems):
    '''
        or LogisticRegression
    '''
    def __init__(self,params,  **kwargs):
        super(Mort_BinaryClass, self).__init__()
        self._labelOfY=None

    def OnY(self, y_train, np_type):
        if self._labelOfY is None:
            self._labelOfY=LabelEncoder()
            self._labelOfY.fit(y_train)
            transformed_labels = self._labelOfY.transform(y_train)
            self._classes = self._labelOfY.classes_
            self._n_classes = len(self._classes)
            if self._n_classes != 2:
                raise ValueError("The class of Y is {}. Not a binary-classification problem!!!".format(self._n_classes) )
        else:
            transformed_labels = self._labelOfY.transform(y_train)
        return super(Mort_BinaryClass, self).OnY(transformed_labels, np_type)

    """LiteMORT Binary classifier.     https://en.wikipedia.org/wiki/Binary_classification"""
    def OnResult(self,result_,pred_leaf=False, pred_contrib=False,raw_score=False):
        # the predicted probability of 2 class
        result_ = np.vstack((1. - result_, result_)).transpose()
        if raw_score or pred_leaf or pred_contrib:
            return result_
        else:
            class_index = np.argmax(result_, axis=1)
            if self._labelOfY is not None:
                return self._labelOfY.inverse_transform(class_index)
            else:
                return class_index
    pass

class Mort_MultiClass(Mort_Problems, _MortClassifierBase):
    pass

class Mort_Regressor(Mort_Problems, _MortRegressorBase):
    """LiteMORT regressor."""
    def __init__(self,params,  **kwargs):
        super(Mort_Regressor, self).__init__()
        self.alpha = 1
        self.gressor = None
        if params.cascade=="lasso":
            self.gressor = Lasso(alpha=self.alpha, normalize=True)
        #self.gressor = Ridge(alpha=0.05, normalize=True)
        #self.gressor = ElasticNet(alpha=1, l1_ratio=0.5, normalize=False)
        self.mse = 0
        self.alg="None"

    def BeforeFit(self,train_set,eval_set):
        if self.gressor is None:
            return False,None,None

        print(f"====== Mort_Regressor::BeforeFit@{self.gressor} alpha={self.alpha}")
        x_train, y_train = train_set
        self.gressor.fit(x_train, y_train)
        pred = self.gressor.predict(x_train)
        self.mse = np.mean((pred - y_train)**2)
        y_train = y_train - pred

        y_eval = None
        if (eval_set is not None and len(eval_set) > 0):
            X_eval, y_eval = eval_set[0]
            y_pred = self.gressor.predict(X_eval)
            y_eval = y_eval-y_pred
        return True,y_train,[y_eval]

    def AfterPredict(self,X_,Y_):
        if self.gressor is not None:
            y_pred = self.gressor.predict(X_)
            Y_= Y_+y_pred
        return Y_




