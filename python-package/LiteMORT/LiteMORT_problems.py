import gc
import numpy as np
import pandas as pd
from ctypes import *
from sklearn import preprocessing

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

    def OnResult(self,result_,pred_leaf=False, pred_contrib=False,raw_score=False):
        return result_

class Mort_BinaryClass(Mort_Problems, _MortClassifierBase):
    def __init__(self,  **kwargs):
        super(Mort_Problems, self).__init__()
        self._labelOfY=None

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
    pass



