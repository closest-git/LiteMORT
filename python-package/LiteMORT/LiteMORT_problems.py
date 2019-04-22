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

class Mort_BinaryClass(Mort_Problems, _MortClassifierBase):
    """LiteMORT Binary classifier.     https://en.wikipedia.org/wiki/Binary_classification"""
    pass

class Mort_MultiClass(Mort_Problems, _MortClassifierBase):
    pass

class Mort_Regressor(Mort_Problems, _MortRegressorBase):
    """LiteMORT regressor."""
    pass



