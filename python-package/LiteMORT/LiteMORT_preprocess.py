import pandas as pd
import numpy as np
import gc

class Mort_Preprocess(object):
    nFeature,nSample=0,0
    features = []
    categorical_feature=[]
    train_X,    train_y=None,None
    eval_X,     eval_y = None, None

    def __init__(self,X,y,features=None,categorical_feature=None,  **kwargs):
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
        if features is None:
            if isinstance(X, pd.DataFrame):
                self.features = X.columns
            else:
                pass
        else:
            self.features = features
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