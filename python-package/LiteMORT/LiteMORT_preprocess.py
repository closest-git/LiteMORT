import pandas as pd
import gc

class Mort_Preprocess(object):
    def __init__(self,  **kwargs):
        pass    #please implement this

    def LabelEncode(self,X,X_test,features=None):
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

    def ToNumeric(self,):
        """
        Each entry in that list must be either 'numerical' or 'categorical'
        """
        pass  # please implement this