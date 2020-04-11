from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import FeatureUnion, Pipeline 
import general.scripts.data_manipulation as manip
import pandas as pd
import numpy as np
from datetime import datetime
import re

# Transformers
class BasicDFCleaning(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    # Calculate some parameters
    def fit(self, X, y = None):
        return self
    # Do the transformation
    def transform(self, X, y = None):
        print 'Started DF Cleaning'
        # Check for the data manipulation library
        try:
            manip
        except NameError:
            print 'Manipulation lib not imported | Exiting!'
            return X
        
        # Remove table alias + Remove unnamed cols
        remove_tbl_alias_cols = manip.remove_tbl_alias_cols
        remove_unnamed_cols = manip.remove_unnamed_cols
        X = remove_tbl_alias_cols(X)
        X = remove_unnamed_cols(X)
        return X

class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, feature_set, return_series=False):
        self.feature_set = feature_set
        self.return_series = return_series
    # Calculate some parameters
    def fit(self, X, y = None):
        return self
    # Do the transformation
    def transform(self, X, y = None):
        if self.return_series and len(self.feature_set) == 1:
            return X[self.feature_set[0]]
        else:
            return X[self.feature_set]

class CategoricalCleaning(BaseEstimator, TransformerMixin):
    def __init__(self):
        import re
    # Calculate some parameters
    def fit(self, X, y = None):
        return self
    # Do the transformation
    def transform(self, X, y = None):
        print 'Started Categorical Cleaning'
        categ_columns = X.columns
        for col in categ_columns:
            self.col_name = col
            X[col] = X[col].progress_apply(self.string_normalize)
        return X
    def string_normalize(self, x):
        x = x.replace('-','_')
        x = re.sub('\s+',' ',x)
        x = x.replace(' ', '_')
        x = x.lower()
        # replace missing value with a prefix
        if x=='missing':
            x = x+'_'+self.col_name
        return x
        
class MakeDFFromNumpyPipeline(BaseEstimator, TransformerMixin):
    def __init__(self, column_list):
        self.column_list = column_list
    # Calculate some parameters
    def fit(self, X, y = None):
        return self
    # Do the transformation
    def transform(self, X, y = None):        
        return pd.DataFrame(X, columns=self.column_list)

class Reshape1DTo2D(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    # Calculate some parameters
    def fit(self, X, y = None):
        return self
    # Do the transformation
    def transform(self, X, y = None):   
        if type(X) == pd.core.series.Series:
            X = X.values
        # Currently return nDArray
        return X.reshape((X.shape[0], 1))

class Series_ApplyFn(BaseEstimator, TransformerMixin):
    def __init__(self, fn, args=None):
        self.fn = fn
        self.args = args
    # Calculate some parameters
    def fit(self, X, y = None):
        return self
    # Do the transformation
    def transform(self, X, y = None): 
        print 'Started Lambda Transform'
        # if reshaped to have 2nd dimension, reshape back to one-D
        if len(X.shape) == 2 and X.shape[1] == 1:
            X = X.reshape((X.shape[0],))
        if type(X) == np.ndarray:
            X = pd.Series(X)
        if self.args == None:
            return X.apply(self.fn)
        else: return X.apply(self.fn, args=self.args)

class DF_ApplyLambda(BaseEstimator, TransformerMixin):
    # USAGE
    #     ('compute', transformers.DF_ApplyLambda(
    #                                             lambda x: 
    #                                                      (
    #                                                          datetime.strptime(x['col1'], "%Y-%m-%d")
    #                                                          -
    #                                                          datetime.strptime(x['col2'], "%Y-%m-%d")
    #                                                      ).days
    #         ))
    def __init__(self, lambda_function):
        self.lambda_function = lambda_function
    # Calculate some parameters
    def fit(self, X, y = None):
        return self
    # Do the transformation
    def transform(self, X, y = None): 
        print 'Started DF Lambda Transform'
        return X.apply(self.lambda_function, axis=1)
    
class GroupByApplyLambda(BaseEstimator, TransformerMixin):
    def __init__(self, group_list, lambda_function):
        self.group_list = group_list
        self.lambda_function = lambda_function
    # Calculate some parameters
    def fit(self, X, y = None):
        return self
    # Do the transformation
    def transform(self, X, y = None): 
        print 'Started Group Column Lambda Transform'
        out = X.groupby(self.group_list).progress_apply(self.lambda_function).reset_index().sort_values(self.group_list)
        return out

# depends currently on tqdm
class GroupFeaturesColumnIntoList(BaseEstimator, TransformerMixin):
    def __init__(self, group_list, column_to_list, sort_list=None, ascending=True):
        self.group_list = group_list
        self.column_to_list = column_to_list
        self.sort_list = sort_list
        self.ascending = ascending
    # Calculate some parameters
    def fit(self, X, y = None):
        return self
    # Do the transformation
    def transform(self, X, y = None): 
        print 'Started Group Column to List transform'
        if self.sort_list == None:
            return X.groupby(self.group_list)[self.column_to_list].progress_apply(list).reset_index().sort_values(self.group_list)
        else:
            return X.sort_values(self.sort_list, ascending=self.ascending).groupby(self.group_list)[self.column_to_list].progress_apply(list).reset_index().sort_values(self.group_list)

class GroupByCreateText(BaseEstimator, TransformerMixin):
    def __init__(self, group_list, column_to_text, word_order_columns=None):
        self.group_list = group_list
        self.word_order_columns = word_order_columns
        self.column_to_text = column_to_text
    def textize(self, column):
#         print df
        outString = ''
        for val in column:
            outString += val
            outString += ' '
#         print '2'
        return outString
    # Calculate some parameters
    def fit(self, X, y = None):
        return self
    # Do the transformation
    def transform(self, X, y = None): 
#         print '3'
        print 'Started GroupByCreateText Transform'
        if self.word_order_columns == None:
#             print '4'
            return X.groupby(self.group_list)[self.column_to_text].progress_apply(self.textize).reset_index().sort_values(self.group_list)
        else:
#             print '5'
            return X.sort_values(self.word_order_columns, ascending=self.ascending).groupby(self.group_list)[self.column_to_text].progress_apply(self.textize).reset_index().sort_values(self.group_list)

class GroupBySelectOne(BaseEstimator, TransformerMixin):
    def __init__(self, group_list, column_to_select):
        self.group_list = group_list
        self.column_to_select = column_to_select
    def fit(self, X, y = None):
        return self
    def pick_one(self, column):
        return column.iloc[0]
    # Do the transformation
    def transform(self, X, y = None): 
        print 'Started GroupBySelectOne Transform'
        return X.groupby(self.group_list)[self.column_to_select].progress_apply(self.pick_one).reset_index().sort_values(self.group_list)

class ColumnToSortedDistinctSeries(BaseEstimator, TransformerMixin):
    def __init__(self, column_name):
        self.column_name = column_name
        pass
    def fit(self, X, y = None):
        return self
    # Do the transformation
    def transform(self, X, y = None): 
        print 'Started ColumnToSortedDistinctSeries Transform'
        uniques = pd.Series(X[self.column_name].unique()).sort_values().values
        return uniques.reshape((uniques.shape[0],1))

# Feature Adds:
# 1) Support for sparse output instead of normal matrix
class GroupMultiCategoricalToPivotedColumnsWithExternalValue(BaseEstimator, TransformerMixin):
    # Multiple criteria 1: as in multiple categorical values possible for one grouped entity
    # Multiple criteria 2: also as in same categorical value can come up in multiple times, out of which this code will choose one
    def __init__(self, column_to_group, column_to_pivot, column_with_values, drop_group=True, fill_value=0, choose_from_multiple_values_criterion=min):
        self.column_to_group = column_to_group
        self.column_to_pivot = column_to_pivot
        self.column_with_values = column_with_values
        self.fill_value = fill_value
        self.choose_from_multiple_values_criterion = choose_from_multiple_values_criterion
        self.drop_group = drop_group
    
    def get_feature_names(self):
        return self.column_names_list.tolist()
    
    def fit(self, X, y = None):
#         print 'IN FIT: ' + str(self.column_to_pivot)
        unique_vals = X[self.column_to_pivot].unique()
        self.column_names_list = 'pivot_'+ self.column_to_pivot + '_' + unique_vals
        self.column_names_list.sort()
        return self
    # Do the transformation
    def transform(self, X, y = None): 
        print 'IN TRANSFORM' + str(self.column_to_pivot)
        print 'Started GroupMultiCategoricalToPivotedColumnsWithExternalValue Transform'
        # Group | Also, resolve Multiple criteria 2 as mentioned above. Now each 'value' has one instance per group
        stacked = X.groupby([self.column_to_group] + [self.column_to_pivot] )[self.column_with_values].progress_apply(self.choose_from_multiple_values_criterion)
        unstacked = stacked.unstack()
        # Delete pivot column name
        del unstacked.columns.name
        # Add pivot colum prefix
        unstacked = unstacked.add_prefix('pivot_'+str(self.column_to_pivot) + '_')
        unstacked = unstacked.reset_index().sort_values(self.column_to_group)
        # fillNa
        unstacked = unstacked.fillna(self.fill_value)
        if self.drop_group:
            unstacked = unstacked.drop(columns=[self.column_to_group])
        
        # Check columns
        set_current_columns = set(unstacked.columns)
        list_cols_needed = []
        for col in self.column_names_list:
            if col not in set_current_columns:
                list_cols_needed.append(col)
                
        for col in list_cols_needed:
#             print 'Adding column - ' + col
            unstacked[col] = self.fill_value
        
        
        return unstacked[self.column_names_list]
    
# Test Cases
# 1) Group: check order should be sorted