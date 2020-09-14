import pandas as pd

### Feature Encoding
### ----------------
### ****************

## One-Hot Encoding
## ****************
# Function to convert a column into one-hot encoding using get_dummies
def convert_col_one_hot_series(series, drop_first=False, method='get_dummies'):

    # If already a float or int, then simply return
    if series.dtype == float or series.dtype == int:
        return None, 0

    # If boolean, then return after converting the column to int16
    if series.dtype == bool:
        one_hot = series.astype('int16')
    else:
        one_hot = pd.get_dummies(series, prefix=(str(series.name)), drop_first=drop_first)
    return one_hot, 1

def replace_col_with_one_hot(df, col, drop_column=False, drop_first=False, method='get_dummies'):

    # Get one-hot encoding
    one_hot, conversion_signal = convert_col_one_hot_series(df[col])

    print('~~~~~~~~~~')

    if conversion_signal == 0:
        print('Not transforming column ' + str(col) + ' | already numerical')
        return df

    if df[col].dtype == bool:
        df[col] = one_hot
        print(one_hot.name)
    else:
        print(one_hot.columns)
        if drop_column:
            df = df.drop(col,axis = 1)

        # Join the encoded df
        df = df.join(one_hot)

    return df

## Freq and Mean Encoding
## ****************
# Stores the feature manipultation values from training set transformations that need to be placed on Test set
testFeatureTransforms = {}
# Frequency Encoding, Mean encoding of High cardinality categorical features (not the best compared to embedding etc. One hot would be too sparse)
def freqEncode(df, feature):
    encoding = df.groupby(feature).size()
    encoding = encoding/len(df)
    df[feature + 'FreqEnc'] = df[feature].map(encoding)
    testFeatureTransforms[(feature + 'FreqEnc')] = encoding

# Mean encoding
def meanEncode(df, feature, target):
    means = df.groupby(feature)[target].mean()
    df[feature + 'MeanEnc'] = df[feature].map(means)
    testFeatureTransforms[(feature + 'MeanEnc')] = means

### Clean DataFrames
### ----------------
### ****************

# Remove table alias from column names of a pandas dataframe
# **********************************************************
def remove_tbl_alias_cols(df):
    new_cols = []

    for col in df.columns:
        new_col = col[col.rfind('.') + 1:]
        new_cols.append(new_col)

    df.columns = new_cols

    return df


# Remove columns 
# **************
def remove_unnamed_cols(df):
    num = 0
    col_list = df.columns[[col[0:7] == 'Unnamed' for col in df.columns]]
    for i in range(len(col_list)):
        col_name = col_list[i]
        df = df.drop(col_name, axis=1)
        num += 1
    print(' Removed Cols: ' + str(num))

    return df


# Clean column names to have only alphabetical/numerical or -,_
# *************************************************************
def cleanse_col_names(col_names):
    if ((type(col_names)) is not list) and ((type(col_names)) is not pd.core.indexes.base.Index):
        print('Handled Error: Input not a list')
        return 0

    # import dependencies
    import re

    i = 0
    new_names = []
    for elm in col_names:
        y = re.search("[a-z0-9_-]+", elm)
        if y is not None:
            new_names.append(y.group())
        else:
            new_names.append('NoName' + str(i))
            i += 1
    return new_names
