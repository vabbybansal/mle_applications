import pandas as pd

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

    print '~~~~~~~~~~'

    if conversion_signal == 0:
        print 'Not transforming column ' + str(col) + ' | already numerical'
        return df

    if df[col].dtype == bool:
        df[col] = one_hot
        print one_hot.name
    else:
        print one_hot.columns
        if drop_column:
            df = df.drop(col,axis = 1)

        # Join the encoded df
        df = df.join(one_hot)

    return df

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