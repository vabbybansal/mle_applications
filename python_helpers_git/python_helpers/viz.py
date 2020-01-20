import pandas as pd

# Return precision and recall for a threshold along with the confusion matrix
def print_conf_matrix(y_test, y_pred, thresh):

    print 'threshold: ' + str(thresh)

    y_pred_bool = (y_pred > thresh)

    conf_matrix = pd.crosstab((pd.Series(y_pred_bool, name='Predicted') >thresh), pd.Series(y_test, name='Actual') == True, dropna = False)
    print '____'
    print conf_matrix

    print '____'
    print 'Precision: ' + str(conf_matrix[1][1] / float(conf_matrix[1][1] + conf_matrix[0][1]))
    print 'Recall: ' + str(conf_matrix[1][1] / float(conf_matrix[1][0] + conf_matrix[1][1]))
    print 'Accuracy: ' + str((conf_matrix[1][1] + conf_matrix[0][0]) / float(conf_matrix.sum().sum()))

    # Get False positives
    false_positive_arr = (y_pred_bool == True) & (y_test == False)
    return false_positive_arr
