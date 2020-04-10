import pandas as pd


# ************************************* Visualizations *************************************
# Create interactive charts such as PR Curve / ROC using plotly library
def plotly_chart(x, y, hovertext=None, format_decimals=2, solitary_label=None, title=None, xaxis=None, yaxis=None):
    # import libs
    # import chart_studio.plotly as py  # works well with version '1.0.0'
    import plotly.graph_objs as go # works well with ploty version '4.1.0'
    from plotly.offline import init_notebook_mode, iplot
    init_notebook_mode()

    lw = 2  # line-width

    # Create the chart (hovertext is same length array having other information such as thresholds etc)
    trace1 = go.Scatter(x=x, y=y,
                        hovertext=hovertext,
                        mode='lines',
                        line=dict(color='darkorange', width=lw),
                        name=solitary_label
                        )

    # Specify details such as title and axis titles 
    layout = go.Layout(title=title,
                       xaxis=dict(title=xaxis),
                       yaxis=dict(title=yaxis))
    fig = go.Figure(data=[trace1], layout=layout)

    # Plot the figure
    iplot(fig)


# Create PR Curve using matplotlib
def draw_pr_curve(precision, recall, x_range=1.0):
    # import dependencies
    import matplotlib.pyplot as plt

    plt.step(recall, precision, color='b', alpha=0.2,
             where='post')
    plt.fill_between(recall, precision, alpha=0.2, color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, x_range])
    plt.show()


# Create ROC curve using matplotlib
def draw_roc_curve(fpr, tpr, auc):
    # import dependencies
    import matplotlib.pyplot as plt

    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr, label='Keras (area = {:.3f})'.format(auc))
    # plt.plot(fpr_rf, tpr_rf, label='RF (area = {:.3f})'.format(auc_rf))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.show()


# Return precision and recall for a threshold along with the confusion matrix
def print_conf_matrix(y_test, y_pred, thresh):
    print 'threshold: ' + str(thresh)

    y_pred_bool = (y_pred > thresh)

    conf_matrix = pd.crosstab((pd.Series(y_pred_bool, name='Predicted') > thresh),
                              pd.Series(y_test, name='Actual') == True, dropna=False)
    print '____'
    print conf_matrix

    print '____'
    print 'Precision: ' + str(conf_matrix[1][1] / float(conf_matrix[1][1] + conf_matrix[0][1]))
    print 'Recall: ' + str(conf_matrix[1][1] / float(conf_matrix[1][0] + conf_matrix[1][1]))
    print 'Accuracy: ' + str((conf_matrix[1][1] + conf_matrix[0][0]) / float(conf_matrix.sum().sum()))

    # Get False positives
    false_positive_arr = (y_pred_bool == True) & (y_test == False)
    return false_positive_arr


# ************************************* Data Frame Manipulations *************************************
# Remove table alias from column names of a pandas dataframe
def remove_tbl_alias_cols(df):
    new_cols = []

    for col in df.columns:
        new_col = col[col.rfind('.') + 1:]
        new_cols.append(new_col)

    df.columns = new_cols

    return df


# Remove columns 
def remove_unnamed_cols(df):
    num = 0
    col_list = df.columns[[col[0:7] == 'Unnamed' for col in df.columns]]
    for i in range(len(col_list)):
        col_name = col_list[i]
        df = df.drop(col_name, axis=1)
        num += 1
    print ' Removed Cols: ' + str(num)

    return df


# Clean column names to have only alphabetical/numerical or -,_
def cleanse_col_names(col_names):
    if ((type(col_names)) is not list) and ((type(col_names)) is not pd.core.indexes.base.Index):
        print 'Handled Error: Input not a list'
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

# print cleanse_col_names('yo')

def temp():
    print 'Yo'
