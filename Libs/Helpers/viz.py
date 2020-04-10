# Imports
import pandas as pd
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve
from sklearn.utils.fixes import signature
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from xgboost import XGBClassifier
from xgboost import plot_importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_curve, average_precision_score, roc_curve, auc, f1_score
from sklearn.metrics import confusion_matrix
from xgboost import plot_importance
from sklearn.preprocessing import StandardScaler

class Metrics(object):
    
    def __init__(self,):
        pass
    
    def predictAndMetrics(self, Y_valid, y_pred):

#         y_pred = self.model.predict_proba(X_valid)[:,1]
        precision, recall, thresholds_pr = precision_recall_curve(Y_valid, y_pred)
        fpr, tpr, thresholds_roc = roc_curve(Y_valid, y_pred)
        auc_roc = auc(fpr, tpr)

        # Draw PR Curve
        print '\n PR Curve'
        viz_generics = VIZ_Generics()

        viz_generics.plotly_chart(recall,
                          precision,
                          hovertext=thresholds_pr,
                          title='PR Curve',
                          xaxis='Recall',
                          yaxis='Precision'
                          )


        # Draw ROC
        print '\n ROC Curve'

        viz_generics.plotly_chart(fpr,
                          tpr,
                          hovertext=thresholds_roc,
                          title='ROC Curve',
                          xaxis='FPR',
                          yaxis='TPR'
                          )

class VIZ_Generics(object):
    
    def __init__(self):
        pass
    
    def highest_recall_at_precision(self, y_test, y_pred, reqd_precision):
        temp_thresh = 0.00
        recall = 0.0
        while temp_thresh < 1:
            p_r = self.p_r(y_test, y_pred, temp_thresh)
            if p_r[0] >= reqd_precision and p_r[1] > recall:
                recall = p_r[1]
            temp_thresh += .001
        return recall
    
    def p_r(self, y_test, y_pred, thresh):
        preds_binaries = pd.Series((y_pred > thresh)).map({True:1, False:0})
        truth_for_pred_pos = y_test[(preds_binaries==1)]
        precision = truth_for_pred_pos[truth_for_pred_pos==1].count()/float(truth_for_pred_pos.count())
        recall = truth_for_pred_pos[truth_for_pred_pos==1].count()/float(y_test[y_test==1].count())

        return (precision, recall)

    
    # Return precision and recall for a threshold along with the confusion matrix
    def print_conf_matrix(self, y_test, y_pred, thresh):

        print 'threshold: ' + str(thresh)

        y_pred_bool = (y_pred > thresh)
        y_pred_binary = pd.Series(y_pred_bool).map({True:1, False:0})

        conf_matrix = pd.crosstab((pd.Series(y_pred_bool, name='Predicted') >thresh), pd.Series(y_test, name='Actual') == True, dropna = False)
        print '____'
        print conf_matrix
        print conf_matrix.shape

        print '____'
        if conf_matrix.shape[0] == 2 and conf_matrix.shape[1] == 2:
            print 'Precision: ' + str(conf_matrix[1][1] / float(conf_matrix[1][1] + conf_matrix[0][1]))
            print 'Recall: ' + str(conf_matrix[1][1] / float(conf_matrix[1][0] + conf_matrix[1][1]))
            print 'Accuracy: ' + str((conf_matrix[1][1] + conf_matrix[0][0]) / float(conf_matrix.sum().sum()))

            # Get False positives
            print 'Returned Binary Preds, False Positive and False negative Array as Tuples'
            false_positive_arr = (y_pred_bool == True) & (y_test == False)
            false_negative_arr = (y_pred_bool == False) & (y_test == True)
            return (y_pred_binary, (false_positive_arr, false_negative_arr))

    def plotly_chart(self, x, y, hovertext=None, format_decimals=2, solitary_label=None, title=None, xaxis=None, yaxis=None):
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
