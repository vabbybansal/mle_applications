# Imports
import pandas as pd
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
# import seaborn as sns
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
        print('\n PR Curve')
        viz_generics = VIZ_Generics()

        viz_generics.plotly_chart(recall,
                          precision,
                          hovertext=thresholds_pr,
                          title='PR Curve',
                          xaxis='Recall',
                          yaxis='Precision'
                          )


        # Draw ROC
        print('\n ROC Curve')

        viz_generics.plotly_chart(fpr,
                          tpr,
                          hovertext=thresholds_roc,
                          title='ROC Curve',
                          xaxis='FPR',
                          yaxis='TPR'
                          )
    def draw_pr_curve_plt(self, Y_valid, y_pred, x_range=1.0):
#     (precision, recall, x_range=1.0):

        precision, recall, thresholds_pr = precision_recall_curve(Y_valid, y_pred)
     
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
    def draw_roc_curve_plt(self, Y_valid, y_pred):
#     (fpr, tpr, auc):
        # import dependencies
        import matplotlib.pyplot as plt
        fpr, tpr, thresholds_roc = roc_curve(Y_valid, y_pred)
        auc_roc = auc(fpr, tpr)
        
        plt.figure(1)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.plot(fpr, tpr, label='(area = {:.3f})'.format(auc_roc))
        # plt.plot(fpr_rf, tpr_rf, label='RF (area = {:.3f})'.format(auc_rf))
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC curve')
        plt.legend(loc='best')
        plt.show()

class VIZ_Generics(object):
    
    def __init__(self):
        self.series = {} 
        pass
    
    
    def compare_dist(self, df, numer_var, group_var, xlim, title):
        # group_var assumed to be Bool
        self.series['false'] = (df[(df[numer_var]<xlim) & (df[group_var]==False)])[numer_var].sample((df[(df[numer_var]<xlim) & (df[group_var]==True)])[numer_var].shape[0])
        print(self.series['false'].shape[0])
        plt.hist(
            self.series['false'],
            bins=int(xlim/3), 
            color='b',
            alpha=0.5,
            label='Lost'
        )
        plt.legend(loc='upper right')
        plt.axvline(x=self.series['false'].median(), color='r', alpha=0.2)

        self.series['true'] = (df[(df[numer_var]<xlim) & (df[group_var]==True)])[numer_var]
        print(self.series['true'].shape[0])
        plt.hist(
            self.series['true'],
            bins=int(xlim/3), 
            color='r',
            alpha=0.5,
            label='Won'
        )
        plt.legend(loc='upper right')
        plt.axvline(x=self.series['true'].median(), color='b', alpha=0.2)
        plt.title(title)
        # plt.xlim(0, xlim)
        # plt.ylim(0, 1000000)

        plt.show()
        
    # Needs to run after compare_dist
    def compare_dist_bins(self, title, cuts=15):
        false = self.series['false'].value_counts().reset_index()
        true = self.series['true'].value_counts().reset_index()

        false.columns = ['metric', 'count_false']
        true.columns = ['metric', 'count_true']

        print(true['count_true'].sum() == false['count_false'].sum())
        total = true.merge(false, on='metric').sort_values('metric').reset_index(drop=True)

        # plt.show()
        total['disputed_amount'] = pd.cut(total['metric'], cuts)
        total.groupby('disputed_amount')[['count_true', 'count_false']].sum().plot.bar()
        plt.title(title)
        plt.xticks(rotation=50)
        plt.show()
     
    def pivot_bar_chart(self, df, categorical_col, group_col, some_col, count_min_limit):
        df[categorical_col] = df[categorical_col].astype(str).fillna(".").apply(lambda x: x[0:20])
        xx = df.groupby([categorical_col, group_col]).count()[some_col].reset_index()
        xx.columns = [categorical_col, group_col, 'counts']

        xx = xx.pivot_table(values='counts', columns=group_col, index=categorical_col).fillna(0).reset_index()
        xx['total'] = xx[False] + xx[True]
        xx.sort_values('total', ascending=False)

        xx[xx['total']>count_min_limit][[categorical_col, True, False]].set_index(categorical_col).plot.bar(stacked=False)
        plt.xticks(rotation=75)
        fig = plt.gcf()
        fig.set_size_inches(15, 4)

        plt.show()

    def compare_density(self, df, numer_var, group_var, title, xlim=100):
        df[df[group_var]==True][numer_var].astype(float).plot.kde(label='Won', color='r', alpha=0.5)
        plt.axvline(x=df[df[group_var]==True][numer_var].astype(float).median(), color='r', alpha=0.2)
        plt.legend(loc='upper right')
        df[df[group_var]==False][numer_var].astype(float).plot.kde(label='Lost', color='b', alpha=0.5)
        plt.axvline(x=df[df[group_var]==False][numer_var].astype(float).median(), color='b', alpha=0.2)
        plt.legend(loc='upper right')
        plt.title(title)
        plt.xlim(0, xlim)
        plt.show()

    
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

        print('threshold: ' + str(thresh))

        y_pred_bool = (y_pred > thresh)
        y_pred_binary = pd.Series(y_pred_bool).map({True:1, False:0})

        conf_matrix = pd.crosstab((pd.Series(y_pred_bool, name='Predicted') >thresh), pd.Series(y_test, name='Actual') == True, dropna = False)
        print('____')
        print(conf_matrix)
        print(conf_matrix.shape)

        print('____')
        if conf_matrix.shape[0] == 2 and conf_matrix.shape[1] == 2:
            print('Precision: ' + str(conf_matrix[1][1] / float(conf_matrix[1][1] + conf_matrix[0][1])))
            print('Recall: ' + str(conf_matrix[1][1] / float(conf_matrix[1][0] + conf_matrix[1][1])))
            print('Accuracy: ' + str((conf_matrix[1][1] + conf_matrix[0][0]) / float(conf_matrix.sum().sum())))

            # Get False positives
            print('Returned Binary Preds, False Positive and False negative Array as Tuples')
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
            