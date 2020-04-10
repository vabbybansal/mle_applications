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

# WITH MODEL TUNING SETUP
class XGBoostModel(object):
    def __init__(self, df, labelStr):
        self.label = labelStr

        # segregate X and Y
        self.X = df.loc[:, ~df.columns.isin([labelStr])]
        self.Y = df[labelStr]

    def modelPipeline(self, gridSearch=[]):
        self.X_train, self.X_valid, self.Y_train, self.Y_valid = self.preTrain()
        if len(gridSearch) == 0:
            self.train(self.X_train, self.Y_train)
            self.predictAndMetrics(self.X_valid, self.Y_valid)
        else:
            perf = []
            for config in gridSearch:
                self.train(self.X_train, self.Y_train, config)
                y_pred = self.model.predict_proba(self.X_valid)[:,1]
                fpr, tpr, thresholds_roc = roc_curve(self.Y_valid, y_pred)
                auc_roc = auc(fpr, tpr)
                perf.append(auc_roc)

            return perf

    def preTrain(self, ratio=0.2):
        # split test train
        X_train, X_valid, Y_train, Y_valid = self.split(ratio)
        return X_train, X_valid, Y_train, Y_valid

    def train(self, X_train, Y_train, config=None):

        print '\nLog Training Data Set Shape:' + str(X_train.shape)

        if config == None:
            # Train
            self.model = XGBClassifier(max_depth=10)
            self.model.fit(X_train, Y_train)
        else:
            self.model = XGBClassifier(**config)
            self.model.fit(X_train, Y_train)

    def predictAndMetrics(self, X_valid, Y_valid):
        print '\nLog Test Data Set Shape:' + str(X_valid.shape)

        self.y_pred = self.model.predict_proba(X_valid)[:,1]
        self.precision, self.recall, self.thresholds_pr = precision_recall_curve(Y_valid, self.y_pred)
        self.fpr, self.tpr, self.thresholds_roc = roc_curve(Y_valid, self.y_pred)
        self.auc_roc = auc(self.fpr, self.tpr)

        # Draw PR Curve
        print '\n PR Curve'

        self.plotly_chart(self.recall,
                          self.precision,
                          hovertext=self.thresholds_pr,
                          title='PR Curve',
                          xaxis='Recall',
                          yaxis='Precision'
                          )

        #         self.draw_pr_curve(self.precision, self.recall)

        # Draw ROC
        print '\n ROC Curve'
        #         self.draw_roc_curve(self.fpr, self.tpr, self.auc_roc)

        self.plotly_chart(self.fpr,
                          self.tpr,
                          hovertext=self.thresholds_roc,
                          title='ROC Curve',
                          xaxis='FPR',
                          yaxis='TPR'
                          )


        # Draw Feature Importance curves
        print '\n Feature Importance - Gain'
        plot_importance(self.model, importance_type='gain')
        plt.show()

    def split(self, ratio):
        X_train, X_valid, Y_train, Y_valid, idx1, idx2 = train_test_split(
            self.X,
            self.Y,
            np.arange(self.X.shape[0]),
            stratify=self.Y,
            test_size = ratio,
            random_state = 111
        )
        return X_train, X_valid, Y_train, Y_valid

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

    def draw_pr_curve(self, precision, recall, x_range=1.0):

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

    def draw_roc_curve(self, fpr, tpr, auc):
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
