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

from sklearn.naive_bayes import BernoulliNB, ComplementNB, MultinomialNB

# custom packages
import general.scripts.viz as viz


class Model(object):
    def __init__(self, Xtrain, ytrain, Xtest, ytest, model_type):
        self.Xtrain = Xtrain
        self.ytrain = ytrain
        self.Xtest = Xtest
        self.ytest = ytest
        self.model_type = model_type
        
        # Model type
        self.model_obj = self.get_model_object(self.model_type)
    
    def get_model_object(self, model_type):
        if model_type == 'MultinomialNB':
            return MultinomialNB(alpha=self.alpha)
        elif model_type == 'XGBoost':
            return XGBClassifier(max_depth=self.max_depth)
        else:
            return MultinomialNB(alpha=self.alpha)

    def fit_predict_viz(self):
        self.pre_fit()
        self.fit()
        pred_prob = self.predict()
        self.viz(pred_prob)
        return self.model_obj
    def pre_fit(self):
        pass
    def fit(self):
        self.model_obj.fit(self.Xtrain, self.ytrain)
    def predict(self):
        pass
    def viz(self, pred_prob):
        pass   

class NBModel(Model):
    def __init__(self, Xtrain, ytrain, Xtest, ytest, model_type='MultinomialNB', alpha=.01):
        self.alpha = alpha        
        super(NBModel, self).__init__(Xtrain, ytrain, Xtest, ytest, model_type)
        # Model type
#         self.model_obj = self.getNBType(self.model_type)
    def pre_fit(self):
        print 'Class Distribution: ' + str(pd.Series(self.ytrain).mean())
    def predict(self, one_off_set=None):
        if one_off_set == None:
            return self.model_obj.predict_proba(self.Xtest)[:,1]
        else:
            return self.model_obj.predict_proba(one_off_set)[:,1]
    def viz(self, pred_prob):
        metrics_f = viz.Metrics()
        metrics_f.predictAndMetrics(self.ytest, pred_prob)        
class XGBoostModel(Model):
    def __init__(self, Xtrain, ytrain, Xtest, ytest, model_type='XGBoost', max_depth=10):  
        self.max_depth = max_depth
        super(XGBoostModel, self).__init__(Xtrain, ytrain, Xtest, ytest, model_type)
    def pre_fit(self):
        print 'Class Distribution: ' + str(pd.Series(self.ytrain).mean())
    def predict(self, one_off_set=None):
        if one_off_set is None:
            return self.model_obj.predict_proba(self.Xtest)[:,1]
        else:
            return self.model_obj.predict_proba(one_off_set)[:,1]
    def viz(self, pred_prob):
        metrics_f = viz.Metrics()
        metrics_f.predictAndMetrics(self.ytest, pred_prob)     
    
# NB
# class NBModel(object):
#     def __init__(self, Xtrain, ytrain, Xtest, ytest, model_type='MultinomialNB', alpha=.01):
#         self.Xtrain = Xtrain
#         self.ytrain = ytrain
#         self.Xtest = Xtest
#         self.ytest = ytest
#         self.model_type = model_type
#         self.alpha = alpha        
#         # Model type
#         self.model_obj = self.getNBType(self.model_type)
#     def getNBType(self, model_type_string):
#         if model_type_string == 'MultinomialNB':
#             return MultinomialNB(alpha=self.alpha)
#         else:
#             return MultinomialNB(alpha=self.alpha)
#     def pre_fit(self):
#         print 'Class Distribution: ' + str(pd.Series(self.ytrain).mean())
#     def fit_predict_viz(self):
#         self.pre_fit()
#         self.fit()
#         pred_prob = self.predict()
#         self.viz(pred_prob)
#         return self.model_obj
#     def fit(self):
#         self.model_obj.fit(self.Xtrain, self.ytrain)
#     def predict(self):
#         return self.model_obj.predict_proba(self.Xtest)[:,1]
#     def viz(self, pred_prob):
#         metrics_f = Metrics()
#         metrics_f.predictAndMetrics(self.ytest, pred_prob)        

# WITH MODEL TUNING SETUP
class XGBoostModel_CV(object):
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
