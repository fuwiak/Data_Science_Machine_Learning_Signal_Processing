import scipy.io
import os
import numpy as np
from data_sets import train_set, test_set1, test_set2

from sklearn.metrics import accuracy_score
import matplotlib.pyplot as py
from sklearn import tree
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score


train_set = scipy.io.loadmat(os.path.abspath(train_set))
test_set1 = scipy.io.loadmat(os.path.abspath(test_set1))
test_set2 = scipy.io.loadmat(os.path.abspath(test_set2))


def get_X_Y(data):
	X = data["X"].tolist()
	Y = data["Y"].tolist()[0]
	return X, Y

X, Y =  get_X_Y(train_set)
X1, Y1 = get_X_Y(test_set1)
X2, Y2 = get_X_Y(test_set2)

#CL1
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, Y)
y_pred = clf.predict(X1)

print(accuracy_score(Y1, y_pred))
print(precision_score(Y1, y_pred))
print(recall_score(Y1, y_pred))
from sklearn.metrics import precision_recall_curve

precision, recall, threshold = precision_recall_curve(Y1, y_pred)
def plot_precision_and_recall(precision, recall, threshold):
    py.plot(threshold, precision[:-1], "r-", label="precision", linewidth=5)
    py.plot(threshold, recall[:-1], "b", label="recall", linewidth=5)
    py.xlabel("threshold", fontsize=19)
    py.legend(loc="upper right", fontsize=19)
    py.ylim([0, 1])

# py.figure(figsize=(14, 7))
plot_precision_and_recall(precision, recall, threshold)
py.show()

def plot_precision_vs_recall(precision, recall):
    py.plot(recall, precision, "g--", linewidth=2.5)
    py.ylabel("recall", fontsize=19)
    py.xlabel("precision", fontsize=19)
    py.axis([0, 1.5, 0, 1.5])

py.figure(figsize=(14, 7))
plot_precision_vs_recall(precision, recall)
py.show()

from sklearn.metrics import roc_curve
false_positive_rate, true_positive_rate, thresholds = roc_curve(Y1, y_pred)

# plotting them against each other# plotti 
def plot_roc_curve(false_positive_rate, true_positive_rate, label=None):
    py.plot(false_positive_rate, true_positive_rate, linewidth=2, label=label)
    py.plot([0, 1], [0, 1], 'r', linewidth=4)
    py.axis([0, 1, 0, 1])
    py.xlabel('False Positive Rate (FPR)', fontsize=16)
    py.ylabel('True Positive Rate (TPR)', fontsize=16)

py.figure(figsize=(14, 7))
plot_roc_curve(false_positive_rate, true_positive_rate)
py.show()

#CL2
# from sklearn import svm
# clf = svm.SVC()
# clf = clf.fit(X, Y)
# y_pred = clf.predict(X1)

# print(accuracy_score(Y1, y_pred))

#CL3

# from sklearn.naive_bayes import GaussianNB
# clf = GaussianNB()
# clf = clf.fit(X, Y)
# y_pred = clf.predict(X1)
# print(accuracy_score(Y1, y_pred))

#CL4 Multi-layer Perceptron classifier

# from sklearn.neural_network import MLPClassifier
# clf = MLPClassifier(solver='lbfgs', alpha=0.2,
#                     hidden_layer_sizes=(3, 3), random_state=1)

# clf.fit(X, Y)                         
# y_pred = clf.predict(X1)
# print(accuracy_score(Y1, y_pred))

#CL5 Stochastic Gradient Descent
# from sklearn.linear_model import SGDClassifier

# clf = SGDClassifier(loss="hinge", penalty="l1")
# clf.fit(X, Y)
# y_pred = clf.predict(X1)
# print(accuracy_score(Y1, y_pred))

#CL6 Logistic Regression (aka logit, MaxEnt) classifier.
# from sklearn.linear_model import LogisticRegression
# clf = LogisticRegression(tol=1e-10, C=1.5)
# clf.fit(X, Y)
# y_pred = clf.predict(X1)
# print(accuracy_score(Y1, y_pred))

#CL7 KNN Nearest Neighbors¶
# from sklearn.neighbors import NearestNeighbors
# clf = NearestNeighbors(n_neighbors=2, algorithm='ball_tree')
# clf.fit(X, Y)
# y_pred = clf.predict(X1)
# print(accuracy_score(Y1, y_pred))






tn, fp, fn, tp = confusion_matrix(Y, y_pred).ravel()


#ROC

from sklearn.metrics import roc_curve


fpr, tpr, thresholds = roc_curve(Y, y_pred, pos_label=1)

# Print ROC curve
py.plot(fpr,tpr)
py.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
py.xlabel('False Positive Rate')
py.ylabel('True Positive Rate')

py.show()

#AUC
from sklearn.metrics import roc_auc_score

print("AUC = ", roc_auc_score(Y, y_pred))