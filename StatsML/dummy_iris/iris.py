# -*- coding: utf-8 -*-


from sklearn.datasets import load_iris

iris = load_iris()

X = iris.data
Y = iris.target

from sklearn.cross_validation import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5)

from sklearn import tree

my_class = tree.DecisionTreeClassifier()
my_class.fit(X_train, Y_train)

predictions = my_class.predict(X_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(Y_test, predictions))
