# Binary Classification of p300 potencial

Trzy zestawy danych zawierające macierze X (przykłady x cechy), Y - opisująca kategorię przykładów (1- target, 0 - non-target).

-   p300_DS1.mat
-   p300_DS2.mat
-   p300_DS3.mat

Sygnały zostały "zmontowane" za pomocą filtra przestrzennego CSP. Cechami wyliczonymi dla każdego przykładu są wariancje uśrednionych po 8 realizacjach sygnałów w odcinku 150 do 550 ms po bodźcu dla kadego ze źródeł estymowanych przez CSP. Zadaniem projektowym jest zaprojektowanie klasyfikatora, który wyuczony na danych ze zbioru p300_DS1.mat miałby możliwie najlepszą klasyfikację na dwóch pozostałych zbiorach i przedstawienie wyników porównawczych dla różnych klasyfikatorów.

Praca przedstawia wynik użycia kilku typowych klasyfikatorów, oraz parametry oceniające jakość klasyfikacji. Do wykonania analizy zostanie użyty jezyk Python w wersji 3. 


## Przygotowanie danych do analizy 

### Wczytanie odpowiednich bibliotek
``` python 
import scipy.io
import os
import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as py
from sklearn import tree
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
```
### Wczytanie odpowiednich danych 


``` python
train_set = scipy.io.loadmat("p300_DS1.mat")
test_set1 = scipy.io.loadmat("p300_DS2.mat")
test_set2 = scipy.io.loadmat("p300_DS3.mat") 
```
### Podział danych na cechy (X) oraz etykiety (Y)
``` python
def get_X_Y(data):
    X = data["X"].tolist()
    Y = data["Y"].tolist()[0]
    return X, Y
X, Y =  get_X_Y(train_set)
X1, Y1 = get_X_Y(test_set1)
X2, Y2 = get_X_Y(test_set2) 
```

### Przygotowanie funkcji miary oceny działania klasyfikatorów

``` python
def plot_precision_and_recall(precision, recall, threshold):
    py.plot(threshold, precision[:-1], "r-", label="precision", linewidth=5)
    py.plot(threshold, recall[:-1], "b", label="recall", linewidth=5)
    py.xlabel("threshold", fontsize=19)
    py.legend(loc="upper right", fontsize=19)
    py.ylim([0, 1])
    
def plot_precision_vs_recall(precision, recall):
    py.plot(recall, precision, "g--", linewidth=2.5)
    py.ylabel("recall", fontsize=19)
    py.xlabel("precision", fontsize=19)
    py.axis([0, 1.5, 0, 1.5])
    
def plot_roc_curve(false_positive_rate, true_positive_rate, label=None):
    py.plot(false_positive_rate, true_positive_rate, linewidth=2, label=label)
    py.plot([0, 1], [0, 1], 'r', linewidth=4)
    py.axis([0, 1, 0, 1])
    py.xlabel('False Positive Rate (FPR)', fontsize=16)
    py.ylabel('True Positive Rate (TPR)', fontsize=16)
```

