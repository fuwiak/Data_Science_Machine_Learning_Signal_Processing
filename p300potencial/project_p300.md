# Binary Classification of p300 potencial

Trzy zestawy danych zawierające macierze X (przykłady x cechy), Y - opisująca kategorię przykładów (1- target, 0 - non-target).

-   p300_DS1.mat
-   p300_DS2.mat
-   p300_DS3.mat

Sygnały zostały "zmontowane" za pomocą filtra przestrzennego CSP. Cechami wyliczonymi dla każdego przykładu są wariancje uśrednionych po 8 realizacjach sygnałów w odcinku 150 do 550 ms po bodźcu dla kadego ze źródeł estymowanych przez CSP. Zadaniem projektowym jest zaprojektowanie klasyfikatora, który wyuczony na danych ze zbioru p300_DS1.mat miałby możliwie najlepszą klasyfikację na dwóch pozostałych zbiorach i przedstawienie wyników porównawczych dla różnych klasyfikatorów.

Praca przedstawia wynik użycia kilku typowych klasyfikatorów, oraz parametry oceniające jakość klasyfikacji. Do wykonania analizy zostanie użyty jezyk Python w wersji 3.

## Analiza

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
### Wczytanie analizowanych zbiorów 

``` python
train_set = scipy.io.loadmat("p300_DS1.mat")
test_set1 = scipy.io.loadmat("p300_DS2.mat")
test_set2 = scipy.io.loadmat("p300_DS3.mat") 
```

