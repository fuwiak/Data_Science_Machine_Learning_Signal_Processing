# -*- coding: utf-8 -*-
import numpy as np
import pylab as py
from svm_modul import *
 
dane = np.loadtxt('data.txt') # dane zorganizowane są w trzech kolumnach
N_przyk, N_wej = dane.shape 
X = dane[:,0:2] # pierwsze dwie kolumny to wejście
y = dane[:,2] # trzecia kolumna to etykiety klas

rysujDaneGrup(X, y, marker=('or','xb'), xlabel='x0', ylabel='x1',legend_list=('klasa0','klasa1'))
py.show()
 
# trenujemy model
model  = svmTrain(X, y, C=1, kernelFunction = 'gaussianKernel', tol = 1e-3, max_passes = 20,sigma = 0.05) 

rysujDaneGrup(X, y, marker=('or','xb'), xlabel='x0', ylabel='x1',legend_list=('klasa0','klasa1'))
rysujPodzial(model,X)
py.show()
