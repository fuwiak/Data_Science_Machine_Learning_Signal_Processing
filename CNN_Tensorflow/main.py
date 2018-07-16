import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
import tensorflow as tf

import matplotlib.pyplot as plt, matplotlib.image as mpimg
import matplotlib.cm as cm

labeled_imag = pd.read_csv('train.csv')
from sklearn.preprocessing import LabelBinarizer
encoder = LabelBinarizer()
