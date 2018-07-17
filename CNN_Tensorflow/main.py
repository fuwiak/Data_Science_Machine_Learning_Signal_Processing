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

img_num = labeled_images.shape[0]
def preprocess(data, labeled = True):
    '''vector data(784) into image(28*28)'''
    images = []
    if labeled:
        images = data.iloc[:,1:]/255
    else:
        images = data/255
    print(images.shape)
    width = height = np.ceil(np.sqrt(images.shape[1])).astype(np.uint8)
    images = np.reshape(np.array(images), (-1, width, height, 1))
    print(images.shape)
    labels = []
    if labeled:
        labels = data.iloc[:,:1]
        labels_count = np.unique(labels).shape[0]
        labels = encoder.fit_transform(labels)
        
    return images, labels


images, labels = preprocess(labeled_images, labeled = True)

train_images, test_images,train_labels, test_labels = train_test_split(images, labels, train_size=0.7, random_state=0)

X = tf.placeholder(tf.float32, [None, 28, 28, 1])

Y_ = tf.placeholder(tf.float32, [None, 10])

lr = tf.placeholder(tf.float32)

pkeep = tf.placeholder(tf.float32)





