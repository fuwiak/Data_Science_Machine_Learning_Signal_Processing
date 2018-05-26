# -*- coding: utf-8 -*-
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
import time


def create_data(theta, count=30, std_dev = 3):
    """
    y = theta[0] + theta[1]*x
    """

    X = np.ones((count,2)).reshape(count,2) 
    X[:,1] = np.linspace(0, 10,count)
    epsilon = st.norm.rvs(size = count, loc=0, scale = std_dev).reshape(count,1)
    Y =  np.dot(X,theta) + epsilon 
    return (X,Y)