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
    
    
def rnorm(xy, y=None):
    if y==None:
        x,y = xy
    else:
        x = xy
    theta = np.linalg.inv(np.dot(x.T, x))
    theta = np.dot(theta, np.dot(x.T,y))
    return theta
    
def agz(x, y, theta_0, alpha, npowt=10, all_thetas=False):
    theta = theta_0
    thetas = []
    for p in range(npowt):
        for j in range(theta.shape[0]):
            delta = np.sum((np.dot(x,theta) - y) * x[j])
            theta[j] = theta[j] - alpha*delta
        if all_thetas: thetas.append(theta.copy())

    if all_thetas:
        return thetas
    else:
        return theta
    
