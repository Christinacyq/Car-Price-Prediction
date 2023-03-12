# dataset cleaned at this point
# categorical to decimal
#visualize

# We expect to see data pre-processing in your project such as feature selection (Forward or backward feature selection, dimensionality reduction methods such as PCA, Lasso, LDA, .. ), taking care of missing features in your dataset, ...
#give label?

# Linear regression For example evaluating your predictive model performance using different metrics (take a look at ML Metrics)

import numpy as np
import pandas as pd
import json
import math
import matplotlib
from matplotlib import pyplot as plt

def isNaN(string):
    return string != string
#our data do not have nan 

def visualise(X, C, K=None):# Visualization of clustering. You don't need to change this function   
    fig, ax = plt.subplots()
    ax.scatter(X[:, 0], X[:, 1], c=C,cmap='rainbow')
    if K:
        plt.title('Visualization of K = '+str(K), fontsize=15)
    plt.show()
    pass

def complete_(data): # [1pts]
    
    return data[~np.isnan(data).any(axis=1), :]
    
def incomplete_(data): # [1pts]
  
    condition = ~np.isnan(data[:,len(data[0])-1])&np.isnan(data[:,0:len(data[0])-1]).any(axis=1)
    return data[condition, :]

def unlabeled_(data): # [1pts]

    condition = np.isnan(data[:,len(data[0])-1])&~np.isnan(data[:,0:len(data[0])-1]).any(axis=1)
    return(data[condition, :])
