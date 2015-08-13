#!/usr/bin/env python

"""
    correlations.py

    Some code for calculating correlation measures
"""

import numpy as np
import scipy.stats

from scipy.special import psi
from scipy.spatial import cKDTree

from sklearn.metrics import mutual_info_score
from sklearn.neighbors import NearestNeighbors

from collections import defaultdict
from itertools import izip

from time import time

def correlation_ratio(y):
    """ Return the correlation ratio of the data in y.
    y must be a list of arrays, each array containing a class of
    1D observations.
    The correlation ratio is defined as one minus the weighted ratio of 
    inter-class variance divided by total variance.
    """

    class_means = np.array([np.mean(yx) for yx in y])
    class_obs = np.array([len(yx) for yx in y])
    all_obs = np.concatenate(y)

    total_mean = np.average(class_means, weights=class_obs)
    
    numer = np.sum(class_obs*(class_means - total_mean)**2)
    denom = np.sum((all_obs - total_mean)**2)
    
    return 1 - numer/denom

def mutual_information(x, y, bins=1000):
    """ Return the mutual information of the data in x, y
    """
    hist, xedges, yedges = np.histogram2d(x, y, bins=bins)
    #hist /= len(x)

    return mutual_info_score(None, None, contingency=hist)

def mutual_information2(x, y, k=3):
    """ Return mutual information estimate based on Ross B C (PlosOne 2014).
    x contains discrete variables, y contains continuous variables.
    """
    x = np.array(x)
    y = np.array(y)

    discrete_vars = set(x)
        
    cats = dict((d, y[x == d][:,np.newaxis]) for d in discrete_vars)
    Nx = dict((d, cats[d].shape[0]) for d in discrete_vars)

    # one KDTree for each discrete variable
    kdtrees = dict((d, cKDTree(cats[d])) for d in discrete_vars)
    
    # one KDTree for all data points
    full_tree = cKDTree(y[:,np.newaxis])
    N = len(x)
    
    # vectorized KD tree. Calculation of the m_i is the bottle neck
    # no idea how to speed it up further
    I = 0
    for d, data in cats.iteritems():
        ds, __ = kdtrees[d].query(data, k=k+1, p=np.inf)
        ds = ds[:,-1]
                
        mis = np.array([len(full_tree.query_ball_point(dat, r, p=np.inf)) 
            - 1.0
                for dat, r in izip(data, ds)])
        
        I -= psi(mis).sum() + len(data)*psi(Nx[d])

    return I/N + psi(N) + psi(k)

def categorize(x, y=None):
    """ Take an array of 2D observation of the type (n, y) (if y=None)
    or n, y (if y != None)
    and return two lists, [n_1, ..., n_k] and
    [[y_11, y_12,...], ...] containing the categories n and
    the observations belonging to each category.
    """
    if y != None:
        x = izip(x, y)

    observations = defaultdict(list)
    
    for n, y in x:
        observations[n].append(y)

    return observations.keys(), observations.values()

if __name__ == '__main__':
    # test correlation ratio, example taken from Wikipedia
    algebra = [45, 70, 29, 15, 21]
    geometry = [40, 20, 30, 42]
    statistics = [65, 95, 80, 70, 85, 73]
    
    print "Correlation ratio:"
    print correlation_ratio([algebra, geometry, statistics])
    
    # test categorization
    observations = [(1, 1), (2, 3), (2, 4), (1, 5), (3, 0), (3, 4),
            (1, 3)]
    
    print "Categorizer:"
    print observations
    print categorize(observations)

    obs_x, obs_y = zip(*observations)

    print obs_x, obs_y
    print categorize(obs_x, obs_y)
    
    print "Mutual information:"
    print mutual_information(obs_x, obs_y)

    xs = [1, 2, 3, 4, 5, 6, 6, 6, 3, 3, 2, 1, 1, 2, 2, 2, 3, 3, 3,
            4, 5, 5, 5, 6, 4, 4, 4]
    ys = np.random.random(len(xs))

    print "Binned MI:"
    print mutual_information(xs, ys, bins=6)

    print "Nearest Neighbor MI:"
    print mutual_information2(xs, ys, k=3)
