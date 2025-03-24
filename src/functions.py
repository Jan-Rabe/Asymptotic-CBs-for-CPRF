# -*- coding: utf-8 -*-
"""
Created on Aug 21  2024

@author: JR
"""

import numpy as np
import sklearn
import itertools
import scipy
from scipy import stats

#Regression functions used in the notebook:

def m_p4_01(X):
    if X.shape[1] != 4:
        raise ValueError("The input array must have a shape of (n_samples, 4)!")
    return 0.1 * (np.sin(X[:, 0] * np.pi * 2) + X[:, 1] + X[:, 2] * X[:, 3])

def m_p3_01(X):
    if X.shape[1] != 3:
        raise ValueError("The input array must have a shape of (n_samples, 3)!")
    return 0.1 * (np.cos(X[:, 0] * np.pi) * (X[:, 0] > 0.5).astype(int) * (1 + X[:, 1] * 0.3) +
            np.log(1 + X[:, 1]) + np.sin(X[:, 2] * 5) * 0.2)

def m_p3_02(X):
    if X.shape[1] != 3:
        raise ValueError("The input array must have a shape of (n_samples, 3)!")
    return 0.1 * (((1 + 0.5 * (X[:, 0] + X[:, 1])) ** 2) * (1 + 0.3 * (X[:, 2] > 0.5).astype(int)))

def m_p3_03(X):
    if X.shape[1] != 3:
        raise ValueError("The input array must have a shape of (n_samples, 3)!")
    return 0.1 * (np.cos(X[:, 0] * np.pi) * (X[:, 0] > 0.5).astype(int) * (1 + X[:, 1] * 0.3) + 
            X[:, 2] * 0.2)

def m_p2_01(X):
    if X.shape[1] != 2:
        raise ValueError("The input array must have a shape of (n_samples, 2)!")
    return 0.1 * (np.sin(X[:, 0] * np.pi * 2) + X[:, 1])

def m_p2_02(X):
    if X.shape[1] != 2:
        raise ValueError("The input array must have a shape of (n_samples, 2)!")
    return 0.5 * (np.sin(X[:, 0] * np.pi * 2) / (2 * np.pi) + X[:, 1])

def test_grid(p,depth=None,n_g=None):
    """Create a test grid of the feature space dependent on k. The grid contains one value in each undividable cell."""
    if n_g is None:
        g=2**depth
    else:
        g=n_g
    xt1=np.arange(0,1,1/g)+np.ones(g)/(2*g)
    prod=list(itertools.product(xt1, repeat=p))
    grid=np.array(prod)
    return grid  

def vol_intersec_2(fin_com_cell,splits_1,splits_2):
    #splits in: number per direction
    if sum(np.minimum(splits_1,splits_2)>fin_com_cell)>0:
        return 0
    return 1/2**(sum(np.maximum(splits_1,splits_2)))

def ehr_splits(cuts, balls, delta, depth):     
    ehr_split = None
    #determine split
    probs=balls/sum(balls)
    num_features=len(balls)
    feature_index=np.random.choice(num_features,p=probs)
    ind=cuts< ((depth/num_features)+delta)
    ind[feature_index]=0
    probs_2=ind/sum(ind)
    if sum(ind)==0:
        fehler_text=f"Something is wrong! cuts: {cuts}, depth: {depth}"
        raise ValueError(fehler_text)
    ball_index=np.random.choice(num_features,p=probs_2)

    new_cuts=np.copy(cuts)
    new_cuts[feature_index]+=1
    new_balls=np.copy(balls)
    new_balls[feature_index]-=1
    new_balls[ball_index]+=1

    if sum(cuts)<depth:
        return ehr_splits(new_cuts,new_balls,delta,depth)
    else:
        return cuts

def sigma_hat_box(x,y,h):
    # number of observations
    n = len(y)
    
    # Calculate distance matrix
    dists = np.linalg.norm(x[:, np.newaxis, :] - x[np.newaxis, :, :], axis=2)
    
    # calculate weights with uniform density
    weights = stats.uniform.pdf(dists, loc=-h,scale=2*h)
    # diagonale to zero
    np.fill_diagonal(weights, 0)
    
    # calculate squares
    squares = (y[:, np.newaxis] - y[np.newaxis, :])**2 / 2
    
    # get sum of weighted squares and sum of weights
    s_weighted_squares = np.sum(weights * squares)
    s_weights = np.sum(weights)
    
    # calculate sigma_hat
    sigma_hat = s_weighted_squares / s_weights
    return sigma_hat

def sigma_hat_gauss(x,y,h):
    # Suppose, y is a 1D-Array and x is a 2D-Array
    n = len(y)
    
    # Calculate distance matrix
    dists = np.linalg.norm(x[:, np.newaxis, :] - x[np.newaxis, :, :], axis=2)
    
    # calculate weights with normal density
    weights = stats.norm.pdf(dists, scale=h)
    # set diagonale to zero
    np.fill_diagonal(weights, 0)
    
    # calculate squares
    squares = (y[:, np.newaxis] - y[np.newaxis, :])**2 / 2
    
    # get sum of weighted squares and sum of weights
    s_weighted_squares = np.sum(weights * squares)
    s_weights = np.sum(weights)
    
    # calculate sigma_hat
    sigma_hat = s_weighted_squares / s_weights
    return sigma_hat  
