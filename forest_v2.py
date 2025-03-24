# -*- coding: utf-8 -*-
"""
Created on Aug 21  2024

@author: JR
"""

import numpy as np
import sklearn
import scipy
import itertools
from abc import ABC, abstractmethod

class HistogramEstimator:
    def __init__(self, n_cells=float('inf')):
        self.n_cells = n_cells
        self.preds=None

    def train(self, X, y):
        p = X.shape[1]
        sums = np.zeros((self.n_cells,) * p)  # Array for sums of the Y
        counts = np.zeros((self.n_cells,) * p) # array for counts of the Y

        for i in range(X.shape[0]):
            indices = np.floor(X[i,:] * self.n_cells).astype(int)  # scale to grid and get Indices
            sums[tuple(indices)] += y[i]
            counts[tuple(indices)] += 1

        self.preds=np.empty((self.n_cells,) * p)
        self.preds=sums/counts
        self.preds[counts==0]=np.nan

    def predict(self, X):
        return np.array([self._predict(sample) for sample in X])

    def _predict(self, sample):
        indices=np.floor(sample * self.n_cells).astype(int)
        return self.preds[tuple(indices)]  
        
    def clear(self):
        self.preds=None

class TreeNode:
    def __init__(self, value=None, left=None, right=None ):
        self.value = value  # The prediction value at the leaf node
        self.left = left    # Left child
        self.right = right  # Right child
        self.feature_index = None  # Index of the feature to split

class RegressionTree(ABC): #Abstrakte Klasse
    def __init__(self, max_depth=float('inf')):
        self.max_depth = max_depth
        self.root = None

    @abstractmethod # Abstrakte Methode
    def fit(self):
        pass

    @abstractmethod # Abstrakte Methode
    def _build_tree(self):
        pass
        
    def predict(self, X):
        return np.array([self._predict(sample, self.root) for sample in X])

    def _predict(self, sample, node):
        new_sample=np.copy(sample)
        if node.left is None and node.right is None:
            return node.value
        if sample[node.feature_index] < 0.5:
            new_sample[node.feature_index]=sample[node.feature_index]*2
            return self._predict(new_sample, node.left)
        else:
            new_sample[node.feature_index]=sample[node.feature_index]*2-1
            return self._predict(new_sample, node.right)

class UniRegressionTree(RegressionTree):

    def fit(self, X, y):
        self.root = self._build_tree(X, y)

    def _build_tree(self, X, y, depth=0):
        num_samples, num_features = X.shape

        # If we reached the maximum depth, return a leaf node
        if depth >= self.max_depth:
            if len(y)==0:
                mean=0
            else:
                mean=np.mean(y)
            return TreeNode(value=mean)

        # Find the split
        rand_split = np.random.randint(0,X.shape[1])
        
        # Split the dataset
        left_indices = X[:, rand_split] < 0.5
        right_indices = ~left_indices

        X_l=X[left_indices]
        X_l[:, rand_split]=X_l[:, rand_split]*2
        X_r=X[right_indices]
        X_r[:, rand_split]=X_r[:, rand_split]*2-1

        left_child = self._build_tree(X_l, y[left_indices], depth + 1)
        right_child = self._build_tree(X_r, y[right_indices], depth + 1)

        # Create a node with the best split
        node = TreeNode()
        node.feature_index = rand_split
        node.left = left_child
        node.right = right_child
        return node    

class EhrRegressionTree(RegressionTree):

    def fit(self, X, y, B, delta):
        num_samples, num_features = X.shape
        cuts=np.zeros(num_features)
        balls=np.ones(num_features)*B
        self.root = self._build_tree(X, y, cuts, balls, delta)

    def _build_tree(self, X, y, cuts, balls, delta, depth=0 ):
        num_samples, num_features = X.shape

        # If we reached the maximum depth, return a leaf node
        if depth >= self.max_depth:
            if len(y)==0:
                mean=0
            else:
                mean=np.mean(y)
            return TreeNode(value=mean)

        # Find the best split
        ehr_split = self._find_ehr_split(cuts, balls, delta, depth)
        if ehr_split is None:
            return TreeNode(value=np.mean(y))
        
        # Split the dataset
        left_indices = X[:, ehr_split['feature_index']] < 0.5
        right_indices = ~left_indices

        X_l=X[left_indices]
        X_l[:, ehr_split['feature_index']]=X_l[:, ehr_split['feature_index']]*2
        X_r=X[right_indices]
        X_r[:, ehr_split['feature_index']]=X_r[:, ehr_split['feature_index']]*2-1

        left_child = self._build_tree(X_l, y[left_indices], ehr_split['cuts'], ehr_split['balls'], delta, depth + 1)
        right_child = self._build_tree(X_r, y[right_indices], ehr_split['cuts'], ehr_split['balls'], delta, depth + 1)

        # Create a node with the best split
        node = TreeNode()
        node.feature_index = ehr_split['feature_index']
        node.left = left_child
        node.right = right_child
        return node

    def _find_ehr_split(self, cuts, balls, delta, depth):   
        
        ehr_split = None
        
        #get split
        probs=balls/sum(balls)
        num_features=len(balls)
        feature_index=np.random.choice(num_features,p=probs)
        ind=cuts< ((depth/num_features)+delta)
        ind[feature_index]=0
        probs_2=ind/sum(ind)
        if sum(ind)==0:
            fehler_text=f"Etwas passt nicht! cuts: {cuts}, depth: {depth}, balls: {balls}, cut index: {feature_index}"
            raise ValueError(fehler_text)
        ball_index=np.random.choice(num_features,p=probs_2)

        new_cuts=np.copy(cuts)
        new_cuts[feature_index]+=1
        new_balls=np.copy(balls)
        new_balls[feature_index]-=1
        new_balls[ball_index]+=1
        
        ehr_split = {
            'feature_index': feature_index,
            'cuts': new_cuts,
            'balls': new_balls
        }
        
        return ehr_split

class centeredCART(RegressionTree):

    def fit(self, X, y, mtry=0.33):
        self.root = self._build_tree(X, y, mtry)

    def _build_tree(self, X, y, mtry, depth=0):
        num_samples, num_features = X.shape

        # If we reached the maximum depth or we don't have enough samples, return a leaf node
        if depth >= self.max_depth or num_samples < 2:
            return TreeNode(value=np.mean(y))

        # Find the cart split
        cart_split = self._find_cart_split(X, y, mtry)
        if cart_split is None:
            return TreeNode(value=np.mean(y))

        # Split the dataset
        left_indices = X[:, cart_split] < 0.5
        right_indices = ~left_indices

        X_l=X[left_indices]
        X_l[:, cart_split]=X_l[:, cart_split]*2
        X_r=X[right_indices]
        X_r[:, cart_split]=X_r[:, cart_split]*2-1

        left_child = self._build_tree(X_l, y[left_indices], mtry, depth + 1)
        right_child = self._build_tree(X_r, y[right_indices], mtry, depth + 1)

        # Create a node with the cart split
        node = TreeNode()
        node.feature_index = cart_split
        node.left = left_child
        node.right = right_child
        return node    

    def _find_cart_split(self, X, y, mtry):
        cart_split = None
        cart_mse = float('inf')

        allowed_features = np.random.choice(X.shape[1], size=int(np.ceil(mtry*X.shape[1])), replace=False)
        for feature_index in allowed_features:
            left_indices = X[:, feature_index] < 0.5
            right_indices = ~left_indices

            if np.any(left_indices) and np.any(right_indices):
                left_mean = np.mean(y[left_indices])
                right_mean = np.mean(y[right_indices])
                mse = (np.mean((y[left_indices] - left_mean) ** 2) +
                       np.mean((y[right_indices] - right_mean) ** 2))

                if mse < cart_mse:
                    cart_mse = mse
                    cart_split = feature_index
        
        return cart_split

# RegressionTreeModel class
class RegressionTreeModel:
    def __init__(self, max_depth=5,tree_type="Uni", B=None, delta=None, mtry=None):
        self.B = B
        self.delta = delta
        self.mtry= mtry
        self.type = tree_type
        if self.type == "Ehr":
            if B is None or delta is None:
                raise ValueError("Parameters B and delta must be provided")
            self.tree = EhrRegressionTree(max_depth)
        elif self.type == "Uni":
            self.tree = UniRegressionTree(max_depth)
        elif self.type == "CART":
            self.tree = centeredCART(max_depth)
        else:
            raise ValueError("tree_type must be 'Ehr', 'Uni' or 'CART'")

    def train(self, X, y):
        """Train the regression tree with the provided X and y."""
        if X is None or y is None:
            raise ValueError("Training data (X and y) must be provided.")
        if self.type == "Ehr":
            self.tree.fit(X, y, self.B, self.delta)
        elif self.type == "Uni":
            self.tree.fit(X, y)
        elif self.type == "CART":
            self.tree.fit(X,y,self.mtry)
        else:
            raise ValueError("tree_type must be 'Ehr', 'Uni' or 'CART'")
        

# RFModel class
class RandomForestModel:
    def __init__(self, n_trees=10, max_depth=5, sample_size_fct=0.7, tree_type="Uni",B=None,delta=None,mtry=0.33):   
        self.type = tree_type
        self.n_trees=n_trees
        self.mtry=mtry
        self.max_depth=max_depth
        self.trees=[]
        self.sample_size_fct=sample_size_fct
        self.B = B
        self.delta = delta
        self.suprema=[]

    def clear(self):
        self.trees=[]

    def train(self, X, y):
        """Train the random forest."""
        
        if self.sample_size_fct>1:
            raise ValueError("Tree sample size must be smaller than one.")
            
        sub_sample_size=int(self.sample_size_fct*len(y))

        if self.type == "Ehr":
            if self.B is None or self.delta is None:
                raise ValueError("Parameters B and delta must be provided")
            for _ in range(self.n_trees):
                # sampling
                indices = np.random.choice(len(y), size=sub_sample_size, replace=False)
                X_sample = X[indices]
                y_sample = y[indices]                
                tree = EhrRegressionTree(max_depth=self.max_depth)
                tree.fit(X_sample, y_sample, self.B, self.delta)
                self.trees.append(tree)
        elif self.type == "Uni":
            for _ in range(self.n_trees):
                # sampling
                indices = np.random.choice(len(y), size=sub_sample_size, replace=False)
                X_sample = X[indices]
                y_sample = y[indices]
                tree = UniRegressionTree(max_depth=self.max_depth)
                tree.fit(X_sample, y_sample)
                self.trees.append(tree)
        elif self.type == "CART":
            for _ in range(self.n_trees):
                # sampling
                indices = np.random.choice(len(y), size=sub_sample_size, replace=False)
                X_sample = X[indices]
                y_sample = y[indices]
                tree = centeredCART(max_depth=self.max_depth)
                tree.fit(X_sample, y_sample,self.mtry)
                self.trees.append(tree)
        else:
            raise ValueError("tree_type must be 'Ehr', 'Uni' or 'CART'")

    def predict(self, X):
        """Predict using the average of all trees."""
        tree_predictions = np.array([tree.predict(X) for tree in self.trees])
        return np.mean(tree_predictions, axis=0)


    def get_sups(self,  p, n_samples, n_forests=100, sig=1):
        X_test=self.test_grid(p)
        for i in range(n_forests):
            Xv = np.random.rand(n_samples*p)
            X = Xv.reshape(n_samples,p)
            e=np.random.normal(0,sig,n_samples)
            self.clear()
            self.train(X,e)  # Train the model
            preds=self.predict(X_test)
            sup=max(abs(preds))
            self.suprema.append(sup)
        return self.suprema

    def test_grid(self,p):
        """Create a test grid of the feature space dependent on k. The grid contains one value in each undividable cell."""
        g=2**self.max_depth
        xt1=np.arange(0,1,1/g)+np.ones(g)/(2*g)
        prod=list(itertools.product(xt1, repeat=p))
        grid=np.array(prod)
        return grid   