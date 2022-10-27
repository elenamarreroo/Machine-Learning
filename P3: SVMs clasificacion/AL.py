# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 11:26:01 2017
Revised on Thu Feb 15 17:24:27 2018
@author: Jordi
A class defining the active learning class and useful methods for it.
"""

import scipy
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split


class AL:
    """ Active Learning class. """

    def __init__(self):
        self.colors = ['r', 'g', 'b', 'k', 'm']
        self.classifier = None
        self.xlab, self.ylab, self.xunlab, self.yunlab = None, None, None, None
        self.idx, self.acc = None, None
        self.gamma, self.C = 1.0, 100

    def setup(self, X, y, test_size=0.3, labeled_size=9, random_state=None, probability=False):
        """
        Setup training, test, labeled and unlabeled datasets.
        Returns the test dataset, for final validation.
        """
        # Split data in training/test
        xtrain, xtest, ytrain, ytest = \
            train_test_split(X, y, test_size=test_size, random_state=random_state)

        # Split training data in labeled/unlabeled
        self.xunlab, self.xlab, self.yunlab, self.ylab = \
            train_test_split(xtrain, ytrain, test_size=labeled_size,
                             stratify=ytrain, random_state=random_state)

        # Classifier: SVM
        idx = np.random.permutation(xtrain.shape[0])[0:1000]
        sigma = np.mean(scipy.spatial.distance.pdist(xtrain[idx,:]))
        self.gamma = 1 / (2 * sigma * sigma)
        self.classifier = SVC(C=self.C, gamma=self.gamma,
                              decision_function_shape='ovr', probability=probability)
        self.idx = []
        self.acc = []

        return xtest, ytest

    def copy(self):
        """
        Create a copy of itself by creating a new ao object and copying contents.
        """
        copy = AL()
        copy.xlab = self.xlab.copy()
        copy.xunlab = self.xunlab.copy()
        copy.ylab = self.ylab.copy()
        copy.yunlab = self.yunlab.copy()
        copy.gamma = self.gamma
        copy.C = self.C
        copy.classifier = SVC(C=self.classifier.C, gamma=self.classifier.gamma,
                              probability=self.classifier.probability, decision_function_shape='ovr')
        copy.idx = self.idx.copy()
        copy.acc = self.idx.copy()
        return copy

    def updateLabels(self, idx):
        """
        Move selected samples from unlabeled to labeled set.
        """
        self.xlab = np.concatenate((self.xlab, self.xunlab[idx, :]), axis=0)
        self.ylab = np.concatenate((self.ylab, self.yunlab[idx]), axis=0)
        self.xunlab = np.delete(self.xunlab, idx, axis=0)
        self.yunlab = np.delete(self.yunlab, idx, axis=0)
        # Save them
        self.idx.append(idx)

    def fit(self):
        self.classifier.fit(self.xlab, self.ylab)

    def score(self, xtest, ytest):
        """ Compute score on xtest/ytest, appends to self.acc and returns estimated value. """
        acc = self.classifier.score(xtest, ytest)
        self.acc.append(acc)
        return acc

    # Convenient functions to show results graphically
    def plot(self, plot_unlab=False, marker='o', ms=6, num_points=0, mec=None, axes=plt):
        """
        Plot unlabeled and labeled points.
        """
        if plot_unlab:
            axes.plot(self.xunlab[:, 0], self.xunlab[:, 1], '.', ms=2, color='gray')
        xlab = self.xlab[-num_points:, :]
        ylab = self.ylab[-num_points:]
        # for i,c in enumerate(np.unique(ylab)):
        for c in np.unique(ylab):
            axes.plot(xlab[ylab == c, 0], xlab[ylab == c, 1], c=self.colors[int(c)],
                     ls='None', ms=ms, marker=marker, mec=mec, mew=2)
        axes.grid(1)

    def plotdf(self, axes=plt):
        """
        Plot decision function for SVM. Only works for binary problems.
        """
        xtrain = np.concatenate((self.xlab, self.xunlab), axis=0)

        x_min = xtrain[:, 0].min()
        x_max = xtrain[:, 0].max()
        y_min = xtrain[:, 1].min()
        y_max = xtrain[:, 1].max()

        grid_size_x = (x_max - x_min)
        grid_size_y = (y_max - y_min)

        x_min -= grid_size_x * 0.2
        x_max += grid_size_x * 0.2
        y_min -= grid_size_y * 0.2
        y_max += grid_size_y * 0.2

        xx, yy = np.meshgrid(np.arange(x_min, x_max, grid_size_x * .01),
                             np.arange(y_min, y_max, grid_size_y * .01))

        z = self.classifier.predict(np.c_[xx.ravel(), yy.ravel()])
        z = z.reshape(xx.shape)

        axes.contourf(xx, yy, z, alpha=0.2) #, cmap=plt.cm.Accent)  # terrain is nice too
        # More colormaps at http://matplotlib.org/examples/color/colormaps_reference.html

    def makeplots(self, query_points, axes=plt):
        """
        Make plots showing selected samples and decision boundaries.
        """
        self.plotdf(axes=axes)
        self.plot(True, axes=axes)
        self.plot(marker='s', num_points=query_points, mec='k', axes=axes)
