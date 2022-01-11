from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
import numpy as np
from glob import glob
import os 
from sklearn.model_selection import KFold
from torch import optim
import torch 
import torchvision
import torch.nn as nn 
import time
from sklearn.neural_network import MLPClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score


def evaluate_classifier(clf, X, y, folds=5):
	"""
		Returns the 5-fold accuracy for classifier clf on X and y
		Args:
			clf (sklearn.base.BaseEstimator): classifier
			X (np.ndarray): Digits data (nsamples x nfeatures)
			y (np.ndarray): Labels for dataset (nsamples)
		Returns:
			(float): The 5-fold classification score (accuracy)
			
	"""
	scores = cross_val_score(clf, X, y,cv=KFold(n_splits=5),scoring="accuracy", n_jobs=-1)
	return np.mean(scores)


def calculate_priors(X, y):
	"""Return the a-priori probabilities for every class
	Args:
		X (np.ndarray): Digits data (nsamples x nfeatures)
		y (np.ndarray): Labels for dataset (nsamples)
	Returns:
		(np.ndarray): (n_classes) Prior probabilities for every class
	"""
	occurances = [0]*len(set(y))
	for label in y :
		occurances[label]+=1
	return np.asarray(list(map(lambda x: x/len(y),occurances)))

def gauss_prob(x,mean,var):
	if var==0:
		var=1e-9
	prob = -( np.square(x-mean)/(2*var)) - 0.5*np.log(2*np.pi*var)
	return prob
 
def digit_mean(X, y, digit):
	'''Calculates the mean for all instances of a specific digit
	Args:
		X (np.ndarray): Digits data (nsamples x nfeatures)
		y (np.ndarray): Labels for dataset (nsamples)
		digit (int): The digit we need to select
	Returns:
		(np.ndarray): The mean value of the digits for every pixel
	'''

	digit_indices = []
	mean = []
	feature_values = []
	for i,label in enumerate(y) :
		if label == digit :
			digit_indices.append(i)

	for i in range(len(X[0])):
		# gather same feature of all digit samples in order to calculate their mean value
		for index in digit_indices:
			feature_values.append(X[index,i])

		# save mean value of digit in mean 
		mean.append(np.asarray(feature_values).mean())
		
		#reset feature_values as empty list for next feature of digit   
		feature_values = []

	return np.asarray(mean)
 

def digit_variance(X, y, digit):
	'''Calculates the variance for all instances of a specific digit
	Args:
		X (np.ndarray): Digits data (nsamples x nfeatures)
		y (np.ndarray): Labels for dataset (nsamples)
		digit (int): The digit we need to select
	Returns:
		(np.ndarray): The variance value of the digits for every pixel
	'''
	digit_indices = []
	variance = []
	feature_values = []
	for i,label in enumerate(y) :
		if label == digit :
			digit_indices.append(i)

	for i in range(len(X[0])):
		# gather same feature of digit in order to calculate their mean value
		for index in digit_indices:
			feature_values.append(X[index,i])

		# append mean value of same feature of all digit samples in mean 
		variance.append(np.asarray(feature_values).var())
		
		#reset feature_values as empty list for next feature of digit   
		feature_values = []

	return np.asarray(variance)

class CustomNBClassifier(BaseEstimator, ClassifierMixin):
	"""Custom implementation Naive Bayes classifier"""

	def __init__(self, use_unit_variance=False):
		self.X_mean_ = None
		self.use_unit_variance = use_unit_variance
		self.X_var_= None


	def fit(self, X, y):
		"""
		This should fit classifier. All the "work" should be done here.
		Calculates self.X_mean_ based on the mean
		feature values in X for each class.
		self.X_mean_ becomes a numpy.ndarray of shape
		(n_classes, n_features)
		fit always returns self.
		"""
		self.y=y
		self.X_mean_ = np.empty((len(set(y)),X.shape[1]))
		self.X_var_ = np.empty((len(set(y)),X.shape[1]))

		for i in range(len(set(y))):
			self.X_mean_[i]=digit_mean(X,y,i)
			self.X_var_[i]=digit_variance(X,y,i)
		
		#If use_unit_variance is True set variance for all classes to one
		if self.use_unit_variance:
			self.X_var_ = np.ones((X.shape[0],X.shape[1]))
		self.apriori = np.log(calculate_priors(X,y))

		return self


	def predict(self, X):
		"""
		Make predictions for X based on the
		euclidean distance from self.X_mean_
		"""
		self.posterior = np.empty((len(set(self.y)),))
		self.predicts = np.empty((X.shape[0],),dtype=np.int64)
		for i,feutures in enumerate(X):
			for c in range(len(set(self.y))):
				self.posterior[c] =self.apriori[c] + np.sum([gauss_prob(a,self.X_mean_[c][feaut_num],self.X_var_[c][feaut_num]) for feaut_num,a in enumerate(feutures)]) 

			self.predicts[i] = np.argmax(self.posterior)
		return self.predicts

	def score(self, X, y):
		"""
		Return accuracy score on the predictions
		for X based on ground truth y
		"""
		return accuracy_score(np.asarray(self.predict(X)),y)
	