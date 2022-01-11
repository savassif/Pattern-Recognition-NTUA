from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from itertools import starmap
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import torch
import torchvision
from time import time
from torch import optim
import torch.nn as nn 
from torch.utils.data import Dataset
from sklearn.ensemble import BaggingClassifier, VotingClassifier


def score_per_class(labels,predictions):
	cc = [0]*10
	for i in range (10):
		cc[i] = (labels[labels == i] == predictions[labels == i]).sum () / labels[labels == i].size
	return cc

def show_sample(X, index):
	'''Takes a dataset (e.g. X_train) and imshows the digit at the corresponding index

	Args:
		X (np.ndarray): Digits data (nsamples x nfeatures)
		index (int): index of digit to show
	'''
	digit=X[index,:]
	digit_reshaped=np.reshape(digit,(16,16))
	plt.imshow(digit_reshaped,'gray_r')
	plt.axis('off')
	plt.title("The {} indexed digit".format(index),size=15)
	plt.show()

def plot_digits_samples(X, y):
	'''Takes a dataset and selects one example from each label and plots it in subplots
	Args:
		X (np.ndarray): Digits data (nsamples x nfeatures)
		y (np.ndarray): Labels for dataset (nsamples)
	'''
	digits_checked=np.zeros(10)
	digits_to_plot=np.empty((10,256))
	
	index=np.random.randint(low=0,high=len(X[:,...]))
	while(digits_checked.sum()!=10):
		
		#Pick random index again if array size exceeded
		if index==len(X[:,...]):
			index=np.random.randint(low=0,high=len(X[:,...]))
			continue
			
		if digits_checked[int(y[index])]==0:
			digits_to_plot[int(y[index])]=X[index,:]
			digits_checked[int(y[index])]=1
		
		index+=1
	f, axes = plt.subplots(2,5, figsize=(18, 18))      
	row=0
	for i,digit_to_plot in enumerate(digits_to_plot):
		digit_reshaped = np.reshape(digit_to_plot,(16,16))
		j=i
		if i>=5:
			row=1
			j=i-5
		axes[row,j].imshow(digit_reshaped,cmap='gray_r')
		axes[row,j].set_title(f'Digit {i}',size=20)
		axes[row,j].axis('off')
	plt.show()

def digit_mean_at_pixel(X, y, digit, pixel=(10, 10)):
	'''Calculates the mean for all instances of a specific digit at a pixel location
	Args:
		X (np.ndarray): Digits data (nsamples x nfeatures)
		y (np.ndarray): Labels for dataset (nsamples)
		digit (int): The digit we need to select
		pixels (tuple of ints): The pixels we need to select. 
	Returns:
		(float): The mean value of the digits for the specified pixels
	'''
	digit_indices = []
	for i,label in enumerate(y) :
		if label == digit :
			digit_indices.append(i)
	
	values = [] 
	for index in digit_indices:
		digit = np.reshape(X[index,...],(16,16))
		values.append(digit[pixel])
	return np.asarray(values).mean()



def digit_variance_at_pixel(X, y, digit, pixel=(10, 10)):
	'''Calculates the variance for all instances of a specific digit at a pixel location
	Args:
		X (np.ndarray): Digits data (nsamples x nfeatures)
		y (np.ndarray): Labels for dataset (nsamples)
		digit (int): The digit we need to select
		pixels (tuple of ints): The pixels we need to select
	Returns:
		(float): The variance value of the digits for the specified pixels
	'''

	digit_indices = []
	for i,label in enumerate(y) :
		if label == digit :
			digit_indices.append(i)
	
	values = [] 
	for index in digit_indices:
		digit = np.reshape(X[index,...],(16,16))
		values.append(digit[pixel])
	return np.asarray(values).var()




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


def euclidean_distance(s, m):
	'''Calculates the euclidean distance between a sample s and a mean template m
	Args:
		s (np.ndarray): Sample (nfeatures)
		m (np.ndarray): Template (nfeatures)
	Returns:
		(float) The Euclidean distance between s and m
	'''
	return np.linalg.norm(s - m)


def euclidean_distance_classifier(X, X_mean):
	'''Classifiece based on the euclidean distance between samples in X and template vectors in X_mean
	Args:
		X (np.ndarray): Digits data (nsamples x nfeatures)
		X_mean (np.ndarray): Digits data (n_classes x nfeatures)
	Returns:
		(np.ndarray) predictions (nsamples)
	'''
	#If only 1 sample is given
	if X.shape==(256,):
		tmp=np.empty((1,256))
		tmp[0]=X
		X=tmp
	
	y_pred=np.empty(X.shape[0])
	for i,x in enumerate(X):
		distances = []
		for digit_mean in X_mean:
			distances.append(euclidean_distance(x,digit_mean))

		y_pred[i]=np.argmin(distances)
	return np.asarray(y_pred).astype(int)

class EuclideanDistanceClassifier(BaseEstimator, ClassifierMixin):  
	"""Classify samples based on the distance from the mean feature value"""

	def __init__(self):
		self.X_mean_ = []
		self.estimations = []


	def fit(self, X, y):
		"""
		This should fit classifier. All the "work" should be done here.

		Calculates self.X_mean_ based on the mean 
		feature values in X for each class.

		self.X_mean_ becomes a numpy.ndarray of shape 
		(n_classes, n_features)

		fit always returns self.
		"""
		self.X_mean_ = []
		digit_indices = []
		feature_values = []

		# initialize self.X_mean_ and digit_indices as a list of 10 empty lists 
		for i in range(10):
			self.X_mean_.append([])
			digit_indices.append([])

		# Find all indices in train set of each digit
		for i,label in enumerate(y) :
			digit_indices[label].append(i)

		for k,digit_positions in enumerate(digit_indices):
			for i in range(len(X[0])):
				# gather same feature of all digit samples in order to calculate their mean value
				for index in digit_positions:
					feature_values.append(X[index,i])

				# append mean value of same feature of all samples of digit k in self.X_mean_[k]
				self.X_mean_[k].append(np.asarray(feature_values).mean())
				
				#reset feature_values as empty list for next feature of digit   
				feature_values = []
				
		self.X_mean_ = np.asarray(self.X_mean_)
		return self
			
	def predict(self, X):
		"""
		Make predictions for X based on the
		euclidean distance from self.X_mean_
		"""
		return euclidean_distance_classifier(X,self.X_mean_) 


	def score(self, X, y):
		"""
		Return accuracy score on the predictions
		for X based on ground truth y
		"""
		# Count number of instances that are same both in self.estimations and y and then divide with the number of the instances 
		# to calculate the prediction accuracy
		return accuracy_score(np.asarray(self.predict(X)),y)



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
	occurances = [0]*10
	for label in y :
		occurances[label]+=1
	return np.asarray(list(map(lambda x: x/len(y),occurances)))

def gauss_prob(x,mean,var):
	if var==0:
		var=1e-9
	prob = -( np.square(x-mean)/(2*var)) - 0.5*np.log(2*np.pi*var)
	return prob


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
		self.X_mean_ = np.empty((10,256))
		self.X_var_ = np.empty((10,256))

		for i in range(10):

			self.X_mean_[i]=digit_mean(X,y,i)
			self.X_var_[i]=digit_variance(X,y,i)
		
		#If use_unit_variance is True set variance for all classes to one
		if self.use_unit_variance:
			self.X_var_ = np.ones((10,256))
		
		self.apriori = np.log(calculate_priors(X,y))

		return self


	def predict(self, X):
		"""
		Make predictions for X based on the
		euclidean distance from self.X_mean_
		"""
		self.posterior = np.empty((10,))
		self.predicts = np.empty((X.shape[0],),dtype=np.int64)
		for i,feutures in enumerate(X):
			for c in range(10):
				self.posterior[c] =self.apriori[c] + np.sum([gauss_prob(a,self.X_mean_[c][feaut_num],self.X_var_[c][feaut_num]) for feaut_num,a in enumerate(feutures)]) 

			self.predicts[i] = np.argmax(self.posterior)
		return self.predicts

	def score(self, X, y):
		"""
		Return accuracy score on the predictions
		for X based on ground truth y
		"""
		return accuracy_score(np.asarray(self.predict(X)),y)
	
def evaluate_linear_svm_classifier(X, y, folds=5):
	""" Create an svm with linear kernel and evaluate it using cross-validation
	Calls evaluate_classifier
	"""
	svclassifier = SVC(kernel='linear')
	
	return evaluate_classifier(svclassifier, X, y, folds)


def evaluate_rbf_svm_classifier(X, y, folds=5):
	""" Create an svm with rbf kernel and evaluate it using cross-validation
	Calls evaluate_classifier
	"""
	svclassifier = SVC(kernel='rbf')
	
	return evaluate_classifier(svclassifier, X, y, folds)


def evaluate_poly_svm_classifier(X, y, folds=5):
	""" Create an svm with rbf kernel and evaluate it using cross-validation
	Calls evaluate_classifier
	"""
	svclassifier = SVC(kernel='poly')
	
	return evaluate_classifier(svclassifier, X, y, folds)

def evaluate_sigmoid_svm_classifier(X, y, folds=5):
	""" Create an svm with rbf kernel and evaluate it using cross-validation
	Calls evaluate_classifier
	"""
	svclassifier = SVC(kernel='sigmoid')
	
	return evaluate_classifier(svclassifier, X, y, folds)

def evaluate_knn_classifier(X, y, folds=5):
	""" Create a knn and evaluate it using cross-validation
	Calls evaluate_classifier
	"""
	nbrs = KNeighborsClassifier(n_neighbors=3)
	
	return evaluate_classifier(nbrs,X,y)
    

def evaluate_sklearn_nb_classifier(X, y, folds=5):
	""" Create an sklearn naive bayes classifier and evaluate it using cross-validation
	Calls evaluate_classifier
	"""
	gnb = GaussianNB()
	return evaluate_classifier(gnb, X, y, folds)

	
	
def evaluate_custom_nb_classifier(X, y, folds=5):
	""" Create a custom naive bayes classifier and evaluate it using cross-validation
	Calls evaluate_classifier
	"""
	clf=CustomNBClassifier()
	return evaluate_classifier(clf, X, y, folds)

	
	
def evaluate_euclidean_classifier(X, y, folds=5):
	""" Create a euclidean classifier and evaluate it using cross-validation
	Calls evaluate_classifier
	"""
	clf=EuclideanDistanceClassifier()
	return evaluate_classifier(clf, X, y, folds)

def evaluate_voting_classifier(X, y, folds=5):
	""" Create a voting ensemble classifier and evaluate it using cross-validation
	Calls evaluate_classifier
	"""
	_1_NN = KNeighborsClassifier(n_neighbors=1)
	svm_prob = SVC(probability=True)
	_3_NN = KNeighborsClassifier(n_neighbors=3)

	VC_HARD = VotingClassifier (estimators = [('SVM', svm_prob), ('3_nn', _3_NN),('1_NN', _1_NN)], voting = 'hard',n_jobs=-1)
	VC_SOFT = VotingClassifier (estimators = [('SVM', svm_prob), ('3_nn', _3_NN),('1_NN', _1_NN)], voting = 'soft', n_jobs=-1)

	mean_hard = evaluate_classifier(VC_HARD,X,y)
	mean_soft = evaluate_classifier(VC_SOFT,X,y)

	return [mean_hard, mean_soft]

def evaluate_bagging_classifier(X, y, folds=5):
	""" Create a bagging ensemble classifier and evaluate it using cross-validation
	Calls evaluate_classifier
	"""
	clf = SVC(probability=True)

	Bagging_clf = BaggingClassifier(base_estimator=clf, n_jobs=-1)

	return evaluate_classifier(Bagging_clf,X,y)

class DigitsDataset (Dataset):
	def __init__ (self, X,y):
		self.train_data  = []
		for i in range(len(X)):
			self.train_data.append([X[i], y[i]])

	def __len__ (self):
		return len(self.train_data)

	def __getitem__ (self, idx):	
		return self.train_data[idx]

class NN1Hidden(torch.nn.Module):
	def __init__ (self):
		super(NN1Hidden,self).__init__()
		self.linear1=torch.nn.Linear(256,64)
		self.output=torch.nn.Linear(64,10)        
		self.logsoftmax=torch.nn.LogSoftmax(dim=1)

	def forward(self,x):
		h1_relu=self.linear1(x).clamp(min=0)
		h2_relu = self.output(h1_relu)
		y_pred=self.logsoftmax(h2_relu)
		return y_pred	

class NN3Hidden(torch.nn.Module):
	def __init__ (self):
		super(NN3Hidden,self).__init__()
		self.linear1=torch.nn.Linear(256,4)
		self.linear2=torch.nn.Linear(4,2)
		self.linear3=torch.nn.Linear(2,1)
		self.output=torch.nn.Linear(1,10)        
		self.logsoftmax=torch.nn.LogSoftmax(dim=1)

	def forward(self,x):
		h1_relu=self.linear1(x).clamp(min=0)
		h2_relu=self.linear2(h1_relu).clamp(min=0)
		h3_relu = self.linear3(h2_relu).clamp(min=0)
		output = self.output(h3_relu)
		y_pred=self.logsoftmax(output)
		return y_pred

class Mynn(torch.nn.Module):
    def __init__ (self):
        super(Mynn,self).__init__()
        neuros_per_layer=[256,64,64,10]
        self.hidden1=torch.nn.Linear(neuros_per_layer[0],neuros_per_layer[1])
        self.hidden2=torch.nn.Linear(neuros_per_layer[1],neuros_per_layer[2])
        self.output=torch.nn.Linear(neuros_per_layer[2],neuros_per_layer[3])        
        self.logsoftmax=torch.nn.LogSoftmax(dim=1)
    
    def forward(self,x):
        h1_relu=self.hidden1(x).clamp(min=0)
        h2_relu=self.hidden2(h1_relu).clamp(min=0)
        output=self.output(h2_relu)
        y_pred=self.logsoftmax(output)
        
        return y_pred

class PytorchNNModel(BaseEstimator, ClassifierMixin):
	def __init__(self,model=Mynn()):
		# WARNING: Make sure predict returns the expected (nsamples) numpy array not a torch tensor.
		# TODO: initialize model, criterion and optimizer
		self.model = model
		self.criterion =nn.CrossEntropyLoss()
		self.optimizer = optim.SGD(self.model.parameters(), lr=0.003, momentum=0.9)
		

	def fit(self, X, y):
		# TODO: split X, y in train and validation set and wrap in pytorch dataloaders
		

		train_data = DigitsDataset(X,y)		
		# Shuffle the indices
		indices = np.arange(0,len(X))
		np.random.shuffle(indices) # shuffle the indicies
		
		train_loader = torch.utils.data.DataLoader(train_data,
						batch_size=64, shuffle=False, sampler=torch.utils.data.SubsetRandomSampler(indices[:np.int(len(train_data[:np.int(np.ceil(0.8*len(train_data)))]))]))

		# Build the validation loader using the rest 20% of indices
		val_loader = torch.utils.data.DataLoader(train_data,
					shuffle=False, sampler=torch.utils.data.SubsetRandomSampler(indices[-np.int(len(train_data[-(len(train_data)-np.int(np.ceil(0.8*len(train_data)))):])):]))
		time0 = time()
		epochs = 150
		
		for e in range(epochs):
			running_loss = 0
			for images, labels in train_loader:

				images = images.view(images.shape[0], -1).float()

				# Training pass
				self.optimizer.zero_grad()

				output = self.model(images)
				loss = self.criterion(output, labels)

				#This is where the model learns by backpropagating
				loss.backward()

				#And optimizes its weights here
				self.optimizer.step()

				running_loss += loss.item()
			else:
				#Every 10 epochs print Loss and accuracy of validation Set
				if e%10==0:
					correct_count, all_count = 0, 0
					for images,labels in val_loader:
						for i in range(len(labels)):
							img = images[i].view(1, 256).float()
							
							# Turn off gradients to speed up this part
							with torch.no_grad():
								logps = self.model(img)

							true_label = labels.numpy()[i]
							_,tmp=torch.max(logps.data,1)
							correct_count += (tmp== true_label).sum().item()
							all_count += 1
					# print("\nModel Accuracy on validation set=", (correct_count/all_count))
					print("Epoch {} (on Val set) -loss: {} -accuracy:{}".format(e, running_loss/len(train_loader),correct_count/all_count))
		
		print("\nTraining Time (in minutes) =",(time()-time0)/60)
		
		return self

	def predict(self, X):        
		y_pred=[]
		#Wraping X in pytorch dataloader
		test_loader = torch.utils.data.DataLoader(X)
		
		for image in test_loader:
			img = image.view(1, 256).float()
			
			with torch.no_grad():
				logps = self.model(image.view(1, 256).float())
				
			# Output of the network are log-probabilities, need to take exponential for probabilities
			ps = torch.exp(logps)
			probab = list(ps.numpy()[0])
			pred_label = probab.index(max(probab))
			y_pred.append(pred_label)

		return y_pred
			
	def score(self, X, y):
		# Return accuracy score.
		return accuracy_score(np.asarray(self.predict(X)),y)# def evaluate_bagging_classifier(X, y, folds=5):

def evaluate_nn_classifier(X, y, model = Mynn(), folds=5):
	""" Create a pytorch nn classifier and evaluate it using cross-validation
	Calls evaluate_classifier
	"""
	model_nn=PytorchNNModel(model)
	return evaluate_classifier(model_nn, X, y, folds)

