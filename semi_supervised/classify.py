#!/bin/python

def train_classifier(X, y, C_value, p_value):
	"""Train a classifier using the given training data.

	Trains logistic regression on the input data with default parameters.
	"""
	from sklearn.linear_model import LogisticRegression
	cls = LogisticRegression(C = C_value, random_state=0, penalty = p_value, max_iter=10000) # add C
	cls.fit(X, y)
	return cls

def train_classifier_2(X, y):
	"""Train a classifier using the given training data.

	Trains Support Vector Machine on the input data with default parameters.
	"""
	from sklearn.svm import LinearSVC
	from sklearn.calibration import CalibratedClassifierCV
	lin_svc = LinearSVC()
	cls = CalibratedClassifierCV(lin_svc, method = 'isotonic') # for predict_proba
	cls.fit(X, y)
	return cls

def train_classifier_3(X, y):
	"""Train a classifier using the given training data.

	Trains Naive Bayes on the input data with default parameters.
	"""
	from sklearn.naive_bayes import MultinomialNB
	cls = MultinomialNB(alpha = 0.01) 
	cls.fit(X, y)
	return cls



def evaluate(X, yt, cls, name='data'):
	"""Evaluated a classifier on the given labeled data using accuracy."""
	from sklearn import metrics
	yp = cls.predict(X)
	acc = metrics.accuracy_score(yt, yp)
	print("  Accuracy on %s  is: %s" % (name, acc))
	return acc
    
    
