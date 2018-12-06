import numpy as np
import pandas as pd
from utils import import_data
from utils import split_dataset
from utils import seed
from utils import print_scores

from sklearn.metrics import make_scorer
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier


def decision_tree_clf():
	"""
	Initialise and return DecisionTreeClassifier
	"""
	clf_entropy = DecisionTreeClassifier(
		criterion = "entropy", random_state = seed,
		max_depth = 3, min_samples_leaf = 5
		)
	return clf_entropy

def train_using_entropy(X_train, Y_train):
	# Decision tree with entropy
	clf_entropy = decision_tree_clf()

	# Perform training
	clf_entropy.fit(X_train, Y_train)
	return clf_entropy

def prediction(X_test, clf_object):
	"""
	Prediction on test
	args
		X_test: list of test features
		clf_object: Classifier object
	returns
		Y_pred: list of test predictions
	"""
	Y_prediction = clf_object.predict(X_test)
	print("Total Predicted Values: ", len(Y_prediction))
	return Y_prediction





def cv_with_entropy(X, Y):
	"""
	Cross validate using 5-fold StratifiedKFold split
	Auto fit() and predict()

	args
		X: 2d array of all feature attributes
		Y: 2d array of all class attributes
	returns
		result: list of f1_macro score of each fold
	"""
	# Decision tree with entropy
	clf_entropy = decision_tree_clf()

	# Returns score
	result = cross_val_score(
		clf_entropy, X, Y, 
		scoring='f1_macro', 
		cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
		)
	return result



def grid_search_cv_DT(X_train, Y_train, X_test, Y_test, scorer):
	"""
	Grid search cross validation for obtaining best hyperparams
	Uses 5-fold StratifiedKFold split
	
	args
		X_train, Y_train, X_test, Y_test: 2d arrays of train/test data
	returns
		Y_pred: list of test predictions
	"""
	# print(DecisionTreeClassifier().get_params())
	params = [
		{
		'criterion': ['gini', 'entropy'], 
		'max_depth': [3, 5, 7],
		'min_samples_leaf': [1, 3, 5, 7]
		}
	]
	# 33% test for each of 3 folds, suitable for 653 rows
	clf = GridSearchCV(
		DecisionTreeClassifier(),
		params,
		cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=seed),
		scoring=scorer
		)
	clf.fit(X_train, Y_train)

	print("Best parameters set found on dev set: ", clf.best_params_)
	print()
	print("Grid scores on development set: ")
	means = clf.cv_results_['mean_test_score']
	stds = clf.cv_results_['std_test_score']
	for mean, std, params in zip(means, stds, clf.cv_results_['params']):
	    print("%0.3f (+/-%0.03f) for %r"
	          % (mean, std * 2, params))
	print()

	Y_pred = clf.predict(X_test)
	return Y_pred
	



def main():
	# Building Phase
	data = import_data(
		"./dataset/crx_clean.data.txt"
		)
	X, Y, X_train, X_test, Y_train, Y_test = split_dataset(data)
	clf_entropy = train_using_entropy(X_train, Y_train)

	# Operational Phase
	print("\n### SINGLE TRAIN-TEST SPLIT ###\n")
	Y_pred_entropy = prediction(X_test, clf_entropy)
	print_scores(Y_test, Y_pred_entropy)

	print("\n### CROSS VAL USING STRATIFIED K FOLD ###\n")
	fold_scores = cv_with_entropy(X, Y)
	print("Cross Validate: ", fold_scores)
	print("Best F1_score: ", max(fold_scores)*100)

	scorer = make_scorer(f1_score, pos_label='+')
	print("\n### GRID SEARCH CROSS VAL USING STRATIFIED K FOLD###\n")
	Y_pred_grid_search = grid_search_cv_DT(X_train, Y_train, X_test, Y_test, scorer)
	print_scores(Y_test, Y_pred_grid_search)



if __name__=="__main__":
	main()
