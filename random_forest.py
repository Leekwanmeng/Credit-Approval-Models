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
from sklearn.ensemble import RandomForestClassifier

def grid_search_cv_RF(X_train, Y_train, X_test, Y_test, scorer):
	"""
	Grid search cross validation for obtaining best hyperparams
	Uses 5-fold StratifiedKFold split
	
	args
		X_train, Y_train, X_test, Y_test: 2d arrays of train/test data
	returns
		Y_pred: list of test predictions
	"""
	# print(RandomForestClassifier().get_params())
	params = [
		{
		'n_estimators': [3, 5, 10, 50, 100],
		'criterion': ['gini', 'entropy'], 
		'max_depth': [10, 50, None]
		}
	]
	# 20% test for each of 5 folds, suitable for 653 rows
	clf = GridSearchCV(
		RandomForestClassifier(),
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
	
	# Operational Phase
	scorer = make_scorer(f1_score, pos_label='+')
	print("\n### GRID SEARCH CROSS VAL USING STRATIFIED K FOLD###\n")
	Y_pred_grid_search = grid_search_cv_RF(X_train, Y_train, X_test, Y_test, scorer)

	print()
	print()
	print(Y_pred_grid_search)
	print()
	print(Y_test)
	print()
	print_scores(Y_test, Y_pred_grid_search)



if __name__=="__main__":
	main()

