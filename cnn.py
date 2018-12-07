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

from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D
from keras.layers import Activation, Dropout, Flatten, Dense

def create_model(optimizer='rmsprop', init='glorot_uniform'):
	model = Sequential()
	model.add(Conv1D(filters=8,
		kernel_size=2,
		input_shape=(46, 1),
		kernel_initializer=init,
		activation='relu'
		))
	model.add(MaxPooling1D())

	model.add(Conv1D(8, 2, activation='relu'))
	model.add(MaxPooling1D())

	model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
	model.add(Dense(units=8, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(units=1, activation='sigmoid'))

	model.compile(loss='binary_crossentropy',
	          optimizer=optimizer,
	          metrics=['accuracy'])

	
	return model


def grid_search_cv_CNN(model, X_train, Y_train, X_test, Y_test, scorer):
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
		'optimizer': ['rmsprop', 'adam'],
		'init': ['glorot_uniform', 'normal', 'uniform']
		# 'epochs': [50, 100, 150],
		# 'batch_size': [5, 10, 20]
		}
	]
	# 20% test for each of 5 folds, suitable for 653 rows
	clf = GridSearchCV(
		estimator=model,
		param_grid=params,
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
	model = create_model()
	model.summary()

	# Building Phase
	data = import_data(
		"./dataset/crx_clean.data.txt"
		)
	X, Y, X_train, X_test, Y_train, Y_test = split_dataset(data)

	# Expand data dimension for kernel to convolve over
	X_train = np.expand_dims(X_train, axis=2) # (None, 46, 1)
	X_test = np.expand_dims(X_test, axis=2)	# (None, 46, 1)

	# create model
	model = KerasClassifier(build_fn=create_model, verbose=0)
	
	
	# Operational Phase
	scorer = make_scorer(f1_score, pos_label='+')
	print("\n### GRID SEARCH CROSS VAL USING STRATIFIED K FOLD###\n")
	Y_pred_grid_search = grid_search_cv_CNN(model, X_train, Y_train, X_test, Y_test, scorer)
	Y_pred_grid_search = np.squeeze(Y_pred_grid_search)
	print()
	print()
	print(Y_pred_grid_search)
	print()
	print(Y_test)
	print()
	print_scores(Y_test, Y_pred_grid_search)

if __name__=="__main__":
	main()
