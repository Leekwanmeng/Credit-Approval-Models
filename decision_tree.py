import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


def import_data(url):
	"""
	args
		url: url string of CLEANED csv data
	returns
		credit_data: Dataframe
	"""

	credit_data = pd.read_csv(url, sep=',', header=None)

	print("Dataset length: ", len(credit_data))
	print("Dataset shape: ", credit_data.shape)
	print("Dataset: \n", credit_data.head())

	credit_data = one_hot_encode_category(credit_data)
	print("One-hot Dataset: \n", credit_data.head())
	print(credit_data.info())
	return credit_data


# Note: For category features, 
# use one-hot encoding instead of dictionary encoding
# to convert to numerical values
# see:
# https://datascience.stackexchange.com/questions/5226/strings-as-features-in-decision-tree-random-forest
# https://www.datacamp.com/community/tutorials/categorical-data
def one_hot_encode_category(credit_data):
	"""
	Splits 'category' columns into one-hot columns
	arg, return
		credit_data: Dataframe
	"""
	cat_columns = []
	for i, _ in enumerate(credit_data):
		# dtype == 'object' after ensuring data has been cleaned
		# i.e no 'float' dtypes as 'object' because of '?' values
		if credit_data[i].dtype == 'object':
			cat_columns.append(i)

	# get_dummies() one-hot encodes data
	credit_data = pd.get_dummies(credit_data, columns=cat_columns)
	return credit_data


def split_dataset(data, class_index, seed):
	"""
	Splits dataset into train and test
	args
		data: Dataframe containing all data
		class_index: int index pointing to class attribute
		seed: int for random state
	"""

	# Separate target class from other attributes
	X = data.values[:, 0:class_index-1]
	Y = data.values[:, class_index]

	# Train-test split
	X_train, X_test, Y_train, Y_test = train_test_split(
		X, Y, test_size = 0.2, random_state = seed)

	return X, Y, X_train, X_test, Y_train, Y_test


def train_using_gini(X_train, Y_train):
	# Decision tree with gini
	clf_gini = DecisionTreeClassifier(
		criterion = "gini", random_state = 100,
		max_depth = 3, min_samples_leaf = 5
		)

	# Perform training
	clf_gini.fit(X_train, Y_train)
	return clf_gini


def train_using_entropy(X_train, Y_train):
	# Decision tree with entropy
	clf_entropy = DecisionTreeClassifier(
		criterion = "entropy", random_state = 100,
		max_depth = 3, min_samples_leaf = 5
		)

	# Perform training
	clf_entropy.fit(X_train, Y_train)
	return clf_entropy


def prediction(X_test, clf_object):
	# Prediction on test with Gini index
	Y_prediction = clf_object.predict(X_test)
	print("Predicted values: \n", Y_prediction)
	return Y_prediction


def calculate_accuracy(Y_test, Y_prediction):
	print("Confusion Matrix: \n", confusion_matrix(Y_test, Y_prediction))
	print("Accuracy: ", accuracy_score(Y_test, Y_prediction)*100)
	print("Report: \n", classification_report(Y_test, Y_prediction))


def main():
	# Building Phase
	data = import_data(
		"./dataset/crx_clean.data.txt"
		)
	X, Y, X_train, X_test, Y_train, Y_test = split_dataset(data, 15, 100)	
	clf_gini = train_using_gini(X_train, Y_train)
	clf_entropy = train_using_entropy(X_train, Y_train)

	# Operational Phase
	print("Results using Gini Index: ")
	Y_pred_gini = prediction(X_test, clf_gini)
	calculate_accuracy(Y_test, Y_pred_gini)

	print("Results using Entropy: ")
	Y_pred_entropy = prediction(X_test, clf_entropy)
	calculate_accuracy(Y_test, Y_pred_entropy)


if __name__=="__main__":
	main()
