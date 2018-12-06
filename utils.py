import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score

seed = 100

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
	# print("Dataset: \n", credit_data.head())
	
	# Bring class attribute to first column
	cols = credit_data.columns.tolist()
	cols = cols[-1:] + cols[:-1]
	credit_data = credit_data[cols]
	print("Reordered Dataset: \n", credit_data.head())

	credit_data = one_hot_encode_category(credit_data)
	print("One-hot Dataset: \n", credit_data.head())
	# print(credit_data.info())
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
		if credit_data[i].dtype == 'object' and not i==15:
			cat_columns.append(i)


	# get_dummies() one-hot encodes data
	credit_data = pd.get_dummies(credit_data, columns=cat_columns)
	
	return credit_data


def split_dataset(data):
	"""
	Splits dataset into train and test
	args
		data: Dataframe containing all data
		seed: int for random state
	"""

	# Separate target class from other attributes
	X = data.values[:, 1:]
	Y = data.values[:, 0]

	# Train-test split
	X_train, X_test, Y_train, Y_test = train_test_split(
		X, Y, test_size = 0.2, random_state = seed)
	print("Train size: ", len(X_train))
	print("Test size: ", len(X_test))
	print()
	return X, Y, X_train, X_test, Y_train, Y_test


def print_scores(Y_test, Y_prediction):
	print("Detailed classification report:")
	print()
	print("The model is trained on the full development set.")
	print("The scores are computed on the full evaluation set.")
	print()
	print("Confusion Matrix: \n", confusion_matrix(Y_test, Y_prediction))
	# print("Accuracy: ", accuracy_score(Y_test, Y_prediction)*100)
	print("F1_score: ", f1_score(Y_test, Y_prediction, pos_label='+')*100)
	print("Report: \n", classification_report(Y_test, Y_prediction))