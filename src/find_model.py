import pandas as pd
from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
import matplotlib
from sklearn.metrics import accuracy_score
matplotlib.use('Agg')
from matplotlib import pyplot
import numpy as np


def prepare_dataframe(training_csv_file):
	df = pd.read_csv(training_csv_file)
	df = df.drop(['A'],axis = 1)
	return df

def prepare_data(df):
	X = df.drop(['107'], axis = 1)
	Y = df['107']
	return (X,Y)

def tune_hyperparameters_make_model(X,Y):
	model = XGBClassifier()

	n_estimators = [400, 550,850,1000]
	learning_rate = [0.01,0.1,1]
	param_grid = dict(learning_rate=learning_rate, n_estimators=n_estimators)
	kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)
	grid_search = GridSearchCV(model, param_grid, scoring="accuracy", n_jobs=-1, cv=kfold)
	grid_result = grid_search.fit(X, Y)
	# summarize results
	print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
	means = grid_result.cv_results_['mean_test_score']
	stds = grid_result.cv_results_['std_test_score']
	params = grid_result.cv_results_['params']
	for mean, stdev, param in zip(means, stds, params):
		print("%f (%f) with: %r" % (mean, stdev, param))
	# plot results
	scores = np.array(means).reshape(len(learning_rate), len(n_estimators))
	for i, value in enumerate(learning_rate):
	    pyplot.plot(n_estimators, scores[i], label='learning_rate: ' + str(value))
	pyplot.legend()
	pyplot.xlabel('n_estimators')
	pyplot.ylabel('Accuracy')
	pyplot.savefig('n_estimators_vs_learning_rate.png')
	return params


	
def main():
	df = prepare_dataframe("linear_features.csv")
	data = prepare_data(df)
	X = data[0]
	Y = data[1]
	params = tune_hyperparameters_make_model(X,Y)
	print (params)



if __name__ == '__main__':
	main()

