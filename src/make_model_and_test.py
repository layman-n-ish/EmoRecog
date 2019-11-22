import pandas as pd
from numpy import loadtxt
from xgboost import XGBClassifier
import matplotlib
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib
def prepare_dataframe(training_csv_file):
	df = pd.read_csv(training_csv_file)
	df = df.drop(['A'],axis = 1)
	return df

def prepare_data(df):
	X = df.drop(['107'], axis = 1)
	Y = df['107']
	return (X,Y)

def make_final_model(X,Y):
	final_model = XGBClassifier(n_estimators = 400, learning_rate = 0.1)
	final_model.fit(X,Y)
	return final_model

def test_final_model(final_model):
	cols = [0]
	test = pd.read_csv("test.csv")
	test.drop(test.columns[cols],axis = 1,inplace = True)
	test.head()
	y_test = test['107']
	X_test = test.drop(['107'], axis = 1)
	y_pred = final_model.predict(X_test)
	accuracy = accuracy_score(y_test, y_pred)
	print("Accuracy: %.2f%%" % (accuracy * 100.0))

def store_model(final_model):
	joblib.dump(final_model, "XGBClassifier_final.pkl")
def main():
	df = prepare_dataframe("linear_features.csv")
	data = prepare_data(df)
	X = data[0]
	Y = data[1]
	final_model = make_final_model(X, Y)
	test_final_model(final_model)
	store_model(final_model)
if __name__ == '__main__':
	main()