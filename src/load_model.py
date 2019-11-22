from sklearn.externals import joblib
def load_model():
	model_xgboost = joblib.load("XGBClassifier_final.pkl")
	model_logistic = joblib.load("Logistic_final.pkl")
	print(model_xgboost)
	print(model_logistic)
	return (model_xgboost,model_logistic)
if __name__ == '__main__':
	model = load_model()