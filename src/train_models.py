import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix   
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import VotingClassifier

df = pd.read_csv('linear_features.csv')
X = df.drop(['A', '107'], axis=1)
y = pd.DataFrame(df['107'], index=df.index)
# print(X.head())
# print(y.head())
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

def plot_confusion_matrix(conf_matrix, title):
    df_cm = pd.DataFrame(conf_matrix, range(7), range(7))
    #print(df_cm)
    sn.set(font_scale=1.4)
    sn.heatmap(df_cm, annot=True,annot_kws={"size": 16})
    print("\nPlotted the confusion matrix!\n")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Model: '+title)
    plt.show()

def train_log_reg():
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.fit_transform(X_test)
    # X_test_scaled = scaler.fit_transform(X_test)

    model = LogisticRegression()
    print("\nLogistic Regresion: Training...\n")
    model.fit(X_train_scaled, y_train)

    y_train_pred = model.predict(X_train_scaled)
    print("Logistic Regression: Accuracy on training set: %f" %accuracy_score(y_train, y_train_pred))

    y_test_pred = model.predict(X_test_scaled)
    print("\nLogistic Regression: Accuracy on validation set: %f" %accuracy_score(y_test, y_test_pred))

def train_svm():
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.fit_transform(X_test)

    model = SVC(kernel='rbf', probability=True)
    print("\nSupport Vector Machine: Training...\n")
    model.fit(X_train_scaled, y_train)
    print("Done training!\n")

    y_train_pred = model.predict(X_train_scaled)
    print("Support Vector Machine: Accuracy on training set: %f" %accuracy_score(y_train, y_train_pred))

    y_test_pred = model.predict(X_test_scaled)
    print("\nSupport Vector Machine: Accuracy on validation set: %f" %accuracy_score(y_test, y_test_pred))

def train_gnb():
    model = GaussianNB()
    print("\nGaussian Naive Bayes: Training...\n")
    model.fit(X_train, y_train)
    print("Done training!\n")

    y_train_pred = model.predict(X_train)
    print("\nGaussian Naive Bayes: Accuracy on training set: %f" %accuracy_score(y_train, y_train_pred))

    y_test_pred = model.predict(X_test)
    print("\nGaussian Naive Bayes: Accuracy on validation set: %f" %accuracy_score(y_test, y_test_pred))

def train_random_forest():
    model = RandomForestClassifier(n_estimators=500, max_depth=70, max_features='auto'
    , min_samples_leaf=3, min_samples_split=15, n_jobs=-1)
    print("\nRandom Forest: Training...\n")
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    print("Random Forest: Accuracy on training set: %f" %accuracy_score(y_train, y_train_pred))

    y_test_pred = model.predict(X_test)
    print("\nRandom Forest: Accuracy on validation set: %f" %accuracy_score(y_test, y_test_pred))

def train_xgboost():
    model = XGBClassifier()
    print("\nXGBoost: Training...\n")
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    print("XGBoost: Accuracy on training set: %f" %accuracy_score(y_train, y_train_pred))

    y_test_pred = model.predict(X_test)
    print("\nXGBoost: Accuracy on validation set: %f" %accuracy_score(y_test, y_test_pred))

    conf_mat = confusion_matrix(y_test, y_test_pred)
    plot_confusion_matrix(conf_mat, 'xg')

# def train_bag():
#     model = BaggingClassifier(n_estimators=500, n_jobs=-1, max_features=1, max_samples=1)
#     print("\nBagging Classifier: Training...\n")
#     model.fit(X_train, y_train)
#     print("Done training!\n")

#     y_train_pred = model.predict(X_train)
#     print("Bagging Classifier: Accuracy on training set: %f" %log_loss(y_train, y_train_pred))

#     y_val_pred = model.predict(X_val)
#     print("\nBagging Classifier: Accuracy on validation set: %f" %log_loss(y_val, y_val_pred))

#     print("\nPredicting on test set...\n\n\n")
#     y_test = model.predict(X_test)

#     pred_to_csv("bag", y_test)

# def train_voting():
#     scaler = StandardScaler()
#     X_train_scaled = scaler.fit_transform(X_train)
#     X_val_scaled = scaler.fit_transform(X_val)
#     X_test_scaled = scaler.fit_transform(X_test)

#     model1 = LogisticRegression(C=0.01)
#     model2 = tree.DecisionTreeClassifier()
#     model3 = SVC(probability=True)

#     model = VotingClassifier(estimators=[('lr', model1), ('dt', model2), ('svc', model3)], voting='soft', n_jobs=-1)
#     print("\nVoting Classifier: Training...\n")
#     model.fit(X_train_scaled, y_train)
#     print("Done training!\n")

#     y_train_pred = model.predict(X_train_scaled)
#     print(y_train_pred)
#     print(y_train)
#     print("\nVoting Classifier: Accuracy on training set: %f" %log_loss(y_train, y_train_pred))

#     y_val_pred = model.predict(X_val_scaled)
#     print("\nVoting Classifier: Accuracy on validation set: %f" %log_loss(y_val, y_val_pred))

#     print("\nPredicting on test set...\n\n\n")
#     y_test = model.predict(X_test_scaled)

#     pred_to_csv("vote", y_test)


if __name__ == "__main__":

    # train_log_reg()
    # train_svm()
    # train_gnb()
    # train_random_forest()
    train_xgboost()
