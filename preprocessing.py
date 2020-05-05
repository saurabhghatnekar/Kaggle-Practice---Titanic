import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.externals import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
# n_neighbors=10
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
import cv2
from sklearn import model_selection
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from sklearn.metrics import accuracy_score
import numpy as np
np.random.seed(1)

data = pd.read_csv("train.csv")

print(data.isnull().sum())

data = data.drop(["Age","Cabin", "Embarked", "PassengerId", "Ticket","Name"],axis = 1) #hypothesis - making a statement
print(data.head())

Y = data.iloc[:,0]
X = data.drop(["Survived"], axis=1)
X_sex = pd.get_dummies(X)
print(X.keys())
X = X_sex
# X = X.drop(["Sex"],axis=1)
# X = pd.concat([X, X_sex], axis=1, sort=False)
print(X.shape)
print(Y.shape)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.9)


#
# saved = []
# models = []
# results = []
# names = []
# models.append(('LR',LogisticRegression()))
# models.append(('LDA',LinearDiscriminantAnalysis()))
# models.append(('KNN',KNeighborsClassifier(n_neighbors=5)))
# models.append(('CART',DecisionTreeClassifier()))
# models.append(('NB',GaussianNB()))
# models.append(('SVM',SVC()))
# models.append(('GBC',GradientBoostingClassifier(n_estimators=112)))
# models.append(('SGD',SGDClassifier()))
# # models.append(('MLP',MLPClassifier(hidden_layer_sizes=2048,alpha=0.001,activation='relu',solver='lbfgs')))
# models.append(("KNN",KNeighborsClassifier(n_neighbors=3, weights='uniform', algorithm='kd_tree')))
#
# for name,model in models:
#     scores = cross_val_score(model, X_train, Y_train, cv=5)
#     model.fit(X_train, Y_train)
#     # joblib.dump(model, "rose"+name+".pkl")
#     score = model.score(X_test, Y_test)
#     names.append(name)
#     results.append(score*100)
#     saved.append(model)
#     print(name,score*100, scores)
#
#
# print(names)
# print(results)
#
# i = np.argmax(results)
# print(i)
# print("\n\nmax:",names[i], '  ', results[i])
#
#
# from sklearn.model_selection import GridSearchCV
# from sklearn.model_selection import RandomizedSearchCV
# parameters = {'n_estimators': range(90, 200)}
# gbc = GradientBoostingClassifier()
# clf = RandomizedSearchCV(gbc, parameters)
# clf.fit(X_train, Y_train)
# # print(GridSearchCV(estimator=gbc,
# #              param_grid=parameters))
#
# # print(X)
#
# print(clf.best_score_)
clf =  GradientBoostingClassifier(n_estimators=112)
clf.fit(X_train, Y_train)

from sklearn.externals import joblib
joblib.dump(clf, "GBC_82.pkl")