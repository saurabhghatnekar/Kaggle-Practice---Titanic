from sklearn.externals import joblib
import pandas as pd
model = joblib.load("GBC_82.pkl")

print(model)

import numpy as np
np.random.seed(1)

data = pd.read_csv("test.csv")

print(data.isnull().sum())
Id = data["PassengerId"]
data = data.drop(["Age","Cabin", "Embarked", "PassengerId", "Ticket","Name"],axis = 1)
print(data.head())
data = data.fillna(data.mean())
# Y = data.iloc[:,0]
# X = data.drop(["Survived"], axis=1)
X = pd.get_dummies(data)

print(X.keys())

p = model.predict(X)

print(list(Id))
print(p)

for i,pp in zip(list(Id),p):
    print(i,pp,sep=",")