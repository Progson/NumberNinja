import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

database = pd.read_csv("data/mnist_test.csv")
X = database.drop(columns=['label'])
Y = database['label']
X_train,X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2)

if(0):
    model = DecisionTreeClassifier()
    model.fit(X_train,Y_train)
    joblib.dump(model,"wycwiczonyModel.joblib")
else:
    model = joblib.load("wycwiczonyModel.joblib")

predictions = model.predict(X_test)
score = accuracy_score(Y_test, predictions)

print(score)

