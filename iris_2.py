import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import os

iris=load_iris()
X=pd.DataFrame(iris.data, columns=iris.feature_names)
y=iris.target

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

model_v1=LogisticRegression(max_iter=200)
model_v1.fit(X_train,y_train)
y_pred_v1=model_v1.predict(X_test)
acc_v1=accuracy_score(y_test,y_pred_v1)
print(f"Accuracy of model_v1: {acc_v1}")

model_v2=LogisticRegression(solver='liblinear',C=0.5,max_iter=200)
model_v2.fit(X_train,y_train)
y_pred_v2=model_v2.predict(X_test)
acc_v2=accuracy_score(y_test,y_pred_v2)
print(f"Accuracy of model_v2: {acc_v2}")

joblib.dump(model_v1,"models/model_v1.pkl")
joblib.dump(model_v2,"models/model_v2.pkl")

results = pd.DataFrame({
    'Model Version': ['v1', 'v2'],
    'Accuracy': [acc_v1, acc_v2]
})

results.to_csv("results/model_results.csv",index=False)
print("Model results saved to results/model_results.csv")
