import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

#loading the dataset
data = pd.read_csv("posture_dataset.csv")
print(data["label"].value_counts())

X = data.drop("label", axis = 1)
y = data["label"]

#train model and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42, stratify = y)

clf = RandomForestClassifier(n_estimators=200, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

joblib.dump(clf, "posture_model.pkl")
print("model saved")