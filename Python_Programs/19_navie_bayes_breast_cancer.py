from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

dataset = pd.read_csv("D:/folders/ML/CSV/breastcancer.csv.xls")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

classifier = GaussianNB()
classifier.fit(X_train, y_train)
GaussianNB(priors=None, var_smoothing=1e-09)

y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix : \n", cm)
print("\nAccuracy Score : ", accuracy_score(y_test, y_pred), "\n")
