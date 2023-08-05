from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import plotly.express as px
import plotly.io as io
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
iris = pd.read_csv("D:/folders/ML/CSV/IRIS.csv.xls")
print(iris.head())
print()
print(iris.describe())
print("Target Labels", iris["species"].unique())

fig = px.scatter(iris, x="sepal_width", y="sepal_length", color="species")
fig.show()
x = iris.drop("species", axis=1)
y = iris["species"]
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=0)

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(x_train, y_train)

x_new = np.array([[6, 2.9, 1, 0.2]])
prediction = knn.predict(x_new)
print("Prediction: {}".format(prediction))
