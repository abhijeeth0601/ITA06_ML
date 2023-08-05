from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

# Importing the dataset
data = pd.read_csv("D:/folders/ML/CSV/CarPrice.csv.xls")

# Data Exploration
data.head()
data.shape
data.isnull().sum()  # Checking if the dataset has NULL Values
data.info()
data.describe()
data.CarName.unique()

# Analysing correlations & using heatmap
print(data.corr())
plt.figure(figsize=(20, 15))
correlations = data.corr()
sns.heatmap(correlations, cmap="coolwarm", annot=True)
plt.show()

# Training a Car Price Prediction Model
predict = "price"
data = data[["symboling", "wheelbase", "carlength",
             "carwidth", "carheight", "curbweight",
             "enginesize", "boreratio", "stroke",
             "compressionratio", "horsepower", "peakrpm",
             "citympg", "highwaympg", "price"]]
x = np.array(data.drop([predict], 1))
y = np.array(data[predict])

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2)

model = DecisionTreeRegressor()
model.fit(xtrain, ytrain)
predictions = model.predict(xtest)

model.score(xtest, predictions)
