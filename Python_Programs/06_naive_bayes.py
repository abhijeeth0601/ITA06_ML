Navie Bayes

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB 
from sklearn.metrics import accuracy score, precision_score, fi_score

lines pd.read_csv('D:/folders/ML/CSV_Files/naivedata.csv')

x=lines.iloc[:, -1] 
y= lines.iloc[:, -1]

x_train, x_test, y_train, y_test train_test_split( x, y, test_size-0.2,random_state=42)
gnb.GaussianNB()
gnb.fit(x_train, y_train) 
y_pred=gnb.predict(x_test)

accuracy = accuracy_score (y_test, y_pred)
print("Accuracy:", accuracy)
#Calculate and display precision
precision = precision_score (y_test, y_pred) print("Precision:", precision)
# Calculate and display F1 score 
fi = fi score(y_test, y_pred)
print("F1 Score:", f1)
