import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Generate some sample data
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
y = np.array([2, 3, 4, 4.5, 5.5])

# Create a linear regression model
linear_model = LinearRegression()
linear_model.fit(X, y)

# Create polynomial features (degree=2)
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

# Create a polynomial regression model
poly_model = LinearRegression()
poly_model.fit(X_poly, y)

# Generate predictions for both models
X_pred = np.arange(1, 6, 0.1).reshape(-1, 1)
linear_predictions = linear_model.predict(X_pred)
poly_predictions = poly_model.predict(poly.transform(X_pred))

# Plot the data, linear regression line, and polynomial regression curve
plt.scatter(X, y, label='Data')
plt.plot(X_pred, linear_predictions, color='blue', label='Linear Regression')
plt.plot(X_pred, poly_predictions, color='red',
         label='Polynomial Regression (degree=2)')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Comparison of Linear and Polynomial Regression')
plt.legend()
plt.show()
