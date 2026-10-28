import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


housing = fetch_california_housing()

X = housing.data[:, housing.feature_names.index('AveRooms')].reshape(-1, 1)
y = housing.target.reshape(-1, 1)

X = X[:150]
y = y[:150]

plt.scatter(X, y, s=20)

model = LinearRegression()
model.fit(X, y)


b1 = model.coef_[0][0]
b0 = model.intercept_[0]

print("Slope (b1):", b1)
print("Intercept (b0):", b0)


y_pred = model.predict(X)


mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

print("MSE:", mse)
print("R²:", r2)

plt.plot(X, y_pred)

plt.xlabel("Average Rooms")
plt.ylabel("Median House Value")
plt.title("Simple Linear Regression (150 Samples)")
plt.show()
