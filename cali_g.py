import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
url = "https://raw.githubusercontent.com/NikolaMitrevski/California-Housing-Dataset/main/housing.csv"
df = pd.read_csv(url)

# Select first 150 samples
X = df['total_rooms'].head(150).values
y = df['median_house_value'].head(150).values

# Normalize X for better gradient descent convergence
X = (X - X.mean()) / X.std()

# Initialize parameters
m = 0
b = 0
lr = 0.01
epochs = 1000
n = len(X)

# Gradient Descent
for i in range(epochs):
    y_pred = m * X + b

    dm = (-2/n) * np.sum(X * (y - y_pred))
    db = (-2/n) * np.sum(y - y_pred)

    m = m - lr * dm
    b = b - lr * db

# Predictions
y_pred = m * X + b

# Evaluation
mse = np.mean((y - y_pred) ** 2)
ss_res = np.sum((y - y_pred) ** 2)
ss_tot = np.sum((y - y.mean()) ** 2)
r2 = 1 - ss_res / ss_tot

print("Slope:", m)
print("Intercept:", b)
print("MSE:", mse)
print("R²:", r2)

# Plot
plt.scatter(X, y, color="blue", label="Actual data")
plt.plot(X, y_pred, color="red", label="Gradient Descent")
plt.xlabel("Standardized Total Rooms")
plt.ylabel("Median House Value")
plt.title("Linear Regression using Gradient Descent")
plt.legend()
plt.show()