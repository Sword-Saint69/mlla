import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
url = "https://raw.githubusercontent.com/NikolaMitrevski/California-Housing-Dataset/main/housing.csv"

df = pd.read_csv(url)

X = df['total_rooms']
y = df['median_house_value']


plt.scatter(X, y, color="blue", s=20)

mean_x = np.mean(X)
mean_y = np.mean(y)


SS_xy = np.sum((X - mean_x) * (y - mean_y))
SS_xx = np.sum((X - mean_x) * (X - mean_x))

b1 = SS_xy / SS_xx
b0 = mean_y - b1 * mean_x

print("Slope (b1):", b1)
print("Intercept (b0):", b0)


y_pred = b0 + b1 * X


n = len(y)
mse = np.sum((y - y_pred) ** 2) / n
print("MSE:", mse)

SS_res = np.sum((y - y_pred) ** 2)
SS_tot = np.sum((y - mean_y) ** 2)

r2 = 1 - (SS_res / SS_tot)
print("R²:", r2)


plt.plot(X, y_pred, color="red")

plt.xlabel("Total Rooms")
plt.ylabel("Median House Value")
plt.title("Simple Linear Regression (150 Samples)")
plt.show()