
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix
)

data = load_breast_cancer()
X = data.data
Y = data.target

# 2. Split dataset (80/20)
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

k_values = [3, 5, 7, 9]
results = []
accuracies = []

for k in k_values:
    knn = KNeighborsClassifier(
        n_neighbors=k,
        p=2
    )
    knn.fit(X_train, Y_train)

    Y_pred = knn.predict(X_test)

    accuracy = accuracy_score(Y_test, Y_pred)
    precision = precision_score(Y_test, Y_pred)
    recall = recall_score(Y_test, Y_pred)
    f1 = f1_score(Y_test, Y_pred)
    cm = confusion_matrix(Y_test, Y_pred)

    accuracies.append(accuracy)

    results.append({
        "K": k,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1
    })

    print(f"\nConfusion Matrix for K = {k}:")
    print(cm)

df_results = pd.DataFrame(results)

print("\nPerformance Comparison:")
print(df_results)
plt.plot(k_values, accuracies, marker="o", color="blue")
plt.xlabel("K (Number of Neighbors)")
plt.ylabel("Accuracy")
plt.title("KNN: Accuracy vs K")
plt.xticks(k_values)
plt.show()
