# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 19:51:50 2024

@author: m_pan
"""

from sklearn.datasets import make_swiss_roll
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Generate Swiss Roll dataset
X, t = make_swiss_roll(n_samples=1000, noise=0.2, random_state=42)

# Part B: Plot the resulting Swiss roll dataset
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=t, cmap=plt.cm.Spectral)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
plt.title("Swiss Roll Dataset")
plt.show()


from sklearn.decomposition import KernelPCA

# Apply Kernel PCA with linear, RBF, and sigmoid kernels
kpca_linear = KernelPCA(n_components=2, kernel='linear')
X_kpca_linear = kpca_linear.fit_transform(X)

kpca_rbf = KernelPCA(n_components=2, kernel='rbf', gamma=0.04)
X_kpca_rbf = kpca_rbf.fit_transform(X)

kpca_sigmoid = KernelPCA(n_components=2, kernel='sigmoid', gamma=0.001)
X_kpca_sigmoid = kpca_sigmoid.fit_transform(X)


# Function to plot the results of kPCA
def plot_kpca(X_transformed, title):
    plt.figure(figsize=(8, 6))
    plt.scatter(X_transformed[:, 0], X_transformed[:, 1], c=t, cmap=plt.cm.Spectral)
    plt.title(title)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.show()

# Plotting results for different kernels
plot_kpca(X_kpca_linear, "kPCA with Linear Kernel")
plot_kpca(X_kpca_rbf, "kPCA with RBF Kernel")
plot_kpca(X_kpca_sigmoid, "kPCA with Sigmoid Kernel")

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Create discrete class labels by binning the continuous labels
num_classes = 10  # Define number of classes
t_discrete = np.digitize(t, bins=np.linspace(np.min(t), np.max(t), num_classes)) - 1  # Binning

# Build a pipeline that uses kPCA and logistic regression
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("kpca", KernelPCA(n_components=2)),
    ("log_reg", LogisticRegression(max_iter=1000))  # Increase max_iter for convergence
])

# Define the parameter grid for GridSearchCV
param_grid = [
    {
        "kpca__kernel": ["rbf", "sigmoid"],
        "kpca__gamma": np.logspace(-2, 2, 10)
    }
]

# Apply GridSearchCV to find the best kernel and gamma value
grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring="accuracy")
grid_search.fit(X, t_discrete)  # Use discrete labels for fitting

# Print the best parameters and the best accuracy
print("Best parameters found by GridSearchCV:", grid_search.best_params_)
print("Best classification accuracy:", grid_search.best_score_)






# Extract the results from the grid search
results = grid_search.cv_results_

# Create a figure for plotting
plt.figure(figsize=(10, 6))

# Extract RBF kernel results
mask_rbf = results['param_kpca__kernel'] == 'rbf'
gamma_rbf = results['param_kpca__gamma'][mask_rbf]
mean_test_rbf = results['mean_test_score'][mask_rbf]

# Debugging print statements
print("gamma_rbf:", gamma_rbf)
print("mean_test_rbf:", mean_test_rbf)

# Check if gamma_rbf is empty or not
if len(gamma_rbf) > 0:
    # Convert gamma_rbf to a numpy array and ensure it is numeric
    gamma_rbf = np.array(gamma_rbf, dtype=float)
    plt.plot(np.log10(gamma_rbf), mean_test_rbf, label='RBF Kernel', marker='o')
else:
    print("No RBF gamma values found.")

# Extract Sigmoid kernel results
mask_sigmoid = results['param_kpca__kernel'] == 'sigmoid'
gamma_sigmoid = results['param_kpca__gamma'][mask_sigmoid]
mean_test_sigmoid = results['mean_test_score'][mask_sigmoid]

# Debugging print statements
print("gamma_sigmoid:", gamma_sigmoid)
print("mean_test_sigmoid:", mean_test_sigmoid)

# Check if gamma_sigmoid is empty
if len(gamma_sigmoid) > 0:
    # Convert gamma_sigmoid to a numpy array and ensure it is numeric
    gamma_sigmoid = np.array(gamma_sigmoid, dtype=float)
    plt.plot(np.log10(gamma_sigmoid), mean_test_sigmoid, label='Sigmoid Kernel', marker='o')
else:
    print("No Sigmoid gamma values found.")

plt.xlabel('log10(Gamma)')
plt.ylabel('Classification Accuracy')
plt.title('Grid Search Results for Kernel and Gamma')
plt.legend()
plt.grid()
plt.show()
