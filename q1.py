# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 18:42:16 2024

@author: m_pan
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, IncrementalPCA
from scipy.io import arff
import pandas as pd

# Part A: Load the .arff file
# Replace 'path_to_your_file.arff' with the actual path to your .arff file
data, meta = arff.loadarff('mnist_784.arff')

# Convert to a pandas DataFrame
df = pd.DataFrame(data)

# Separate features (X) and target (y)
# The last column 'class' is the target, and the first 784 columns are pixel features
X = df.iloc[:, :-1].apply(pd.to_numeric, errors='coerce')  # Ensure all features are numeric
y = df['class'].astype(int)  # Convert the class column to integer

# Print the dataset size
print(f"Dataset size: {X.shape}")

# Function to display digits
def plot_digits(instances, images_per_row=10, fig_size=(10, 10)):
    size = 28  # MNIST images are 28x28 pixels
    images_per_row = min(len(instances), images_per_row)
    images = [instance.reshape(size, size) for instance in instances]
    n_rows = (len(instances) - 1) // images_per_row + 1
    row_images = []
    for row in range(n_rows):
        rimages = images[row * images_per_row: (row + 1) * images_per_row]
        row_images.append(np.concatenate(rimages, axis=1))
    image = np.concatenate(row_images, axis=0)
    plt.figure(figsize=fig_size)
    plt.imshow(image, cmap='binary', interpolation='nearest')
    plt.axis('off')
    plt.show()

# Display first 100 digits from X
plot_digits(X.values[:100])  # Convert DataFrame to NumPy array

# Part B: Apply PCA to retrieve the 1st and 2nd principal components
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
print("Explained variance ratio for 1st and 2nd principal components:", pca.explained_variance_ratio_)

# Part C: Plot the projections of the 1st and 2nd principal components
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', s=1, alpha=0.7)
plt.xlabel('1st Principal Component')
plt.ylabel('2nd Principal Component')
plt.colorbar()
plt.title("Projections on 1st and 2nd Principal Components")
plt.show()

# Part D: Use Incremental PCA to reduce the dimensionality to 154 dimensions
ipca = IncrementalPCA(n_components=154)
X_reduced = ipca.fit_transform(X)

# Part E: Display the original and compressed digits
X_reconstructed = ipca.inverse_transform(X_reduced)

# Plot original and compressed digits side by side
def compare_digits(original, compressed, n_digits=10):
    plt.figure(figsize=(10, 5))
    for i in range(n_digits):
        # Original
        plt.subplot(2, n_digits, i + 1)
        plt.imshow(original[i].reshape(28, 28), cmap='binary', interpolation='nearest')
        plt.axis('off')
        # Compressed
        plt.subplot(2, n_digits, i + 1 + n_digits)
        plt.imshow(compressed[i].reshape(28, 28), cmap='binary', interpolation='nearest')
        plt.axis('off')
    plt.show()

# Compare original and compressed digits (first 10)
compare_digits(X.values[:10], X_reconstructed[:10])
