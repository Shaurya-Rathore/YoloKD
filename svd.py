import numpy as np

# Define the matrix A
A = np.array([[3, 2, 2],
              [2, 3, -2]])

# Perform Singular Value Decomposition
U, S, VT = np.linalg.svd(A)

# Print the results
print("Matrix U:")
print(U)
print("\nSingular values (diagonal of Σ):")
print(S)
print("\nMatrix V^T:")
print(VT)

