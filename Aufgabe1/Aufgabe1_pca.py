"""
    Test module
"""

# ===== IMPORTS =======================================
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ===== MAIN ==========================================
if __name__ == "__main__":

    # Import data
    data = pd.read_csv("boston.csv")

    prices = data["TGT"]

    data.drop("TGT", axis=1, inplace=True)
    data.drop("Index", axis=1, inplace=True)

    # Center and normalize variance
    for entry in data:
        variance = data[entry].var()
        data[entry] -= data[entry].mean()
        data[entry] /= math.sqrt(variance)

    # Creating the design matrix
    print(f"Creating design matrix {len(data)}x{len(data.columns)}")
    design_matrix = np.empty((len(data), len(data.columns)))
    design_matrix[:, :len(data.columns)] = data.values

    # Compute svd
    U, D, V = np.linalg.svd(design_matrix)

    # Change D so that it holds the Eigenvalues of the Covmatrix
    # The Change is needed because through svd we only get the deviation in not normalized
    # Those Eigenvalues represent the new variances of the transformed data
    # therefore D is the new Covariance Matrix, because the transformed data
    # is uncorrelated
    # The eigenvalues are ordered by greatness. So biggest eigenvalue is at (0, 0)
    new_cov_matrix = D ** 2 / (len(data) - 1)
    eigenvectors = V

    # Compute defined variance and cumulative
    l = len(new_cov_matrix)
    defined_variance = new_cov_matrix[:l] / np.sum(new_cov_matrix)
    cumulative_variance = np.cumsum(defined_variance)

    output_variance = pd.DataFrame({
        "D": new_cov_matrix,
        "Erklärte Varianz": defined_variance,
        "Kumulierte erklärte Varianz": cumulative_variance
    })

    print(f"{output_variance}\n\n")

    # Transform the data
    scores = np.dot(design_matrix, eigenvectors[:3].T)
    scores = pd.DataFrame(scores, columns=["X", "Y", "Z"])

    # Compute correlation matrix with the original data
    combination = np.hstack((scores, design_matrix))
    multi_cov = np.corrcoef(combination, rowvar=False)
    print(f"Korelationsmatrix mit ursprünglichen Daten:\n{multi_cov}\n\n")

    # Show the Scatter Plot
    scores.drop("Z", axis=1, inplace=True)

    # Split according to price
    scores = scores.join(prices)
    lower = scores[scores["TGT"] < prices.median()]
    higher = scores[scores["TGT"] >= prices.median()]

    lower.drop("TGT", axis=1, inplace=True)
    higher.drop("TGT", axis=1, inplace=True)

    plt.scatter(lower["X"], lower["Y"], label="Lower than Median", color="red")
    plt.scatter(higher["X"], higher["Y"], label="Higher or equal than Median", color="green")
    plt.legend()
    plt.title("Transformed Data Scatter")
    plt.show()
