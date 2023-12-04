"""
    Deep Learning Task 1

    date: 03.12.2023

"""

# ===== IMPORTS =================================
import numpy as np
import matplotlib.pyplot as plt


# ===== MAIN ====================================
if __name__ == "__main__":

    ### Aufgabe 1.1

    ## Create Train Data and Show plot 

    # Create a 200x2 matrix with values in the range [-6, 6] and train_labels
    train_data = np.random.uniform(low=-6, high=6, size=(200, 2))
    train_labels = np.where((train_data[:, 0] >= 0) & (train_data[:, 1] >= 0) | (train_data[:, 0] <= 0) & (train_data[:, 1] <= 0), 1, 0)

    # Plot the data points
    plt.scatter(train_data[:, 0], train_data[:, 1], c=train_labels, cmap='viridis', marker='o', label='Data Points')

    # Set axis train_labels
    plt.axhline(0, color='black', linestyle='--', linewidth=1)
    plt.axvline(0, color='black', linestyle='--', linewidth=1)
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Scatter Plot of Train Data')

    plt.legend()
    plt.show()


    ## Create Test Data and show plot

    # Create a 200x2 matrix with values in the range [-6, 6] and train_labels
    test_data = np.random.uniform(low=-6, high=6, size=(200, 2))
    test_labels = np.where((test_data[:, 0] >= 0) & (test_data[:, 1] >= 0) | (test_data[:, 0] <= 0) & (test_data[:, 1] <= 0), 1, 0)

    # Plot the data points
    plt.scatter(test_data[:, 0], test_data[:, 1], c=test_labels, cmap='viridis', marker='o', label='Data Points')

    # Set axis test_labels
    plt.axhline(0, color='black', linestyle='--', linewidth=1)
    plt.axvline(0, color='black', linestyle='--', linewidth=1)
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Scatter Plot of Test Data')

    plt.legend()
    plt.show()


    ## Neurons with fixed weight
    