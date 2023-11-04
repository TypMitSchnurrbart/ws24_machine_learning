"""
    Task 2 of Machine Learning
"""

#===== IMPORTS =======================================
import os
import pandas as pd
import numpy as np
from skimage import io, transform
import math
import matplotlib.pyplot as plt

#===== MAIN ==========================================
if __name__ == "__main__":


    # Task 1: Get all pictures of people with more than 70 pics
    pic_data = "lfw_funneled"
    celeb_list = []
    for celeb in os.listdir(pic_data):

        try:
            if len(os.listdir(f"{pic_data}/{celeb}")) >= 70:
                celeb_list.append(celeb)
        except NotADirectoryError:
            continue



    # Read in the pictures
    design_matrix = {
        "label" : [],
        "data" : []
    }
    test_pics = {
        "label" : [],
        "data" : []
    }

    for celeb in celeb_list:
        for index, pic in enumerate(os.listdir(f"{pic_data}/{celeb}")):

                print(celeb)

                # Read in the picture
                image = io.imread(f"{pic_data}/{celeb}/{pic}", as_gray = True)
                image = transform.resize(image, (32, 32), anti_aliasing=True)
                image = image.flatten()

                # Save the test picture
                if index == 0:
                    test_pics["label"].append(celeb)
                    test_pics["data"].append(image)
                else:
                    design_matrix["label"].append(celeb)
                    design_matrix["data"].append(image)

    labels = pd.DataFrame(design_matrix["label"])
    design_matrix = pd.DataFrame(design_matrix["data"])

    test_labels = pd.DataFrame(test_pics["label"])
    test = pd.DataFrame(test_pics["data"])

    print(design_matrix)

    exit()

    # Perform PCA 
    # Center and normalize variance
    for entry in design_matrix:
        variance = design_matrix[entry].var()
        design_matrix[entry] -= design_matrix[entry].mean()
        test[entry] -= design_matrix[entry].mean()
        design_matrix[entry] /= math.sqrt(variance)

    # SVD
    U, D, V = np.linalg.svd(design_matrix)
    eigenvectors = V

    # Display the first 150 eigenvalues
    D = D ** 2 / 1023
    plt.plot(D[:150])
    plt.title("Eigenvalues of Faces")
    plt.xlabel("Order")
    plt.ylabel("Eigenvalues")
    plt.show()

    # Show the first 12 eigenfaces
    eigenfaces = eigenvectors[:12]
    for index, eigenface in enumerate(eigenfaces):
        image = eigenface.reshape((32, 32))

        plt.imshow(image, cmap='gray')  # 'gray' colormap for grayscale images
        plt.title(f"Eigenface {index}")
        plt.axis('off')  # Turn off axis labels
        plt.show()

    # Transform the data to the first 7 eigenfaces
    scores = np.dot(design_matrix, eigenvectors[:7].T)
    test_scores = np.dot(test, eigenvectors[:7].T)

    scores = pd.DataFrame(scores)
    test_scores = pd.DataFrame(test_scores)

    print("Matching...")

    # Go through every trainings image for every test pic
    for real_index, test_pic in test_scores.iterrows():
        min_distance = -1
        acitve_index = 0
        test_pic = test_pic.values

        for index, ref_pic in scores.iterrows():

            ref_pic = ref_pic.values

            distance = np.linalg.norm(test_pic - ref_pic)

            if distance < min_distance or min_distance < 0:
                min_distance = distance
                acitve_index = index

        print(f"Matched: {test_labels[0][real_index]} to {labels[0][acitve_index]}")


   