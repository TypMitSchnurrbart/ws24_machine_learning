"""
    Bayes Classificator
"""

#===== IMPORTS =======================================
import os
import pandas as pd
import numpy as np
import random

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
    design_matrix = []

    for celeb in celeb_list:
        for index, pic in enumerate(os.listdir(f"{pic_data}/{celeb}")):

            # Read in the picture
            image = io.imread(f"{pic_data}/{celeb}/{pic}", as_gray = True)
            image = transform.resize(image, (32, 32), anti_aliasing=True)
            image = image.flatten()

            # Save the test picture
            if celeb == "George_W_Bush":
                image = np.insert(image, 0, 1)

            else:
                image = np.insert(image, 0, -1)

            design_matrix.append(image)


    # Randomize and then seperate
    random.shuffle(design_matrix)

    sep = int(len(design_matrix)*0.6)
    train_data = design_matrix[0:sep]
    test_data = design_matrix[sep:-1]

    design_matrix = pd.DataFrame(train_data)
    test_data = pd.DataFrame(test_data)

    # Get labels back to perform pca
    design_labels = list(design_matrix.iloc[:, 0])
    design_matrix = design_matrix.iloc[:, 1:]

    test_labels = test_data.iloc[:, 0]
    test_data = test_data.iloc[:, 1:]

    # PCA
    # Center and normalize
    for entry in design_matrix:
        variance = design_matrix[entry].var()
        design_matrix[entry] -= design_matrix[entry].mean()
        design_matrix[entry] /= math.sqrt(variance)

        # TODO
        test_data[entry] -= design_matrix[entry].mean()
        test_data[entry] /= math.sqrt(variance)

    # SVD
    # TODO desing_matrix[1:]
    U, D, V = np.linalg.svd(design_matrix)
    eigenvectors = V
    eigenfaces = eigenvectors[:7]

    # Transform the data to the first 7 eigenfaces
    scores = pd.DataFrame(np.dot(design_matrix, eigenvectors[:7].T))
    test_scores = pd.DataFrame(np.dot(test_data, eigenvectors[:7].T))


    ### Bayes Classificator
    bush = {
        "mean" : [],
        "variance" : []
    }

    not_bush = {
        "mean" : [],
        "variance" : []
    }


    # Get Mean and Var of each pixel depending on class
    scores["labels"] = design_labels
    bush_scores = scores.loc[scores["labels"] == 1.0]
    not_bush_scores = scores.loc[scores["labels"] == -1.0]

    for entry in bush_scores:
        bush["mean"].append(bush_scores[entry].mean())
        bush["variance"].append(bush_scores[entry].var())

    for entry in not_bush_scores:
        not_bush["mean"].append(not_bush_scores[entry].mean())
        not_bush["variance"].append(not_bush_scores[entry].var())

    bush = pd.DataFrame(bush)
    not_bush = pd.DataFrame(not_bush)


    # Compute the A-Priori of train data
    p_is_bush = design_labels.count(1) / len(design_labels)
    p_not_bush = 1 - p_is_bush


    # Calculate the Gaussian probability values for each feature and class
    # Check for test data
    print("\nChecking results on TEST data:\n")
    true_positive = 0
    false_positive = 0
    true_negativ = 0
    false_negativ = 0
    dates = 0
    for ind, test_score in test_scores.iterrows():

        dates += 1
        pdfs = {
            "P_bush" : [],
            "P_not" : []
        }

        # Compute alle the PDFs for this test scores for every 
        for index, value in enumerate(test_score.values):

            mean = bush["mean"][index]
            not_mean = not_bush["mean"][index]

            variance = bush["variance"][index]
            not_variance = not_bush["variance"][index]

            pdfs["P_not"].append((1 / (math.sqrt(2 * math.pi * not_variance))) * math.exp(-(value - not_mean)**2 / (2 * not_variance)))
            pdfs["P_bush"].append((1 / (math.sqrt(2 * math.pi * variance))) * math.exp(-(value - mean)**2 / (2 * variance)))
        

        # Now we got the pdfs P(x | c), probability of feature x under class c
        # Now look at decision
        prd_bush = 1
        for pdf in pdfs["P_bush"]:
            prd_bush *= (pdf * p_is_bush)

        prd_not_bush = 1
        for pdf in pdfs["P_not"]:
            prd_not_bush *= (pdf * p_not_bush)

        # Bush detected
        print(f"{prd_bush} ?= {prd_not_bush}")
        if prd_bush > prd_not_bush:

            if test_labels[ind] == 1:
                true_positive += 1
            else:
                false_positive += 1
        
        # Not bush detected
        else:
            
            if test_labels[ind] == 1:
                false_negativ += 1
            else:
                true_negativ += 1
        
    print(f"""
    True Positive:\t{true_positive/dates}
    False Positive:\t{false_positive/dates}
    False Negativ:\t{false_negativ/dates}
    True Negativ:\t{true_negativ/dates}
    """)
