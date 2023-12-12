'''
    Task 3.1 Machine Learning

'''

# ===== IMPORTS ======================================
import numpy as np
import sklearn
import random
from sklearn import datasets
from sklearn import model_selection
import matplotlib.pyplot as plt
from sklearn import svm


# ===== MAIN =========================================
if __name__ == "__main__":

    # Get the data
    digits = datasets.load_digits()

    ### 3.1.1
    print(f"""Dataset Info:
    Number of Labels:\t{len(np.unique(digits.target))}
    Labels:\t\t{np.unique(digits.target)}
    Number of Datasets:\t{len(digits.data)}
    Dimensions:\t8x8 = {len(digits.data[0])}
    """)

    # Select random ten pics
    random_index = []
    for i in range(10):
        random_index.append(random.randint(0, len(digits.data)))

    if input("Show Pics? [y/n] ") == "y":
        for index in random_index:
            image = digits.data[index].reshape((8, 8))
            plt.imshow(image, cmap='gray')  # 'gray' colormap for grayscale images
            plt.title(f"Sample Picture {index}")
            plt.axis('off')  # Turn off axis labels
            plt.show()


    ### 3.1.2
    train_data, test_data, train_label, test_label = model_selection.train_test_split(digits.data, digits.target, test_size=0.25)

    SVM = svm.SVC(C=1.0, gamma=0.015, kernel="rbf")
    SVM.fit(train_data, train_label)
        
    print(f"""
    Score on Train:\t{round(SVM.score(train_data, train_label), 3)}
    Score on Test:\t{round(SVM.score(test_data, test_label), 3)}
    => Overfitting
    """)


    SVM = svm.SVC(C=100.0, gamma=0.001, kernel="rbf")
    SVM.fit(train_data, train_label)
        
    print(f"""
    With better parameters:\n
    Score on Train:\t{round(SVM.score(train_data, train_label), 3)}
    Score on Test:\t{round(SVM.score(test_data, test_label), 3)}
    """)

    














