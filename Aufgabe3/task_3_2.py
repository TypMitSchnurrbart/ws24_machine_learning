'''
    Task 3.2 Machine Learning

'''

# ===== IMPORTS ======================================
import numpy as np
import sklearn
import random
import math
from sklearn import datasets
from sklearn import model_selection
import matplotlib.pyplot as plt
from sklearn import svm


# ===== MAIN =========================================
if __name__ == "__main__":

    # Get the data
    digits = datasets.load_digits()


    ### 3.2a
    for i in range(0, 3):

        train_data, test_data, train_label, test_label = model_selection.train_test_split(digits.data, digits.target, test_size=0.25)
        SVM = svm.SVC(C=100.0, gamma=0.001, kernel="rbf")
        SVM.fit(train_data, train_label)
            
        print(f"""
        Model {i}:\n
        Score on Train:\t{round(SVM.score(train_data, train_label), 3)}
        Score on Test:\t{round(SVM.score(test_data, test_label), 3)}
        """)

    
    ### 3.2b
    SVM = svm.SVC(C=100.0, gamma=0.001, kernel="rbf")
    scores = model_selection.cross_val_score(estimator=SVM,
                                            X=train_data,
                                            y=train_label,
                                            cv=10)
    print(f"""
    Cross Validation Scores:
    Kreuzvalidierungsgenauigkeit:\t{round(scores.mean(), 3)}
    Standardabweichung:\t\t\t{round(math.sqrt(scores.var()), 3)}
    """)















