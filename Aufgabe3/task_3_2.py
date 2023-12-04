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


    ### 3.2a)
    for i in range(0, 3):

        train_data, test_data, train_label, test_label = model_selection.train_test_split(digits.data, digits.target, test_size=0.25)
        SVM = svm.SVC(C=100.0, gamma=0.001, kernel="rbf")
        SVM.fit(train_data, train_label)
            
        print(f"""
        Model {i}:\n
        Score on Train:\t{round(SVM.score(train_data, train_label), 3)}
        Score on Test:\t{round(SVM.score(test_data, test_label), 3)}
        """)

    
    ### 3.2b)
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

    ### 3.2c)
    # Cross Validation is the progress of splitting into n set and then train n times
    # always with n-1 train sets and one test set for all permutations! 

    # Try to find best gamma
    gammas = np.logspace(-7, -1, 10)
    gammas = np.around(gammas, 7)
    results_train = []
    results_test = []
    for split in range(0, 5):

        train_data, test_data, train_label, test_label = model_selection.train_test_split(digits.data, digits.target, train_size=500, test_size=500)
        results_train.append([])
        results_test.append([])

        for gamma in gammas:

            SVM = svm.SVC(C=10.0, gamma=gamma, kernel="rbf")
            SVM.fit(train_data, train_label)

            results_train[-1].append(SVM.score(train_data, train_label))
            results_test[-1].append(SVM.score(test_data, test_label))


    # Plot the results for train
    plt.xscale("log")
    for index, split in enumerate(results_train):
        plt.plot(gammas, split)
    plt.legend(gammas)
    plt.show()

    # Plot the results for test
    plt.xscale("log")
    for index, split in enumerate(results_test):
        plt.plot(gammas, split)
    plt.legend(["Split 1", "Split 2", "Split 3", "Split 4", "Split 5"])
    plt.show()


    ### 3.2d)
    svc_params = {
    'C': np.logspace(-1, 2, 4),
    'gamma': np.logspace(-4, 0, 5), 
    }

    train_data, test_data, train_label, test_label = model_selection.train_test_split(digits.data, digits.target, train_size=500)

    SVM = svm.SVC(C=10.0, gamma=0.1, kernel="rbf")
    model = model_selection.GridSearchCV(param_grid=svc_params, cv=3, estimator=SVM).fit(train_data, train_label)

    print(f"""
    Best Params:\t{model.best_params_}
    Best Score:\t\t{model.best_score_}
    Score on Test data:\t{model.score(test_data, test_label)}
    """)

    if input("Want detailed results? [y/N]\t") == "y" : print(f"\n{model.cv_results_}")
















