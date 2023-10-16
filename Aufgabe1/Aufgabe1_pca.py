"""
    Test module
"""

#===== IMPORTS =======================================
import json
import csv
import numpy as np
import pandas as pd
import math

#===== FUNCTIONS =====================================
def pca():

    return


#===== MAIN ==========================================
if __name__ == "__main__":

    # Import data
    data = pd.read_csv("boston.csv")

    data.drop("TGT", axis=1, inplace=True)
    data.drop("Index", axis=1, inplace=True)

    # Center and normalize variance
    for entry in data:
        variance = data[entry].var()
        data[entry] -= data[entry].mean()
        data[entry] /= math.sqrt(variance)

    # Display head
    print(data.head())

    # Creating the design matrix
    print(f"Creating design matrix {len(data)}x{len(data.columns)}")
    design_matrix = np.empty((len(data), len(data.columns)))
    design_matrix[:, :len(data.columns)] = data.values

    # Compute svd
    test = np.linalg.svd(design_matrix)

    print(test)


    
    

        
