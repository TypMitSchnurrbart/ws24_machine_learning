"""
    Test module
"""

#===== IMPORTS =======================================
import json
import csv
import numpy

#===== FUNCTIONS =====================================
def pca():

    return


#===== MAIN ==========================================
if __name__ == "__main__":

    # Data Init
    data = []

    # Import data
    with open("boston.csv", "r") as file:
        boston_data = csv.DictReader(file)

        for line in boston_data:

            data.append([])
            
            for key in line:
                data[-1].append(float(line[key]))


    # Compute mean tensor
    mean_tensor = [0]
    var_index = 0
    for attrb in range(0,len(data[-1])-1):

        mean_cache = 0
        var_index += 1
        for index, line in enumerate(data):
            mean_cache += line[var_index]

        mean_tensor.append(mean_cache / len(data))

    
    # Compute the centering for every data point
    var_index = 1
    for attrb in range(0,len(data[-1])-1):

        for index, line in enumerate(data):
            data[index][var_index] -= mean_tensor[var_index]

        var_index += 1

    print(data)
        
    

        
