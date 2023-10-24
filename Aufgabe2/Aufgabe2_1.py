#===== IMPORTS =======================================
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


#===== MAIN ==========================================
d = np.load('fishing.npz')
Xt = d['Xt'] #"livebait","camper","persons","child"
Xte = d['Xte']
true_fish = d['yt']
test_fish = d['yte']

# Clean dataframe
data = {
    "intercept" : [],
    "livebait" : [],
    "camper" : [],
    "persons" : [],
    "child" : [],
}

for value in Xt:

    data["intercept"].append(1.0)
    data["livebait"].append(value[0])
    data["camper"].append(value[1])
    data["persons"].append(value[2])
    data["child"].append(value[3])

data = pd.DataFrame(data)

#weights = np.dot(data.T, data).inverse()