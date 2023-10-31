#===== IMPORTS =======================================
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from scipy import special
import numpy as np
from numpy.linalg import inv
import pandas as pd
import math


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

# Calculate the weights by hand, solve W = (X.T * X)**-1 * X.T * y
XT_inverse = inv(np.dot(data.T, data))
full_transform = np.dot(XT_inverse, data.T)
weights = np.dot(full_transform, true_fish)

# Show the found weights
print(f"Solution of regression by hand:\n{weights}\n\n")

# Now do the same but with SciPy
X = np.asarray(data)
Y = np.asarray(true_fish)

model = LinearRegression(fit_intercept=False).fit(X, Y)
print(f"Solution with SkLearn:\n{model.coef_}\n\n")

# Compute RMSE / variance / NLL 
pred_fish = np.matmul(data, weights)

RMSE = np.sqrt(np.mean((true_fish - pred_fish)**2))
variance = np.mean((true_fish - pred_fish)**2)
NLL = 0.5 * np.log(2 * np.pi * variance) + 0.5 * np.mean((true_fish - pred_fish)**2) / variance
print(f"RMSE:\t\t{RMSE}\nVariance:\t{variance}\nNLL:\t\t{NLL}\n")


# Compute mean of pred and real
mean_true = np.mean(true_fish)
mean_pred = np.mean(pred_fish)

percentile025 = mean_pred - 1.96 * RMSE
percentile975 = mean_pred + 1.96 * RMSE

print(f"""
Mean True:\t{mean_true}
Mean Pred:\t{mean_pred}
2.5%-ile:\t{percentile025}
97.5%-tile\t{percentile975}\n
Gaussian not suited because negative values possible!\n""")


# Now with Poisson Distribution
