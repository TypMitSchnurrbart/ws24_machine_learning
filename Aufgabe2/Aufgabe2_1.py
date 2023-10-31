#===== IMPORTS =======================================
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from scipy import special
import numpy as np
from numpy.linalg import inv
import pandas as pd
import math

def poisson_pmf(mu, k):
    return np.exp(-mu) * (mu**k) / factorial(k)

def poisson_nll(beta, X, y):
    mu = np.exp(np.dot(X, beta))
    return -np.sum(mu - y * np.log(mu))



#===== MAIN ==========================================
d = np.load('fishing.npz')
Xt = d['Xt'] #"livebait","camper","persons","child"
Xte = d['Xte']
true_fish_train = d['yt']
true_test_fish = d['yte']

# Clean dataframe
data = {
    "intercept" : [],
    "livebait" : [],
    "camper" : [],
    "persons" : [],
    "child" : [],
}


test_data = {
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

for value in Xte:

    test_data["intercept"].append(1.0)
    test_data["livebait"].append(value[0])
    test_data["camper"].append(value[1])
    test_data["persons"].append(value[2])
    test_data["child"].append(value[3])

data = pd.DataFrame(data)
test_data = pd.DataFrame(test_data)

# Calculate the weights by hand, solve W = (X.T * X)**-1 * X.T * y
XT_inverse = inv(np.dot(data.T, data))
full_transform = np.dot(XT_inverse, data.T)
weights = np.dot(full_transform, true_fish_train)

# Show the found weights
print(f"Solution of regression by hand:\n{weights}\n\n")

# Now do the same but with SciPy
X = np.asarray(data)
Y = np.asarray(true_fish_train)

model = LinearRegression(fit_intercept=False).fit(X, Y)
print(f"Solution with SkLearn:\n{model.coef_}\n\n")

# Compute RMSE / variance / NLL 
pred_fish = np.matmul(test_data, weights)

RMSE = np.sqrt(np.mean((true_test_fish - pred_fish)**2))
variance = np.mean((true_test_fish - pred_fish)**2)

# NLL based on Gaussian
NLL = 0.5 * np.log(2 * np.pi * variance) + 0.5 * np.mean((true_test_fish - pred_fish)**2) / variance
print(f"RMSE:\t\t{RMSE}\nVariance:\t{variance}\nNLL:\t\t{NLL}\n")


# Compute mean of pred and real
mean_true = np.mean(true_fish_train)
mean_pred = np.mean(pred_fish)

percentile025 = mean_pred - 1.96 * RMSE
percentile975 = mean_pred + 1.96 * RMSE

print(f"""
Mean True:\t{mean_true}
Mean Pred:\t{mean_pred}
2.5%-ile:\t{percentile025}
97.5%-tile\t{percentile975}\n
Gaussian not suited because negative values possible!\n""")

### UNTIL HERE IS CORRECT AND WORKING

# TODO:
# Now with Poisson Distribution and Gradient Descent
beta = np.ones(data.shape[1])
learning_rate = 0.001
num_epochs = 5000

for epoch in range(num_epochs):
    mu = np.exp(np.dot(data, beta))
    gradient = np.dot(data.T, true_fish_train - mu)
    beta += learning_rate * gradient  

pred_values = np.exp(np.dot(test_data, beta))

# Calculate Root Mean Square Error (RMSE) on the test set
RMSE = np.sqrt(np.mean((pred_values - true_test_fish) ** 2))

# Calculate the NLL on the test set
NLL = poisson_nll(beta, test_data, true_test_fish)

print(f"With Gradient Descent and Poisson:\nRMSE:\t\t{RMSE}\nNLL:\t\t{NLL}\n")
 


