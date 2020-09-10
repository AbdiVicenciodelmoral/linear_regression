#import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.linear_model import LinearRegression
# Import the functions from file
import linear_regression as LR

#Generate a random data points for Linear Regression
X, y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=3)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)


##### Linear Regression using scikit learn ####
#Create model using the Sci-Kit Learn libraries

#call new model
linreg = LinearRegression()

#Fit the model to the data
linreg.fit(X_train,y_train)

# Prints the coefficient (weight aka: m) and the intercept (Bias aka: b)
print("sci_kit m = ",linreg.coef_)
print("sci_kit b = ",linreg.intercept_)

#predict output of y based on x values
y_pred = linreg.predict(X)



##### Linear Regression Ordinary Least Squares #####
# Create model using OLS method

# init class
LR_OLS = LR.LinearRegression_OLS()

# Fit Model and get the estimated weight(m) and bias(b).
m,b = LR_OLS.fit(X_train,y_train)
print("OLS_m = ",m)
print("OLS_b = ",b)

# predict(input x, estimated m, estimated b)
ols_y_predict = LR_OLS.predict(X,m,b)

# Calculate the mean squared error
OLS_mse = LR_OLS.mean_squared_error(y,ols_y_predict)
print("OLS_mse = ",OLS_mse)


##### Linear Regression Gradient Descent #####
# Create model using Gradient Descent
LR_Grad_Desc = LR.LinearRegression_Gradient_Descent(learning_rate=0.01, n_iters=1000)


grD_m, grD_b = LR_Grad_Desc.fit(X_train, y_train)
print("GradDesc_m = ",grD_m)
print("GradDesc_b = ",grD_b)

# Make predictions for the values of x
grDesc_y_pred = LR_Grad_Desc.predict(X)

# Calculate the mean squared error
OLS_mse = LR_Grad_Desc.mean_squared_error(y,grDesc_y_pred)
print("OLS_mse = ",OLS_mse)

# Plot the data
fig = plt.figure(figsize=(8,6))
cmap = plt.get_cmap('Blues')
m1 = plt.scatter(X_train, y_train, color=cmap(0.8), s=10)
m2 = plt.scatter(X_test, y_test, color=cmap(0.5), s=10)
plt.plot(X, y_pred, color='Green', linewidth=2, label="Sci kit Prediction")
plt.plot(X, ols_y_predict, color='Yellow', linewidth=2, label="OLS Prediction")
plt.plot(X, grDesc_y_pred, color='Orange', linewidth=2, label="Gradient Descent Prediction")
plt.grid(True)
plt.show()
