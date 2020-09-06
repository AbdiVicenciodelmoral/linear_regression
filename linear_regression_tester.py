import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import graph_plotter as gr
import linear_regression as LR
from sklearn.linear_model import LinearRegression
from sklearn import datasets
import os 


X_train, Y_train = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=4)
m= 0
b = 0
##### Linear Regression using scikit learn ####
#Create model using the Sci-Kit Learn libraries
#call new model
linreg = LinearRegression()

X_train = X_train.reshape(-1,1)

#Fit the model to the data
linreg.fit(X_train,Y_train)

#predict output y based input x
y_pred = linreg.predict(X_train)


##### Linear Regression Ordinary Least Squares #####

# init class
LR_OLS = LR.LinearRegression_OLS()

# Fit Model and get the estimated weight(m) and bias(b).
m,b = LR_OLS.fit(X_train,Y_train)
print("m = ",m)
print("b = ",b)

#Calculates the RMSE
rmse_err = LR_OLS.root_mean_squared_error(X_train,Y_train)

# predict(input x, estimated m, estimated b)
input_x = 80
single_predict = LR_OLS.predict(input_x,m,b)

print("input x = ", input_x)
print("y = ", single_predict)
print("RMSE = ", rmse_err)
#Constructs and calls graph plotter class
y_pred_ols = LR_OLS.predict(X_train,m,b)
graph = gr.graph_plot()
graph.plot(X_train,Y_train,y_pred_ols,y_pred)
