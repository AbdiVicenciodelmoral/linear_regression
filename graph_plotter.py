#import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class graph_plot:

    def __init__(self):
        self.x = None
        self.y = None

    def plot(self, X_set, Y_set,y1,y2):
        #This plots the line 
        plt.plot(X_set, y1, color='#ff4d9d', label='OLS Linear Regression')
        plt.plot(X_set, y2, color='#8b0000', label='Scikit Linear Regression')
        #This plots the data points
        plt.scatter(X_set, Y_set, color='#40E0D0', label='Data Points')
        # x-axis label
        plt.xlabel('X values')
        #y-axis label
        plt.ylabel('Y values')
        plt.legend()
        plt.show()
