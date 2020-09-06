import numpy as np

class LinearRegression_OLS:
    def __init__(self):
        self.mean_x = None
        self.mean_y = None
        self.var = 0
        self.covar = 0
        self.m = 0
        self.b = 0
        self.data_length = None
    
    def fit(self,x_Vals,y_Vals):
        # We need to get the mean of all the x
        # and y values, respectively.
        # mean = (sum of all values)/(number of values)
        self.mean_x = np.mean(x_Vals)
        self.mean_y = np.mean(y_Vals)

        # and now need to calculate the variance and covariance
        # variance = (sum of all x values - mean of x all values)^2)/(number of values)
        # covariance = (sum of all x values - mean of x all values) (sum of all y values - mean of y all values))/(number of values)
        # in order to sum all the values of x we need to iterate through
        # the column containing x.
        self.data_length = len(x_Vals)
        for i in range(len(x_Vals)):
            self.var += (x_Vals[i]-self.mean_x)**2
            self.covar += (x_Vals[i]-self.mean_x)*(y_Vals[i]-self.mean_y)

        # Now we need to estimate the coefficients for the
        # approximation y = mean of y - m * mean of x
        self.m = self.covar/self.var
        self.b = self.mean_y - self.m * self.mean_x
        return self.m,self.b
    
    def predict(self,x,m,b):
        return np.dot(x,m) + b


    #Root Mean Square Error (RMSE) is the standard deviation 
    #of the residuals (prediction errors). Residuals are a 
    #measure of how far from the regression line data points 
    #are; RMSE is a measure of how spread out these residuals 
    #are. In other words, it tells you how concentrated the 
    #data is around the line of best fit. 
    def root_mean_squared_error(self, test_x, test_y):
        rmse = 0
        for i in range(self.data_length):
            pred_y =  self.b + self.m* test_x[i]
            rmse += (test_y[i] - pred_y) ** 2
            
        rmse = np.sqrt(rmse/self.data_length)
        return rmse

    # SST = Total sum of squares 
    # SSR = Total sum of squares of residuals
    # R² score can measure the accuracy of the linear model
    # R² = SSR/SST
    def R_score(self,test_x,test_y):
        ss = 0
        sr = 0
        for i in range(self.data_length) :
            pred_y = self.b + self.m * test_x[i]
            sr += (test_y[i] - pred_y) **2
            ss += (test_y[i] - self.mean_y) ** 2
            
        score  = 1 - (sr/ss)


