"""
Created on Sat Dec 15 23:47:23 2018

@author: shubhcyanogen

"""
import numpy as np
import pandas as pd
import scipy as sp
import scipy.optimize
from scipy import stats
from scipy.stats import linregress
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split

def estimate_coef(x, y):

    n = np.size(x)

    m_x, m_y = np.mean(x), np.mean(y)
    SS_xy = np.sum(y*x) - n*m_y*m_x
    SS_xx = np.sum(x*x) - n*m_x*m_x

    b_1 = SS_xy / SS_xx
    b_0 = m_y - b_1*m_x
    return(b_0, b_1)

def plot_regression_line(x, y, b):

    plt.scatter(x, y, color = "m",
    marker = "o", s = 30)

    y_pred = b[0] + b[1]*x

    plt.plot(x, y_pred, color = "g")

    plt.xlabel('x')
    plt.ylabel('y')

    plt.show()

def main():
#input taken from a csv file
    x_val = pd.read_csv('/Users/shubhcyanogen/Desktop/X_values.csv')
    y_val = pd.read_csv('/Users/shubhcyanogen/Desktop/Y_values.csv')
    x = x_val.values
    y = y_val.values
    b = estimate_coef(x, y)
    print("Estimated coefficients:\nb_0 = {} \
          \nb_1 = {}".format(b[0], b[1]))
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=101)
    lm = LinearRegression()
    X_train= X_train.reshape(-1, 1)
    y_train= y_train.reshape(-1, 1)
    X_test = X_test.reshape(-1, 1)
    lm.fit(X_train,y_train)
    predictions = lm.predict(X_test)
    plt.scatter(y_test,predictions)
    print("Blue Dots Are Predictions & Red Are Given Points")
#method for linear regresision analysis
    lm = LinearRegression()
    lm.fit(X_train, y_train)
    b0 = lm.intercept_
    b1 = lm.coef_
    Y_pred = b0 + b1 * X_train
    y_predict = lm.predict(X_test)
    print("Mean Absolute Error :",mean_absolute_error(y_test, y_predict))
    print("Mean Squared Error :",mean_squared_error(y_test, y_predict))
    print("Root Mean Squared Error :",np.sqrt(mean_absolute_error(y_test, y_predict)))
    result = stats.linregress(x[:,0], y[:,0])
    print("Regression Analysis")
    print(result)
# plotting regression line
    plot_regression_line(x, y, b)
if __name__ == "__main__":
    main()
