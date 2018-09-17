import numpy as np
import pandas as pd

def cost(theta,size,no,mean,std):
    return theta[0] + (theta[1]*(size-mean[0])/std[0]) + (theta[2]*(no-mean[1])/std[1]) 

def gradientDescentMulti(X, y, theta, alpha, iterations):
    m = len(y)
    for _ in range(iterations):
        temp = np.dot(X, theta) - y
        temp = np.dot(X.T, temp) #multiply with the transpose of X
        theta = theta - (alpha/m) * temp
    return theta   

data = pd.read_csv('dataset-mul.csv', sep = ',', header = None)
X = data.iloc[:,0:2] # read first two columns into X
y = data.iloc[:,2] # read the third column into y
m = len(y) # no. of training samples
data.head()

mean = np.mean(X)
std = np.std(X)

X = (X - np.mean(X))/np.std(X)

ones = np.ones((m,1))
X = np.hstack((ones, X))
alpha = 0.01
num_iters = 400
theta = np.zeros((3,1))
y = y[:,np.newaxis]

theta = gradientDescentMulti(X, y, theta, alpha, num_iters)
print(theta)

print(cost(theta,2104,3,mean,std))