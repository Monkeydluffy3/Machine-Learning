import numpy as np
import pandas as pd

def cost(theta,size,no,mean,std):
    return theta[0] + (theta[1]*(size-mean[0])/std[0]) + (theta[2]*(no-mean[1])/std[1]) 
  

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

t = (np.linalg.inv(np.dot(X.T,X))) 

theta = np.dot(np.dot(t,X.T),y)

print(theta)

print(cost(theta,2104,3,mean,std))