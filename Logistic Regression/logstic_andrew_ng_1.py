from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

def sigmod(x):
    return 1/(1+np.exp(-x))

def predict(var1,var2,thetha):
    return sigmod(thetha[0]+(thetha[1]*var1)+(thetha[2]*var2))

def graident(X,Y,thetha,iterations,alpha):
    m = len(X)
    for i in range(0,iterations):
        temp = sigmod(np.dot(X,thetha)) - Y
        temp = np.dot(X.T,temp)
        thetha = thetha - (alpha/m)*temp
    return thetha

data = pd.read_csv('data.csv',sep = ',',header = None)
X = data.iloc[:,:2]
Y = data.iloc[:,2]
m = len(X)
data.head()

mean = np.mean(data)
std = np.std(data)

X = (X - np.mean(X))/np.std(X)

ones = np.ones((m,1))
X = np.hstack((ones, X))
alpha = 0.01
iterations = 400
thetha = [1.0,1.0,1.0]

thetha = graident(X,Y,thetha,iterations,alpha)
#print(thetha)

while True:
    var1 = float(input())
    var2 = float(input())

    var1 = (var1-mean[0])/std[0]
    var2 = (var2-mean[1])/std[1]

    print(predict(var1,var2,thetha))

    x = input("Do u want to quit (Y/N)")
    if x == 'y':
        break
