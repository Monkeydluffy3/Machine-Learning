import pandas as pd
import numpy as np 
from matplotlib import pyplot as plt 
import random
import math

def predict(thetha,x,y):
    k = thetha[0]+(thetha[1]*x)+(thetha[2]*y)+(thetha[3]*x*x)+(thetha[4]*y*y)
    if k > 0.5:
        k = 1
    else:
        k = 0    
    return k

def sigmod(x):
    return 1/(1+np.exp(-x))

def graident(thetha,train_input,train_output,alpha,iterations):
    m = len(train_input)
    for i in range(0,iterations):
        temp = sigmod(np.dot(train_input,thetha))
        temp = temp - train_output
        temp = np.dot(train_input.T,temp)
        thetha = thetha - (alpha/m)*temp
    return thetha
             

data = pd.read_csv('2_data.csv',sep = ',',header=None)
#print(data)

train_input  = data.iloc[:,:2]
train_output = data.iloc[:,2]
m = len(train_input)

x = train_input[:][0]
y = train_input[:][1]

temp = []

for i in range(0,m):
    temp.append([1,x[i],y[i],(x[i]*x[i]),(y[i]*y[i])])

data = pd.DataFrame(temp)
#print(data)


# # plotting graph
# col = 'red'
# for i in range(0,m):
#     if train_output[i] == 1:
#         col = 'red'
#     else:
#         col = 'blue'        
#     plt.scatter(x[i],y[i],color=col)
# plt.xlabel('x-axis')
# plt.ylabel('y-axis')    
# plt.show()

ones = np.ones((m,1))
train_input = np.hstack((ones,train_input))


alpha = 0.01
iterations = 400
thetha = [1.0,1.0,1.0,1.0,1.0]


thetha = graident(thetha,data,train_output,alpha,iterations)
print(thetha)


# # plotting graph
# col = 'red'

# for i in range(0,m):
#     if train_output[i] == 1:
#         col = 'red'
#     else:
#         col = 'blue'        
#     plt.scatter(x[i],y[i],color=col)   
# plt.xlabel('x-axis')
# plt.ylabel('y-axis')  
# plt.show()

while True:
    x = float(input())
    y = float(input())
    print(predict(thetha,x,y))
