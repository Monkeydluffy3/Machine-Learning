import csv
from matplotlib import pyplot as plt 

def compute(t0,t1,x_data,y_data):
        sum1 = 0
        for i in range(0,len(x_data)):
                sum1+=(t0+(t1*x_data[i])-y_data[i])*x_data[i]        
        return sum1   

def com(t0,t1,x_data,y_data):
       sum1 = 0
       for i in range(0,len(x_data)):
                sum1+=(t0+(t1*x_data[i])-y_data[i])       
       return sum1

def predict(t1,x_value):
        return t0+(t1*x_value)


cs  = open('train.csv','r')
xy_list = csv.reader(cs,delimiter = ',')
x_data = []
y_data = []
z = 0
xy = []
for row in xy_list:
        xy.append(row) 

for i in range(1,len(xy)):
        if len(xy[i])==2:                
                x_data.append(float(xy[i][0]))
                y_data.append(float(xy[i][1]))


#plott graph
plt.scatter(x_data,y_data)
plt.xlabel("x-axis")
plt.ylabel("y-axis")
plt.show()

m = len(x_data)
t1 = 1.0
t0 = 0
alpha  = 0.0001
count  = 0

#calculate parameters of the model
while True:
        temp1 = t0 - (alpha*com(t0,t1,x_data,y_data))/m
        temp = t1 - (alpha*compute(t0,t1,x_data,y_data))/m
        if temp == t1:
                break
        t1 = temp
        t0 = temp1
        count = count+1
        if count > 1500:
                break           

#test the model

print(t0,t1)

te = open('test.csv','r')
t_list = csv.reader(te,delimiter = ',')
xy.clear()
for row in t_list:
        xy.append(row) 

count = 0
for i in range(1,len(xy)):
        if len(xy[i])==2:                
                x_value = float(xy[i][0])
                y_value = float(xy[i][1])

                gen_y = predict(t1,x_value)

                if (gen_y-y_value) < 1:
                        count = count+1
                elif (y_value-gen_y) < 1:
                        count = count+1       

l = len(xy)
print("Accuracy of Model is: "+str((count/l)*100))