import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

X_test = np.loadtxt('X_test.csv', delimiter = ',')
X_train = np.loadtxt('X_train.csv', delimiter = ',')
y_test = np.loadtxt('y_test.csv', delimiter = ',')
y_train = np.loadtxt('y_train.csv', delimiter = ',')

b_list = [5,7,9,11,13,15]
var = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]

def RBF(x,y,b):
    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()
    return np.exp(-1/b*np.dot(x-y,x-y))


result = pd.DataFrame(index = b_list, columns=var)

#a
for b in b_list:
    for v in var:
        Ktrain = np.zeros((X_train.shape[0],X_train.shape[0]))
        Ktest = np.zeros((X_test.shape[0],X_train.shape[0]))
        for i in range(X_train.shape[0]):
            Ktrain[i,:] = np.apply_along_axis(RBF,1,X_train,X_train[i,:],b)
        for i in range(X_test.shape[0]):
            Ktest[i,:] = np.apply_along_axis(RBF,1,X_train,X_test[i,:],b)
        y_pred = Ktest @ (np.linalg.inv(v * np.identity(Ktrain.shape[0]) + Ktrain)) @ y_train
        x = np.asarray(y_test).flatten()
        y = np.asarray(y_pred).flatten()
        rmse = np.sqrt(np.dot(x-y,x-y)/X_test.shape[0])
        result.loc[b,v] = rmse

print (result)
filename = 'result.csv'
pickle.dump(result, open(filename, 'wb'))

#b
print (result.min())

#c
b = 5
var = 2
X_train2 = X_train[:,3].reshape(-1,1)
Ktrain = np.zeros((X_train2.shape[0],X_train2.shape[0]))
Ktest = np.zeros((X_train2.shape[0],X_train2.shape[0]))

for i in range(X_train2.shape[0]):
    Ktrain[i,:] = np.apply_along_axis(RBF,1,X_train2,X_train2[i,:],b)

y_pred = Ktrain @ (np.linalg.inv(var * np.identity(Ktrain.shape[0]) + Ktrain)) @ y_train
x = np.asarray(X_train2).flatten()
sort_x = np.argsort(x)
y1 = np.asarray(y_train).flatten()
y2 = np.asarray(y_pred).flatten()
plt.figure()
plt.scatter(x[sort_x],y1[sort_x])
plt.plot(x[sort_x],y2[sort_x],'b-')
plt.xlabel('Car Weight (Dim 4)')
plt.ylabel('Y value')
plt.savefig("hw2c")
