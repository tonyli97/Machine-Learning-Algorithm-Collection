import numpy as np
import matplotlib.pyplot as plt

X_test = np.loadtxt('X_test.csv', delimiter = ',')
X_train = np.loadtxt('X_train.csv', delimiter = ',')
y_test = np.loadtxt('y_test.csv', delimiter = ',')
y_train = np.loadtxt('y_train.csv', delimiter = ',')

#part1 a)
U,S,V = np.linalg.svd(X_train)

df_list = []
wrr1 = []
wrr2 = []
wrr3 = []
wrr4 = []
wrr5 = []
wrr6 = []
wrr7 = []
for lamda in range(0,5001):
    I = np.identity(X_train.shape[1])
    wrr = np.dot(np.linalg.inv(lamda * I + X_train.T @ X_train),X_train.T @ y_train)
    df = np.trace(X_train @ np.linalg.inv(X_train.T@X_train + lamda*I) @ X_train.T)
    test = np.sum(S * S / (S*S + lamda))
    df_list.append(test)
    wrr1.append(wrr[0])
    wrr2.append(wrr[1])
    wrr3.append(wrr[2])
    wrr4.append(wrr[3])
    wrr5.append(wrr[4])
    wrr6.append(wrr[5])
    wrr7.append(wrr[6])

plt.plot(df_list,wrr1,label="1.cylinders")
plt.plot(df_list,wrr2,label="2.displacement")
plt.plot(df_list,wrr3,label="3.horsepower")
plt.plot(df_list,wrr4,label="4.weight")
plt.plot(df_list,wrr5,label="5.acceleration")
plt.plot(df_list,wrr6,label="6.year_made")
plt.plot(df_list,wrr7,label="7.I")
plt.legend(loc='best')
plt.xlabel("degree of freedom")
plt.ylabel("wrr")
plt.show()

#part1 c)
RMSD_list = []
for lamda in range(0,51):
    I = np.identity(X_train.shape[1])
    wrr = np.dot(np.linalg.inv(lamda * I + X_train.T @ X_train),X_train.T @ y_train)
    y = np.dot(X_test,wrr)
    RMSD = np.sqrt(np.sum(np.power(y_test - y,2)) / y.shape[0])
    RMSD_list.append(RMSD)

lamda_list = list(range(51))

plt.plot(lamda_list,RMSD_list)
plt.xlabel("lamda")
plt.ylabel("RMSE")
plt.show()



#part2 d)
#Construct X for polynomial regression p = 2
tmp = np.ones((X_train.shape[0],1))
X_train_tmp = X_train[:,0:6]
X_train_tmp_2 = X_train_tmp ** 2
X_train_tmp_2 = (X_train_tmp_2 - np.mean(X_train_tmp_2,axis=0)) / np.std(X_train_tmp_2,axis = 0)
X_train_2 = np.concatenate((np.concatenate((tmp,X_train_tmp),axis = 1),X_train_tmp_2),axis = 1)
X_test_tmp = X_test[:,0:6]
X_test_tmp_2 = X_test_tmp **2
X_test_tmp_2 = (X_test_tmp_2 - np.mean(X_test_tmp_2,axis=0)) / np.std(X_test_tmp_2,axis=0)
tmp = np.ones((X_test.shape[0],1))
X_test_2 = np.concatenate((np.concatenate((tmp,X_test_tmp),axis = 1),X_test_tmp_2),axis=1)


RMSD_list_2 = []
for lamda in range(0,101):
    I = np.identity(X_train_2.shape[1])
    wrr = np.dot(np.linalg.inv(lamda * I + X_train_2.T @ X_train_2),X_train_2.T @ y_train)
    y = np.dot(X_test_2,wrr)
    RMSD = np.sqrt(np.sum(np.power(y_test - y,2)) / y.shape[0])
    RMSD_list_2.append(RMSD)

plt.plot(list(range(0,101)),RMSD_list_2,label="p=2")


#Construct X for polynomial regression p = 3
tmp = np.ones((X_train.shape[0],1))
X_train_tmp = X_train[:,0:6]
X_train_tmp_2 = X_train_tmp ** 2
X_train_tmp_2 = (X_train_tmp_2 - np.mean(X_train_tmp_2,axis=0)) / np.std(X_train_tmp_2,axis=0)
X_train_tmp_3 = X_train_tmp ** 3
X_train_tmp_3 = (X_train_tmp_3 - np.mean(X_train_tmp_3,axis=0)) / np.std(X_train_tmp_3,axis=0)
X_train_3 = np.concatenate((np.concatenate((np.concatenate((tmp,X_train_tmp),axis=1),X_train_tmp_2),axis=1),X_train_tmp_3),axis=1)
X_test_tmp = X_test[:,0:6]
X_test_tmp_2 = X_test_tmp **2
X_test_tmp_2 = (X_test_tmp_2 - np.mean(X_test_tmp_2,axis=0)) / np.std(X_test_tmp_2,axis=0)
X_test_tmp_3 = X_test_tmp **3
X_test_tmp_3 = (X_test_tmp_3 - np.mean(X_test_tmp_3,axis=0)) / np.std(X_test_tmp_3,axis=0)
tmp = np.ones((X_test.shape[0],1))
X_test_3 = np.concatenate((X_test_2,X_test_tmp_3),axis=1)



RMSD_list_3 = []
for lamda in range(0,101):
    I = np.identity(X_train_3.shape[1])
    wrr = np.dot(np.linalg.inv(lamda * I + X_train_3.T @ X_train_3),X_train_3.T @ y_train)
    y = np.dot(X_test_3,wrr)
    RMSD = np.sqrt(np.sum(np.power(y_test - y,2)) / y.shape[0])
    RMSD_list_3.append(RMSD)

plt.plot(list(range(0,101)),RMSD_list_3,label="p=3")
plt.plot(lamda_list,RMSD_list,label="p=1")
plt.legend(loc='best')
plt.xlabel("lamda")
plt.ylabel("RMSE")
plt.show()
