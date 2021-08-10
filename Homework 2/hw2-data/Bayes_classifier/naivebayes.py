import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import random
import math
from scipy.stats import poisson
from scipy.special import expit


def train_test_create(group,i):
    X_test = group[i].loc[:,group[i-1].columns!="y"]
    y_test = group[i]["y"]
    X_train = pd.DataFrame().reindex(columns=group[0].columns)
    for it in range(len(group)):
        if it == i:
            continue
        X_train = pd.concat([X_train,group[it]])
    y_train = X_train["y"]
    X_train = X_train.loc[:,group[i-1].columns!="y"]
    X_train = X_train.to_numpy()
    y_train = y_train.to_numpy()
    X_test = X_test.to_numpy()
    y_test = y_test.to_numpy()
    return X_train,y_train,X_test,y_test

def naive_bayes(X_train,y_train):
    pi = np.mean(y_train)

    sum_x_id_0 = np.zeros(X_train.shape[1])
    sum_x_id_1 = np.zeros(X_train.shape[1])
    ny_0 = 0
    ny_1 = 0
    for i in range(X_train.shape[0]):
        if y_train[i] == 0:
            sum_x_id_0 += X_train[i,:]
            ny_0 += 1
        else:
            sum_x_id_1 += X_train[i,:]
            ny_1 += 1
    lamda0 = (sum_x_id_0 + 1)/(ny_0 + 1)
    lamda1 = (sum_x_id_1 + 1)/(ny_1 + 1)

    return pi,lamda0,lamda1


def predict(X_test,pi,lamda0,lamda1):
    y_pred = np.zeros(X_test.shape[0])
    for j in range(X_test.shape[0]):
        pi_0 = pi**(0)*(1-pi)**(1-0)
        p1 = poisson.pmf(X_test[j],lamda0)
        prod1 = np.prod(p1)

        pi_1 = pi**(1)*(1-pi)**(1-1)
        p2 = poisson.pmf(X_test[j],lamda1)
        prod2 = np.prod(p2)

        y0 = pi_0 * prod1
        y1 = pi_1 * prod2
        if y0 < y1:
            y_pred[j] = 1
        else:
            y_pred[j] = 0
    return y_pred


def steepest_ascent(group):
    iters = 1000
    fig, axs = plt.subplots(2, 5, sharex=True, sharey=True)
    for its in range(10):
        X_train,y_train,X_test,y_test = train_test_create(group,its)
        y_train[y_train == 0] = -1
        y_test[y_test == 0] = -1
        X_train = np.column_stack((X_train,np.ones(X_train.shape[0])))
        X_test = np.column_stack((X_test,np.ones(X_test.shape[0])))
        weights = np.zeros(X_train.shape[1]).reshape(-1,1)
        objective = []
        for it in range(iters):
            eta = 0.01 / 4600
            sigma_i = expit(y_train.reshape(-1,1) * (X_train @ weights))
            objective.append(np.sum(np.log(sigma_i)))
            update = X_train.T @ (y_train.reshape(-1,1) * (1-sigma_i))
            weights += eta*update
        axs[math.floor(its/5),its%5].plot(range(iters),objective)
        axs[math.floor(its/5),its%5].set_title(its)
    fig.text(0.5, 0.04, 'iterations', ha='center')
    fig.text(0.0004, 0.5, "objective function", va='center', rotation='vertical')
    plt.savefig("hw2c",dpi=300)

def newton(group):
    iters = 100
    fig, axs = plt.subplots(2, 5, sharex=True, sharey=True)
    b00 = 0
    b01 = 0
    b11 = 0
    b10 = 0
    for its in range(10):
        X_train,y_train,X_test,y_test = train_test_create(group,its)
        y_train[y_train == 0] = -1
        y_test[y_test == 0] = -1
        X_train = np.column_stack((X_train,np.ones(X_train.shape[0])))
        X_test = np.column_stack((X_test,np.ones(X_test.shape[0])))
        weights = np.zeros(X_train.shape[1]).reshape(-1,1)
        objective = []
        for it in range(iters):
            eta = 0.01 / 4600
            sigma_i = expit(y_train.reshape(-1,1) * (X_train @ weights))
            objective.append(np.sum(np.log(sigma_i)))
            second_grad = -(sigma_i*(1-sigma_i)*X_train).T @ (X_train)
            first_grad = X_train.T @ (y_train.reshape(-1,1)*(1-sigma_i))
            weights -= eta*np.linalg.inv(second_grad).dot(first_grad)

        axs[math.floor(its/5),its%5].plot(range(iters),objective)
        axs[math.floor(its/5),its%5].set_title(its)
        y_pred = np.sign(X_test @ weights)
        for j in range(y_pred.shape[0]):
            if y_pred[j] == -1 and y_test[j] == -1:
                b00 += 1
            elif y_pred[j] == -1 and y_test[j] == 1:
                b10 += 1
            elif y_pred[j] == 1 and y_test[j] == -1:
                b01 += 1
            else:
                b11 += 1

    fig.text(0.5, 0.04, 'iterations', ha='center')
    fig.text(0.0004, 0.5, "objective function", va='center', rotation='vertical')
    plt.savefig("hw2d")

    return (b00,b01,b10,b11)

def main():
    header = pd.read_csv('README',skiprows=2,header=None,delim_whitespace=True)
    X = pd.read_csv('X.csv',sep = ',',names=list(header[1]))
    y = pd.read_csv('y.csv',header=None)
    X['y'] = y
    X_shuffle = X.sample(frac=1).reset_index(drop=True)

    group1 = X_shuffle.iloc[:460,:]
    group2 = X_shuffle.iloc[460:460*2,:]
    group3 = X_shuffle.iloc[460*2:460*3,:]
    group4 = X_shuffle.iloc[460*3:460*4,:]
    group5 = X_shuffle.iloc[460*4:460*5,:]
    group6 = X_shuffle.iloc[460*5:460*6,:]
    group7 = X_shuffle.iloc[460*6:460*7,:]
    group8 = X_shuffle.iloc[460*7:460*8,:]
    group9 = X_shuffle.iloc[460*8:460*9,:]
    group10 = X_shuffle.iloc[460*9:460*10,:]
    group = [group1,group2,group3,group4,group5,group6,group7,group8,group9,group10]

    c00 = 0
    c01 = 0
    c11 = 0
    c10 = 0
    avg_lamda0 = np.zeros(54)
    avg_lamda1 = np.zeros(54)
    for i in range(10):
        X_train,y_train,X_test,y_test = train_test_create(group,i)
        pi,lamda0,lamda1 = naive_bayes(X_train,y_train)
        avg_lamda0 += lamda0
        avg_lamda1 += lamda1
        y_pred = predict(X_test,pi,lamda0,lamda1)
        for j in range(y_pred.shape[0]):
            if y_pred[j] == 0 and y_test[j] == 0:
                c00 += 1
            elif y_pred[j] == 0 and y_test[j] == 1:
                c10 += 1
            elif y_pred[j] == 1 and y_test[j] == 0:
                c01 += 1
            else:
                c11 += 1


    print (c00,c01,c10,c11)

    avg_lamda0 = avg_lamda0 / 10
    avg_lamda1 = avg_lamda1 / 10
    x = np.linspace(1,54,num =54)
    fig, axs = plt.subplots(2)
    fig.suptitle('Average Possion Parameters in each Dimensions')
    axs[0].stem(x,avg_lamda0,use_line_collection=True)
    axs[0].set_xlabel("Dimension")
    axs[0].set_ylabel("Avg lamda value for y=0")
    axs[1].stem(x,avg_lamda1,use_line_collection=True)
    axs[1].set_xlabel("Dimension")
    axs[1].set_ylabel("Avg lamda value for y=1")
    plt.savefig("hw2b",dpi=300)

    #logistic regression
    steepest_ascent(group)
    print (newton(group))


if __name__ == '__main__':
    main()
