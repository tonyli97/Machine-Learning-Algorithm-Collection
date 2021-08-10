import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from IPython.display import display


class Adaboost():
    def __init__(self,X,y):
        self.X = X
        self.y = y
        self.n = X.shape[0]
        self.w = np.repeat(1/self.n,self.n).reshape(-1,1)

        self.w_avg = []
        self.boost_eps = []
        self.boost_alpha = []
        self.eps_upper_bound = []
        self.pred_errors = []

        self.eps_total = 0
        self.total_pred = 0

    def sample(self):
        self.bt = np.random.choice(self.n,self.n,replace=True,p=list(np.asarray(self.w).flatten()))

    def get_weights(self):
        X = self.X[self.bt,:]
        y = self.y[self.bt]
        self.weights = (np.linalg.inv(X.T @ X)) @ X.T @ y

    def predict(self):
        self.prediction = np.sign(self.X @ self.weights)
        self.eps = np.sum(self.w[~np.equal(self.y,self.prediction)])
        if self.eps>0.5:
            self.weights = -self.weights
            self.predict()

    def get_alpha(self):
        self.alpha = 0.5*np.log((1-self.eps)/self.eps)
        self.boost_eps.append(self.eps)
        self.boost_alpha.append(self.alpha)
        self.eps_total += (0.5-self.eps)**2
        upper_bound = np.exp(-2*self.eps_total)
        self.eps_upper_bound.append(upper_bound)

    def pred_error(self):
        self.total_pred += self.alpha * self.prediction
        pred_error = np.mean(np.sign(self.total_pred) != self.y)
        self.pred_errors.append(pred_error)


    def update_weights(self):
        self.w = self.w * np.exp(-self.alpha * self.y * self.prediction)
        #self.w_avg.append(np.mean(self.w))
        self.w = self.w / np.sum(self.w)




Prob1_X = pd.read_csv('./Prob1_X.csv',header=None)
Prob1_y = pd.read_csv('./Prob1_y.csv',header=None)


Prob1_X = Prob1_X.to_numpy()
Prob1_y = Prob1_y.to_numpy()

T=2500
n = Prob1_X.shape[0]
adaboost = Adaboost(Prob1_X,Prob1_y)
for t in range(T):
    adaboost.sample()
    adaboost.get_weights()
    adaboost.predict()
    adaboost.get_alpha()
    adaboost.pred_error()
    adaboost.update_weights()



plt.figure()
plt.plot(range(1,T+1),adaboost.pred_errors,'b')
plt.plot(range(1,T+1),adaboost.eps_upper_bound,'r')
xticks=np.linspace(1,T,6)
xlabel='Boosting Iteration'
ylabel='Error'
title='Train Errors and Upper Bound vs Boosting Iteration'
savfigname='hw3_1a_train_error.png'

plt.xticks(xticks)
plt.xlabel(xlabel)
plt.ylabel(ylabel)
plt.title(title)
plt.savefig(savfigname)
plt.show()
plt.close()


x = np.linspace(0, 1000, 1000)
plt.figure()
plt.stem(x, adaboost.w,linefmt='--',markerfmt='D')
plt.savefig("1bstem")
plt.show()
plt.close()


plt.figure()
plt.plot(range(1,T+1),adaboost.boost_eps)
plt.xticks(np.linspace(1,T,6))
plt.xlabel('Boosting Iters')
plt.ylabel('eps')
plt.title('eps vs Boosting Iters')
plt.savefig('hw3_1c_eps.png')
plt.show()
plt.close()



plt.figure()
plt.plot(range(1,T+1),adaboost.boost_alpha)
plt.xticks(np.linspace(1,T,6))
plt.xlabel('Boosting Iters')
plt.ylabel('Alpha')
plt.title('Alpha vs Boosting Iters')
plt.savefig('hw3_1c_alpha.png')
plt.show()
plt.close()




# 2.kmeans
#a
T = 20
n = 500
mix_weights = (0.2,0.5,0.3)
cov = np.matrix([[1,0],[0,1]])
mean1 = np.array([0,0])
mean2 = np.array([3,0])
mean3 = np.array([0,3])
gauss1 = np.random.multivariate_normal(mean1,cov,n)
gauss2 = np.random.multivariate_normal(mean2,cov,n)
gauss3 = np.random.multivariate_normal(mean3,cov,n)
choice = np.random.choice(range(3),size=n, p=mix_weights)
gen_data = np.concatenate((gauss1[choice==0,:],gauss2[choice==1,:],gauss3[choice==2,:]))
ks = range(2,6)
colors = ['red','green','black','yellow','blue']
Ks = [3,5]
clusters_tmp = []

def cluster_helper(row,centers):
    errors = np.sum((centers-row)**2,axis=1)
    #print (row)
    #print (errors)
    tmp = np.argmin(np.sum((centers-row)**2,axis=1))
    return tmp,errors[tmp]


for j in range(len(ks)):
    k = ks[j]
    objections = []
    cluster = None
    centers = np.random.uniform(low=0,high=1,size=(k,2))
    for t in range(T):
        cluster = np.apply_along_axis(cluster_helper,1,gen_data,centers)
        objections.append(np.sum(cluster[:,1]))
        for i in range(k):
            centers[i,:] = np.mean(gen_data[cluster[:,0]==i],axis=0)
    plt.plot(range(1,T+1),objections,colors[j])
    if ks[j] in [3,5]:
        clusters_tmp.append(cluster[:,0])

plt.xticks(range(1,T+1))
plt.xlabel('Iter')
plt.ylabel('Objective')
plt.title('Objective vs Iter for K = 2,3,4,5')
plt.legend(['K = %d'%i for i in ks])
plt.savefig('hw3_2a_kmean_obj')
plt.show()

plt.figure()
colors_arr = [colors[int(x)] for x in clusters_tmp[0]]
tmp_x = gen_data[:,0]
tmp_y = gen_data[:,1]
plt.scatter(tmp_x,tmp_y,c=colors_arr)
plt.xlabel('Dim 1')
plt.ylabel('Dim 2')
plt.title('Scatter plot with cluster assignment for K=3')
plt.savefig('hw3_2_b_3.png')
plt.show()

plt.figure()
colors_arr = [colors[int(x)] for x in clusters_tmp[1]]
tmp_x = gen_data[:,0]
tmp_y = gen_data[:,1]
plt.scatter(tmp_x,tmp_y,c=colors_arr)
plt.xlabel('Dim 1')
plt.ylabel('Dim 2')
plt.title('Scatter plot with cluster assignment for K=5')
plt.savefig('hw3_2_b_5.png')
plt.show()





#3 GMM

def EM(data, K, num_iter):
    cov_ind = np.cov(data.transpose())
    cov = np.array([cov_ind]*K)
    data_mean = np.array(data.describe().loc['mean',:])
    all_mean = np.random.multivariate_normal(data_mean, cov[0], K)
    pi_weight = np.ones(K)*np.array(1/K)
    n = data.shape[0]

    obj_acc = []
    for i in range(0,num_iter):
        phi_bottom = 0
        phi = [0] * K

        for j in range(0, K):
            phi_bottom += multivariate_normal.pdf(data,mean=all_mean[j],
                                    cov=cov[j], allow_singular=True)*pi_weight[j]

        for k in range(0, K):
            phi[k] = pi_weight[k] * multivariate_normal.pdf(data,mean=all_mean[k],
                                    cov=cov[k], allow_singular=True)/phi_bottom

        nk = np.zeros(K)
        for it in range(0,K):
            pi_k = np.zeros(K)
            nk[it] = np.sum(phi[it])
            pi_weight[it] = nk[it]/n

        for i in range(0,K):
            all_mean[i] = (1/nk[i])*np.matmul(np.matrix(phi[i].reshape(1,-1)), np.matrix(data))


        for i in range(0,K):
            x_mu1 = np.array(data)-all_mean[i]
            cov[i] = np.matmul(np.multiply(phi[i].reshape(-1,1), x_mu1).transpose(),x_mu1)/nk[i]

        obj_one = np.sum(np.log(phi_bottom))
        obj_acc.append(obj_one)

    return obj_acc, pi_weight, all_mean, cov





def GMM(data, K, num_iter, n_times):
    obj_acc_all = []
    pi_k_all = []
    all_init_mean_all = []
    init_cov_all = []
    for i in range(0, n_times):
        obj_acc, pi_weight, all_init_mean, init_cov = EM(data, K, num_iter)
        obj_acc_all.append(obj_acc)
        pi_k_all.append(pi_weight)
        all_init_mean_all.append(all_init_mean)
        init_cov_all.append(init_cov)
    return obj_acc_all, pi_k_all, all_init_mean_all,init_cov_all


def plot(obj_acc_all):
    x_axis = np.arange(5,31)
    plt.figure(figsize=(8,6))
    i = 0
    for item in obj_acc_all:
        item = item[4:]
        plt.plot(x_axis, item, label=i)
        i += 1
        plt.legend()
        plt.xlabel('iterations')
        plt.ylabel('Log Marginal Objective Function')



def max_objective(obj_acc_all):
    i = 0
    acc = []
    for item in obj_acc_all:
        acc.append(item[-1])
        m = max(acc)
        index_part = [i for i, j in enumerate(acc) if j == m]
        max_index = index_part[0]
    return max_index


def Bayes(data, K, best_weight_class1, best_mean_class1, best_cov_class1,
              best_weight_class0, best_mean_class0, best_cov_class0):
    class1 = 0
    for i in range(0,K):
        class1 += multivariate_normal.pdf(data,mean=best_mean_class1[i], cov=best_cov_class1[i],
                                                  allow_singular=True)*best_weight_class1[i]
    class0 = 0
    for i in range(0,K):
        class0 += multivariate_normal.pdf(data,mean=best_mean_class0[i], cov=best_cov_class0[i],
                                                      allow_singular=True)*best_weight_class0[i]
    class1 = class1.tolist()
    class0 = class0.tolist()
    pred_result = []
    for i in range(0, len(class1)):
        if class1[i]>=class0[i]:
            pred_result.append(1)
        else:
            pred_result.append(0)
    return pred_result

def confusion_matrix(pred_result, true_list):
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for i in range(0,len(true_list)):
        if pred_result[i]==1 and true_list[i]==1:
            TP += 1
        elif pred_result[i]==0 and true_list[i]==1:
            FN += 1
        elif pred_result[i]==1 and true_list[i]==0:
            FP += 1
        elif pred_result[i]==0 and true_list[i]==0:
            TN += 1

    data = [('TP:'+str(TP), 'FP:'+str(FP)),('FN:'+str(FN), 'TN:'+str(TN))]
    df = pd.DataFrame(data)
    df = df.rename({0: 'Predicted Postive', 1: 'Predicted Negative'}, axis='index')
    df = df.rename({0: 'Actual Postive', 1: 'Actual Negative'}, axis='columns')
    accuracy = (TP+TN)/(TP+TN+FP+FN)
    display(df)
    print('Accuracy:'+ str(accuracy))




def part3a(X_train0, X_train1):
    obj_acc_all_1, pi_weight_1, all_mean_all_1,cov_all_1 = GMM(X_train1, 3, 30,10)
    plot(obj_acc_all_1)
    plt.title('Log Marginal Objective Function vs Iters for Class 1')
    plt.savefig('p3a1')


    obj_acc_all_0, pi_weight_0, all_mean_all_0,cov_all_0 = GMM(X_train0, 3, 30,10)
    plot(obj_acc_all_0)
    plt.title('Log Marginal Objective Function vs Iters for Class 0')
    plt.savefig('p3a2')




def part3b(X_train0, X_train1):
    for t in range(1,5):
        obj_acc_all_1, pi_weight_1, all_init_mean_all_1,init_cov_all_1 = GMM(X_train1, t, 30,10)
        obj_acc_all_0, pi_weight_0, all_init_mean_all_0,init_cov_all_0 = GMM(X_train0, t, 30,10)


        ind_1 = max_objective(obj_acc_all_1)
        best_weight_1 = pi_weight_1[ind_1]
        best_mean_1 = all_init_mean_all_1[ind_1]
        best_cov_1 = init_cov_all_1[ind_1]
        ind_0 = max_objective(obj_acc_all_0)
        best_weight_0 = pi_weight_0[ind_0]
        best_mean_0 = all_init_mean_all_0[ind_0]
        best_cov_0 = init_cov_all_0[ind_0]
        pred_result1 = Bayes(X_test, t, best_weight_1, best_mean_1, best_cov_1,
                      best_weight_0, best_mean_0, best_cov_0)

        confusion_matrix(pred_result1,y_test.iloc[:,0].tolist())


def init_data(X_train,y_train,X_test,y_test):
    train_total = pd.concat([X_train, y_train], axis=1, sort=False)
    train1 = train_total[train_total.iloc[:,-1]==1]
    train0 = train_total[train_total.iloc[:,-1]==0]
    X_train1 = train1.iloc[:,0:10]
    X_train0 = train0.iloc[:,0:10]
    y_train1 = train1.iloc[:,-1]
    y_train0 = train0.iloc[:,-1]
    return X_train0,y_train0,X_train1,y_train1


X_train = pd.read_csv('Prob3_Xtrain.csv', header=None)
y_train = pd.read_csv('Prob3_ytrain.csv', header=None)
X_test = pd.read_csv('Prob3_Xtest.csv', header=None)
y_test = pd.read_csv('Prob3_ytest.csv', header=None)


X_train0,y_train0,X_train1,y_train1 = init_data(X_train,y_train,X_test,y_test)
part3a(X_train0, X_train1)
part3b(X_train0, X_train1)
