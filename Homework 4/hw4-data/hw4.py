import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigs

data = pd.read_csv("CFB2019_scores.csv", header = None)
names_pd = pd.read_csv('TeamNames.txt',sep="\n", header=None)
names_pd.columns = ['Team_Name']
M = np.zeros((769, 769))
data = data.to_numpy()
names = names_pd.to_numpy()


for i in range(data.shape[0]):
    row = data[i]
    a_index, a_points, b_index, b_points = row[0], row[1], row[2], row[3]
    a = a_index-1
    b = b_index-1
    if a_points > b_points:
        M[a,a] += 1 + a_points/(a_points + b_points)
        M[b,a] += 1 + a_points/(a_points + b_points)
        M[b,b] += b_points/(a_points + b_points)
        M[a,b] += b_points/(a_points + b_points)
    elif a_points < b_points:
        M[a,a] += a_points/(a_points + b_points)
        M[b,a] += a_points/(a_points + b_points)
        M[b,b] += 1 + b_points/(a_points + b_points)
        M[a,b] += 1 + b_points/(a_points + b_points)
    else:
        continue

M_norm = M/np.sum(M, axis=1).reshape(-1,1)


def rank_teams(t):
    w0 = np.ones(769)*np.array(1/769).reshape(1,-1)
    wt = w0 @ np.linalg.matrix_power(M_norm,t)
    d = {'index':np.arange(1,770), 'wt':wt.tolist()[0]}
    df = pd.DataFrame(d)
    df_temp = df.sort_values(by=['wt'],ascending=False)
    result_pd = pd.concat([df_temp, names_pd], axis=1, join='inner').head(25)
    return result_pd


#P1 a)

print (rank_teams(10))
print ("--------------------------------------")
print ("--------------------------------------")
print (rank_teams(100))
print ("--------------------------------------")
print ("--------------------------------------")
print (rank_teams(1000))
print ("--------------------------------------")
print ("--------------------------------------")
print (rank_teams(10000))


#P1 b)
T = 10000
eigen_value, eigen_vector = eigs(M_norm.T,1)
eigen_vector = eigen_vector.flatten()
w_inf = eigen_vector/np.sum(eigen_vector)
diffs = []
w0 = np.ones(769)*np.array(1/769).reshape(1,-1)
wt = w0 @ M_norm
diffs.append(np.sum(abs(wt-w_inf)))
for t in range(T-1):
    wt = wt @ M_norm
    diffs.append(np.sum(abs(wt - w_inf)))

plt.figure()
plt.plot(range(1,T+1),diffs)

plt.xticks(np.linspace(1,T,9))
plt.xlabel('t')
plt.ylabel('1-norm between w_inf and w_t')
plt.title('1-norm between w_inf and w_t for t=1,...,10000')
plt.savefig('hw4_1b.png')
#plt.show()


#P2 a)
T = 100
nyt = pd.read_csv('nyt_data.txt',sep='\n',header=None)
X = np.zeros((3012,8447))
W = np.random.uniform(1,2,(3012,25))
H = np.random.uniform(1,2,(25,8447))
d_index = 0
for i in range(nyt.shape[0]):
    row = nyt.iloc[i,:].tolist()[0].split(',')
    for r in row:
        one_word = r.split(':')
        n = int(one_word[0])
        m = int(one_word[1])
        X[n-1, d_index] = m
    d_index += 1


obj_lst = []
for i in range(T):
    temp2 = X/(W @ H +1e-16)
    W_norm = W.T / np.sum(W.T, axis=1).reshape(-1,1)
    H = H * (W_norm @ temp2)

    temp2 = X/(W @ H +1e-16)
    H_norm = H.T / np.sum(H.T, axis=0)
    W = W * (temp2 @ H_norm)

    obj = np.sum(np.log(1/(W@H+1e-16)) * X + W@H)
    obj_lst.append(obj)



plt.figure()
plt.plot(range(1,T+1),obj_lst)
plt.xticks(np.linspace(1,T,10))
plt.xlabel('Iters')
plt.ylabel('Objective')
plt.title('Objective vs Iters')
plt.savefig('hw4_2a.png')

# 2b)

W_NMF_norm = W / np.sum(W, axis=0)
word = pd.read_csv('nyt_vocab.dat', header=None)
word.columns = ['word']
df_important_word = pd.DataFrame(W_NMF_norm)

acc_alltable = []
k = df_important_word.shape[1]
for i in range(k):
    d = {'weight':df_important_word.iloc[:,i].tolist()}
    df = pd.DataFrame(d)
    df = df.sort_values(by='weight', ascending=False).iloc[0:10,:]
    df2 = pd.concat([df, word], axis=1, join='inner').head(25)
    acc_alltable.append(df2)
    print ('---------------------------------------------------------')
    print (i)
    print (df2)
