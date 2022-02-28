import datetime
import numpy as np
import pandas as pd
from matplotlib import cm, pyplot as plt
from hmmlearn.hmm import GaussianHMM,MultinomialHMM,GMMHMM
import xlrd
workbook = xlrd.open_workbook("bit.xls")  # 文件路径
worksheet=workbook.sheet_by_name("Sheet1")
l1 = []
l2 = []
nrows=worksheet.nrows
for i in range(nrows): #循环打印每一行
    l1.append('d'+str(int(worksheet.row_values(i)[0])))
    l2.append(float(worksheet.row_values(i)[1]))
a = int(len(l2) * 0.7) - 1
lst = []
while a + 1 < len(l2):
    X_Test = [l2[:a]]
    X_Pre = [l2[a:]]
    X_Test = np.array(X_Test).reshape(-1,1)
    X_Pre = np.array(X_Pre).reshape(-1,1)
    model = GaussianHMM(n_components=5, covariance_type='diag',algorithm='viterbi', n_iter=20)
    model.fit(X_Test)
    expected_returns_volumes = np.dot(model.transmat_, model.means_)
    expected_returns = expected_returns_volumes[:,0]
    hidden_states = model.predict(X_Pre[0].reshape(-1,1)) #将预测的第一组作为初始值
    current_price = l2[a-1]
    n = current_price + expected_returns[hidden_states] - 1300
    #current_price = predicted_price[i]
    lst.append(n[0])
    print(n[0],expected_returns[hidden_states],l2[a-1],l2[a])
    a += 1
print(lst)
#获得输入数据
'''
#数据集的划分
X_Test = [l2[:-30]]
#X_Test = np.array([ i  for i in l2[:-30]])
X_Pre = [l2[-30:]]
X_Test = np.array(X_Test).reshape(-1,1)
X_Pre = np.array(X_Pre).reshape(-1,1)
#模型的搭建
print(X_Test.shape)
model = GaussianHMM(n_components=3, covariance_type='diag',algorithm='viterbi', n_iter=20000)
#model = MultinomialHMM(n_components=3,n_iter=1000, tol=0.01)
#model = GMMHMM(n_components=3,n_mix=3,covariance_type='diag',n_iter=1000)
model.fit(X_Test)
print("隐藏状态的个数", model.n_components)  #
print("均值矩阵")
print(model.means_)
print("协方差矩阵")
print(model.covars_)
print("状态转移矩阵--A")
print(model.transmat_)
#训练数据的隐藏状态划分
expected_returns_volumes = np.dot(model.transmat_, model.means_)
expected_returns = expected_returns_volumes[:,0]
predicted_price = []  #预测值
current_price = l2[-30]
for i in range(len(X_Pre)):
    hidden_states = model.predict(X_Pre[i].reshape(-1,1)) #将预测的第一组作为初始值
    print(expected_returns[hidden_states])
    predicted_price.append(current_price + expected_returns[hidden_states])
    #current_price = predicted_price[i]
print(np.array(predicted_price).shape)
x = l1[-29: ]
y_act = l2[-29:]
print(current_price)
y_pre = predicted_price[:-1]
plt.figure(figsize=(8,6))
plt.plot_date(x, y_act,linestyle="-",marker="None",color='g')
plt.plot_date(x, y_pre,linestyle="-",marker="None",color='r')
plt.legend(['Actual', 'Predicted'])
plt.show()'''