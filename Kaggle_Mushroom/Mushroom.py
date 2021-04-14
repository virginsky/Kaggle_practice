#  ------------------------------- Day 23 - 210412 실습 -------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn 
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
import sys
import os
dir = os.path.dirname(os.path.realpath(__file__))

df0 = pd.read_csv(dir+'./mushrooms.csv')
df = df0.copy()
print(df)
print(df.info())        # y = class (e=edible, p=poisonous) / 21개의 변수x
print(df.describe())
print(df.isna().sum())  # 공란데이터 없음


# # 라벨인코딩이라는 것은 이름으로 되어있는 것을 숫자로 혹은 그 반대로 변환하는 것입니다.
    # # onehotencoding -> 앞뒤 순서 있을때 보전, matrix 증가
        # ex) 자연어 처리 쪽
    # # Labelencoder -> label만 변환해줌 -> 앞뒤 특성을 잃어버림
        # -> 머신러닝은 이게 적합 -> matrix = data가 크다는 것 -> 속도 급감, 혹은 shape이 안맞음
from sklearn.preprocessing import LabelEncoder
Labelencoder = LabelEncoder()
for col in df.columns:
  df[col] = Labelencoder.fit_transform(df[col]) # column별로 label Encoder 적용
print(df)


# # 컬럼별 유니크한 요소들이 몇개나 있나 확인
count_var = []

for col in df.columns:
  count_var.append(df[col].unique().sum())    # series로 접근 -> columns으로 접근했기에?
size = np.arange(len(count_var))

fig = plt.figure(figsize=(15,10))
ax = fig.add_subplot(1,1,1,facecolor = 'r')
ax.bar(size,count_var,color = 'k')
ax.set(title = 'Unique elements per column',
       ylabel = 'No of unique elements',
       xlabel = 'Features')
plt.show()


# 상관관계 분석
df.corr() 
plt.figure(figsize=(10,10))
sns.heatmap(df.corr(),cmap='inferno',square=True)
plt.show()


# target / 변수 지정
target = df['class']
train = df.drop('class',axis = 1)


# 분포확인..?
fig=plt.figure(figsize = (15,10))
ax=fig.add_subplot(1,1,1,facecolor='blue')
pd.value_counts(target).plot(kind='bar', cmap = 'cool')
plt.title("Class distribution")
plt.show()


########################## 여기서부터 모르겠음 #################################

def sigmoid(theta,X):               # logistic regression = sigmoid (S자 형식)
    X = np.array(X)
    theta = np.asarray(theta)
    return((1/(1+math.e**(-X.dot(theta)))))


# Function for the cost function of the logistic regression.
def cost(theta, X, Y):
    first = np.multiply(-Y, np.log(sigmoid(theta,X)))
    second = np.multiply((1 - Y), np.log(1 - sigmoid(theta,X)))
    return np.sum(first - second) / (len(X))


# It calculates the gradient of the log-likelihood function.
def log_gradient(theta,X,Y):
    first_calc = sigmoid(theta, X) - np.squeeze(Y).T
    final_calc = first_calc.T.dot(X)
    return(final_calc.T)


# This is the function performing gradient descent.
def gradient_Descent(theta,X,Y,itr_val,learning_rate=0.00001):
    cost_iter=[]
    cost_val=cost(theta,X,Y)
    cost_iter.append([0,cost_val])
    change_cost = 1
    itr = 0
    while(itr < itr_val):
        old_cost = cost_val
        theta = theta - (0.01 * log_gradient(theta,X,Y))
        cost_val = cost(theta,X,Y)
        cost_iter.append([i,cost])
        itr += 1
    return theta


def pred_values(theta,X,hard=True):
    X = (X - np.mean(X,axis=0))/np.std(X,axis=0)
    pred_prob = sigmoid(theta,X)
    pred_value = np.where(pred_prob >= .5 ,1, 0)
    return pred_value


theta = np.zeros((train.shape)[1])
theta = np.asmatrix(theta)
theta = theta.T
target = np.asmatrix(target).T
y_test = list(target)


import math
params = [10,20,30,50,100]
for i in range(len(params)):
    th = gradient_Descent(theta,train,target,params[i])
    y_pred = list(pred_values(th, train))
    score = float(sum(1 for x,y in zip(y_pred,y_test) if x == y)) / len(y_pred)
    print("The accuracy after " + '{}'.format(params[i]) + " iterations is " + '{}'.format(score))


from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()
clf.fit(train, target)
clf.score(train, target)