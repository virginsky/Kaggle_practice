
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn 
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.tree import DecisionTreeClassifier
import sys
import os
from Mushroom_preprocessing import df       # preprocessing.py파일 중 전처리완료한 데이터셋인 df 호출

# print(df)

'''
# 라벨인코딩 - 일단 이걸로 돌려보자 (나중에 종류가 많은 항목들은 원핫인코딩식으로?)
from sklearn.preprocessing import LabelEncoder
labelencoder=LabelEncoder()
for columns in df.columns:
    df[columns] = labelencoder.fit_transform(df[columns])
# print(df)   

# 데이터를 로딩하고, 학습과 테스트 데이터 셋으로 분리
X = df.drop(['class'], axis=1)
Y = df['class']
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, random_state=42) # stratify : target:
# print(X_train)
# print(X_test)

# DecisionTree Classifier 생성 및 학습
dt_clf = DecisionTreeClassifier(max_depth=7,random_state=0)
dt_clf.fit(X_train , y_train)
print("Accuracy on training set: {:.3f}".format(dt_clf.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(dt_clf.score(X_test, y_test)))

# export_graphviz()의 호출 결과로 out_file로 지정된 tree.dot 파일을 생성함. 
from sklearn.tree import export_graphviz
export_graphviz(dt_clf, out_file="tree.dot", impurity=True, filled=True)
'''



##########################################################################################


# tree로 의미해석이 안되서 X변수만 one-hot encoding 후 다시 분석 시작
df0 = df.copy()
# print(df0)
# print(df0['class'].value_counts())

xx = df0.drop(['class'], axis=1)
X = pd.get_dummies(xx)
from sklearn.preprocessing import LabelEncoder
labelencoder=LabelEncoder()
Y = labelencoder.fit_transform(df0['class'])
# print(X)
# print(Y)
# print(Y.value_counts())

X_train, X_test, y_train, y_test = train_test_split(
    X, Y, random_state=42) 
# print(X_train)
# print(X_test)

# DecisionTree Classifier 생성 및 학습
dt_clf = DecisionTreeClassifier(max_depth=7,random_state=0)
dt_clf.fit(X_train , y_train)
print("Accuracy on training set: {:.3f}".format(dt_clf.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(dt_clf.score(X_test, y_test)))

# export_graphviz()의 호출 결과로 out_file로 지정된 .dot 파일을 생성함. 
from sklearn.tree import export_graphviz
export_graphviz(dt_clf, out_file="tree3.dot", feature_names=X.columns, impurity=True, filled=True)
