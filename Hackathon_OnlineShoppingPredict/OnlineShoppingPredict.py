import numpy as np
import pandas as pd
import sklearn 
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import metrics
import sys
import os
import statistics

dir = os.path.dirname(os.path.realpath(__file__))

df = pd.read_csv(dir+'./train.csv')

# print(df)
# print(df.info())        # y = Monthly Amount항목을 만들어야 함 고객별 sum of monthly (quantity * price)
# print(df.describe())
# print(df.isna().sum())  # 공란데이터 없음



# sns.set_theme(style="whitegrid")
# ax = sns.violinplot(x=df.amount)
# plt.show()

print(df.order_date)
df['order_date'] = pd.to_datetime(df['order_date'], format='%Y-%m-%d %H:%M:%S', errors='raise')
df['year'] = df['order_date'].dt.year
df['month'] = df['order_date'].dt.month
# print(df.year)
# print(df.month)

df0 = df.copy() 
df0=df0.drop(columns = 'order_id')
df0=df0.drop(columns = 'product_id')
df0=df0.drop(columns = 'description')
df0=df0.drop(columns = 'quantity')
# df0=df0.drop(columns = 'order_date')
df0=df0.drop(columns = 'price')
print(df0)

# # 연도별,월별 total 합계 그래프 -> 별 의미가 없는듯..?
# accsum_total = df0.pivot_table(values='total', index='month', columns='year', aggfunc=sum)
# ax=accsum_total.plot()
# plt.show()
print(df0.pivot_table(values='total', index='month', columns='year', aggfunc=sum))
'''
y = 2011-12 id별 total합계가 300을 넘냐(1), 안넘냐(0)
x = country, 전년도 12월 total계(2009-12 / 2010-12), 1개월전 total계, 2개월전 total계, 3개월전 total계 ... 10개월전 total계 (이것도 다 300넘냐안넘냐로 단순화?)
그럼 데이터구성이
id  country     year    month   t.total
'''