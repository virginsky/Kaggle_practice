import numpy as np
import pandas as pd
import sklearn 
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
import sys
import os
import statistics
dir = os.path.dirname(os.path.realpath(__file__))

df = pd.read_csv(dir+'./cardio_train.csv', sep= ';')

'''
# print(df)
# print(df.info())        # y = cardio (0=?, 1=?) / 12개 변수(id제외 시 11개)
# print(df.describe())
# print(df.isna().sum())  # 공란데이터 없음
# print(df['cardio'].value_counts())

# # 모든 변수에 대해 그래프 출력(전반적인 내용 확인)
figure, ( (ax1, ax2, ax3),(ax4, ax5,ax6) ) = plt.subplots(nrows=2, ncols=3)
figure.set_size_inches(30,30)
sns.countplot(data=df, hue="cardio", x="gender", ax=ax1)      # 1 - women, 2 - men      
sns.countplot(data=df, hue="cardio", x="cholesterol", ax=ax2)      # 1: normal, 2: above normal, 3: well above normal       
sns.countplot(data=df, hue="cardio", x="gluc", ax=ax3)     #1: normal, 2: above normal, 3: well above normal
sns.countplot(data=df, hue="cardio", x="smoke", ax=ax4)          
sns.countplot(data=df, hue="cardio", x="alco", ax=ax5)         
sns.countplot(data=df, hue="cardio", x="active", ax=ax6)      
plt.show()

figure, ( (ax1, ax2, ax3),(ax4, ax5, ax6) ) = plt.subplots(nrows=2, ncols=3)
figure.set_size_inches(30,30)
sns.countplot(data=df, hue="cardio", x="age", ax=ax1)            
sns.countplot(data=df, hue="cardio", x="height", ax=ax2)        # 100cm 미만도 믿어야되나? 아이들? 나이 최소값은 10798=29살.. 특이값일 가능성이 높아보이는데
sns.countplot(data=df, hue="cardio", x="weight", ax=ax3)        # good
sns.countplot(data=df, hue="cardio", x="ap_hi", ax=ax4)         # 특이값 처리필요_수축기혈압
sns.countplot(data=df, hue="cardio", x="ap_lo", ax=ax5)         # 특이값 처리필요_이완기혈압
plt.show()
print(df['age'].value_counts())
print(df['height'].value_counts())
print(df['weight'].value_counts())
print(df['ap_hi'].value_counts())
print(df['ap_lo'].value_counts())
'''

# # 특이값 처리

# plt.figure(figsize=(15,10),facecolor='w') 
# sns.scatterplot(df["ap_hi"], df["ap_lo"], hue = df["cardio"])   
#   # 50이하나 200이상은 측정오류로 간주 drop? 중간값으로 치환?(평균은 너무 높아질듯)
# plt.show()
print(statistics.median(df['ap_hi']))
# print(statistics.median(df['ap_lo']))


# # plt.scatter(data=df, x='height', y='weight')
# plt.figure(figsize=(15,10),facecolor='w') 
# sns.scatterplot(df["height"], df["weight"], hue = df["cardio"])   
#   #평균의 x2 이상이나 0.5이하면 남녀 평균으로 치환
# plt.show()

# def preprocess(df):
#     # def replace_height_outlier(df, lower_bound, upper_bound):
#     #     df = 1gender평균의 x2거나 0.5인 경우 1grooup height평균으로 replace
#     #     df = 2gender평균의 x2거나 0.5인 경우 2grooup height평균으로 replace
#     def replace_ap_outlier(df):
#         out = df[ df[ap_hi] > 200 or df[ap_hi] < 50  ]
#         statistics.median(df['ap_hi']) = out
#         return df
# return df

df.replace({'ap_hi':>200}, 120)

plt.figure(figsize=(15,10),facecolor='w') 
sns.scatterplot(df["ap_hi"], df["ap_lo"], hue = df["cardio"])   
  # 50이하나 200이상은 측정오류로 간주 drop? 중간값으로 치환?(평균은 너무 높아질듯)
plt.show()


# group_h = df['height'].groupby(df['gender'])
# group_w = df['weight'].groupby(df['gender'])
# print(group_h.mean())
# print(group_w.mean())

'''
# 변수간 상관관계 분석을 해보자 - age/height/weight는 강한 관계가 있을것으로 예상
df0=df.drop(columns = 'cardio')
df0=df.drop(columns = 'id')
corrMatt = df0.corr()
print(corrMatt)
mask = np.array(corrMatt)
mask[np.tril_indices_from(mask)] = False
fig,ax = plt.subplots()
fig.set_size_inches(30,10)
sns.heatmap(corrMatt, mask=mask, vmax=1.0,square=True, annot=True)
plt.show()          # 예상과달리 크게 상관관계가 없네..? 성인대상이라 그런가봄? 여튼 ㄱ
'''

