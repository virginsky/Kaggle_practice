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

df = pd.read_csv(dir+'./cardio_train.csv', sep= ';')


# print(df)
# print(df.info())        # y = cardio (0=음성, 1=양성) / 12개 변수(id제외 시 11개)
# print(df.describe())
# print(df.isna().sum())  # 공란데이터 없음
# print(df['cardio'].value_counts())

# # # 모든 변수에 대해 그래프 출력(전반적인 내용 확인)
# figure, ( (ax1, ax2, ax3),(ax4, ax5,ax6) ) = plt.subplots(nrows=2, ncols=3)
# figure.set_size_inches(30,30)
# sns.countplot(data=df, hue="cardio", x="gender", ax=ax1)      # 1 - women, 2 - men      
# sns.countplot(data=df, hue="cardio", x="cholesterol", ax=ax2)      # 1: normal, 2: above normal, 3: well above normal       
# sns.countplot(data=df, hue="cardio", x="gluc", ax=ax3)     #1: normal, 2: above normal, 3: well above normal
# sns.countplot(data=df, hue="cardio", x="smoke", ax=ax4)          
# sns.countplot(data=df, hue="cardio", x="alco", ax=ax5)         
# sns.countplot(data=df, hue="cardio", x="active", ax=ax6)      
# plt.show()

# figure, ( (ax1, ax2, ax3),(ax4, ax5, ax6) ) = plt.subplots(nrows=2, ncols=3)
# figure.set_size_inches(30,30)
# sns.countplot(data=df, hue="cardio", x="age", ax=ax1)            
# sns.countplot(data=df, hue="cardio", x="height", ax=ax2)        # 100cm 미만도 믿어야되나? 아이들? 나이 최소값은 10798=29살.. 특이값일 가능성이 높아보이는데
# sns.countplot(data=df, hue="cardio", x="weight", ax=ax3)        # good
# sns.countplot(data=df, hue="cardio", x="ap_hi", ax=ax4)         # 특이값 처리필요_수축기혈압
# sns.countplot(data=df, hue="cardio", x="ap_lo", ax=ax5)         # 특이값 처리필요_이완기혈압
# plt.show()
# print(df['age'].value_counts())
# print(df['height'].value_counts())
# print(df['weight'].value_counts())
# print(df['ap_hi'].value_counts())
# print(df['ap_lo'].value_counts())

# id 컬럼삭제
df0 = df.copy()
df0 = df0.drop('id', axis=1)

# age(day)를 age(year)로 변경 후 범주화
def preprocessing_age(x):
      x = x/365
      return int(x)
df0['age']=df['age'].apply(preprocessing_age)

df0.loc[df0['age']<40, 'age_bin'] = 0 
df0.loc[(df0['age'] >= 40) & (df0['age'] < 50), 'age_bin'] = 1 
df0.loc[(df0['age'] >= 50) & (df0['age'] < 60), 'age_bin'] = 2
df0.loc[df0['age'] >= 60, 'age_bin'] = 3
df0=df0.drop('age', axis=1)
# print(df0)


# # 이상치 처리 - ap

# plt.figure(figsize=(15,10),facecolor='w') 
# sns.scatterplot(df["ap_hi"], df["ap_lo"], hue = df["cardio"])   
# plt.show()

# 앞2개자리만 따오고 0번째가 1,2인 경우 x10
def preprocessing_ap(x):
      x = str(abs(x))
      x = x[:2]
      if x[0] == '1' or x[0] == '2':
            x = int(x) * 10
      else:
            x = int(x)
      return x
df0['ap_lo1'] = df0['ap_lo'].apply(preprocessing_ap)
df0['ap_hi1'] = df0['ap_hi'].apply(preprocessing_ap)
# print(df0.describe())  # 25% / 75%로 자르기엔 폭이 너무 작아져서(80~90 / 120~140), 임의로 50이하인 값만 평균으로 변경
df0['ap_lo1'] = np.where(df0['ap_lo1']<50, int(df0['ap_lo1'].mean()), df0['ap_lo1'])
df0['ap_hi1'] = np.where(df0['ap_hi1']<50, int(df0['ap_hi1'].mean()), df0['ap_hi1'])
# print(df0.describe())

# ap_lo>ap_hi인 경우 값 바꿔주기 
df0['minus']= df0['ap_hi1'] - df0['ap_lo1']
df0['ap_lo2'] = np.where(df0['minus']<0, df0['ap_hi1'], df0['ap_lo1'])
df0['ap_hi2'] = np.where(df0['minus']<0, df0['ap_lo1'], df0['ap_hi1'])
# print(df0.describe())

# 필요없는 컬럼 제거
df0=df0.drop(columns = 'ap_lo')
df0=df0.drop(columns = 'ap_lo1')
df0=df0.drop(columns = 'ap_hi')
df0=df0.drop(columns = 'ap_hi1')
df0=df0.drop(columns = 'minus')
# print(df0.info()) 


# # 이상치 처리 - 키,몸무게 
# BMI 지수 생성 : 몸무게(kg) / 키(m) 의 제곱
df0['bmi'] = round(df0['weight'] / (df0['height']/100)**2 , 1)
df0=df0.drop(columns = 'weight')
df0=df0.drop(columns = 'height')
# print(df0.describe())

# BMI 이상치는 평균값으로 치환
Q1 = df0['bmi'].describe()['25%']
Q3 = df0['bmi'].describe()['75%']
IQR = Q3-Q1
min_bmi = Q1 - 1.5*IQR
max_bmi = Q3 + 1.5*IQR
# print(min_bmi)
# print(max_bmi)
df0['BMI'] = np.where(df0['bmi'] < min_bmi, int(df0['bmi'].mean()), (np.where(df0['bmi'] > max_bmi, int(df0['bmi'].mean()), df0['bmi'])))
# print(df0.describe())
df0=df0.drop(columns = 'bmi')

# #새로생성한 변수들 그래프 확인
# figure, ( (ax1, ax2),(ax3,ax4)  ) = plt.subplots(nrows=2, ncols=2)
# figure.set_size_inches(30,30)
# sns.countplot(data=df0, hue="cardio", x="age_bin", ax=ax1)  
# sns.countplot(data=df0, hue="cardio", x="ap_lo2", ax=ax2)   
# sns.countplot(data=df0, hue="cardio", x="ap_hi2", ax=ax3)   
# sns.countplot(data=df0, hue="cardio", x="BMI", ax=ax4)          
# plt.show()





##################################### 분석시작 ###################################

X = df0.drop(['cardio'], axis=1)
y = df0['cardio']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42) 
# print(X_train.shape)
# print(X_test.shape)


# DecisionTree Classifier 생성 및 학습
dt_clf = DecisionTreeClassifier(max_depth=10,random_state=0)
dt_clf.fit(X_train , y_train)
print("DecisionTree Accuracy on training set: {:.3f}".format(dt_clf.score(X_train, y_train)))
print("DecisionTree Accuracy on test set: {:.3f}".format(dt_clf.score(X_test, y_test)))



# randomforest 분석
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)
rf_pre=rf.predict(X_test)
print("RandomForest score: ",rf.score(X_test,y_test))





# Ada boosting

# Create adaboost classifer object
abc = AdaBoostClassifier(n_estimators=50,
                         learning_rate=1,random_state = 7777)
# Train Adaboost Classifer
model = abc.fit(X_train, y_train)
# Predict the response for test dataset
y_pred = model.predict(X_test)
# Model Accuracy, how often is the classifier correct?
print("Ada boosting Accuracy:",metrics.accuracy_score(y_test, y_pred))