import numpy as np
import pandas as pd
import sklearn 
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets
from sklearn import metrics
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import sys
import os
import statistics
from scipy.stats import norm
import scipy.stats


dir = os.path.dirname(os.path.realpath(__file__))
df = pd.read_csv(dir+'./train.csv')

# print(df)               # 
# print(df.info())        # datetime 유형 바꿔줘야 함 / 러쉬아워 구분 신규변수? / 휴일 구분? / 픽드랍 위도경도 차이의 절대값의 합을 새로운 변수로 생성(단순 거리위주의 분석) / 지역에 따른 구분도 필요하려나(상습 트래픽지역?)
# print(df.describe())    # 
# print(df.isna().sum())  # 공란데이터 없음

'''
# <05.26 탐색적 데이터 분석>
# (하)타겟 변수를 확인하고 이상치가 존재하는지 파악하시오
# 이상치를 확인하고, plot을 하시오. 
# 정규분포 형식으로 바꾸어서 plot을 하시오.
# print(df["trip_duration"].describe())
df['trip_duration_min'] = df["trip_duration"]/60
plt.figure(figsize=(12,8))
sns.boxplot(df["trip_duration_min"])
plt.show()
plt.figure(figsize=(8,5))
sns.distplot(np.log(df["trip_duration_min"].values)).set_title("Distribution of Trip Duration")
plt.title("Distribution of trip duration (min) in Log Scale")
plt.show()
'''


# 2. (중)Data에서 pickup_datetime을 날짜 형식으로 바꾸시오.
# pickup_datetime 을 month, day, weekday, hour, dayoweek으로 바꾸어서 출력하시오.
# print(df.info())
df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'], format='%Y-%m', errors='raise')
df['dropoff_datetime'] = pd.to_datetime(df['dropoff_datetime'], format='%Y-%m', errors='raise')
# print(df.info())

# print(df['pickup_datetime'].dt.month)
# print(df['pickup_datetime'].dt.day)
# print(df['pickup_datetime'].dt.weekday)
# print(df['pickup_datetime'].dt.hour)
# print(df['pickup_datetime'].dt.dayofweek)




# 3. (상) 거리변수에 대해서 측정하시오
# 위도와 경도에 따라서 거리를 계산해보고, 그 거리가 틀리다면, 새로운 좌표계를 도입하여서 출력하시오.
def uclidean(lat1, lng1, lat2, lng2):
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    R = 6371.0
    lat = lat2 - lat1
    lng = lng2 - lng1
    d = np.sin(lat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng * 0.5) ** 2
    h = 2 * R * np.arcsin(np.sqrt(d))
    return h

df['uclidean'] = uclidean(df.pickup_latitude, df.pickup_longitude, df.dropoff_latitude, df.dropoff_longitude)
df['manhatan'] = (abs(df.dropoff_longitude - df.pickup_longitude) + abs(df.dropoff_latitude - df.pickup_latitude)) * 113.2
# print(df['uclidean'])
# print(df['uclidean'].describe())
# print(df['manhatan'])
# print(df['manhatan'].describe())


# 4. 범주형 변수를 원 핫 인코딩을 하시오.
# 데이터 내에 범주형 변수가 있으면, 원 핫 인코딩을 하고 출력하시오.(train,test에 모두 적용하시오.)
# 또한 원래 데이터에서 drop하시오.
print(df['store_and_fwd_flag'].value_counts())
df = pd.get_dummies(df['store_and_fwd_flag'], columns = ['store_and_fwd_flag'], prefix='store_and_fwd_flag')
print(df)




# 5.  데이터에 이상치가 있다면 처리하고 이유를 밝히시오.
# 데이터에 이상치가 있다면 처리하고 이유를 명백히 밝히시오.




# <05.26 머신러닝 기반 데이터 분석>
# Train과 Test를 분리하고, 실제 분석에 필요한 부분만 남기고 drop하시오.
# (왜 drop을 했는지 명백히 밝히시오)
# 2. 2가지 이상의 머신러닝 모델을 적용해보고 비교해보시오.
# 어떠한 것이 좋은지 밝히시오.
# 3. 적용해 본 모델의 하이퍼 파라미터 최적화 하시오.
# 적용 해 본 모델의 하이퍼 파라미터를 최적화 하시오.(변수 2개 이상)
# 4. OLS 방법을 적용해보시오.
# 적용한 코드와 summary도 밝히고 결과를 해석하시오.
# 5. Feature importance or Engineering 방법을 해석하고 모델에 적용하시오.
# Feature Importance or engineering 방법을 사용하여, 모델을 해석하거나 적용하시오.





# <05.26 빅데이터 시각화>
# Corrlation matrix를 출력하고 결과 해석하시오.
# 위도, 경도에 따라 plot 을 그리시오.
# hint : Implot이나 follium
# 3. Pick_up의 hour에 따라 seaborn plot을 그리시오.
# Pick_up (hour)에 따라 seaborn plot을 그려보고 결과를 해석하시오.
# 4. 택시의 이동경로를 plot 하고 결과 해석하시오.
# 5. 택시의 속도를 plot하고 결과를 해석하시오.