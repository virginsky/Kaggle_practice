import numpy as np
import pandas as pd
import sklearn 
import statsmodels.api as sm
import matplotlib.pyplot as plt
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

dir = os.path.dirname(os.path.realpath(__file__))
df = pd.read_csv(dir+'./train.csv')

# print(df)               # 
# print(df.info())        # datetime 유형 바꿔줘야 함 / 러쉬아워 구분 신규변수? / 휴일 구분? / 픽드랍 위도경도 차이의 절대값의 합을 새로운 변수로 생성(단순 거리위주의 분석) / 지역에 따른 구분도 필요하려나(상습 트래픽지역?)
# print(df.describe())    # 
# print(df.isna().sum())  # 공란데이터 없음


# <05.26 탐색적 데이터 분석>
# (하)타겟 변수를 확인하고 이상치가 존재하는지 파악하시오
# 이상치를 확인하고, plot을 하시오. 
# 정규분포 형식으로 바꾸어서 plot을 하시오.
print(df["trip_duration"].describe())

# 2. (중)Data에서 pickup_datetime을 날짜 형식으로 바꾸시오.
# pickup_datetime 을 month, day, weekday, hour, dayoweek으로 바꾸어서 출력하시오.
# 3. (상) 거리변수에 대해서 측정하시오
# 위도와 경도에 따라서 거리를 계산해보고, 그 거리가 틀리다면, 새로운 좌표계를 도입하여서 출력하시오.
# 4. 범주형 변수를 원 핫 인코딩을 하시오.
# 데이터 내에 범주형 변수가 있으면, 원 핫 인코딩을 하고 출력하시오.(train,test에 모두 적용하시오.)
# 또한 원래 데이터에서 drop하시오.
# 5.  데이터에 이상치가 있다면 처리하고 이유를 밝히시오.
# 데이터에 이상치가 있다면 처리하고 이유를 명백히 밝히시오.