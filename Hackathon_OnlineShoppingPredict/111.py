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
import os
import statistics

dir = os.path.dirname(os.path.realpath(__file__))
# dir = os.path.dirname(os.path.realpath(__file__))
df = pd.read_csv(dir+'./train.csv')
# print(df)               # order_id, product_id, description, Qty, price 분석에 필요x
# print(df.info())        # order date type이 시분초까지 다 나와있는 object -> 조절 필요
# print(df.describe())    # Qty 및 total에 마이너스값이 있지만 환불일 것으로 추정(고객별 합계 확인해서 더블체크 필요)
# print(df.isna().sum())  # 공란데이터 없음
# ## order date 형식 변경(datetime 사용)
df['order_date'] = pd.to_datetime(df['order_date'], format='%Y-%m', errors='raise')
# print(df['order_date'])
# print(df.info())        #object -> datetime64로 변경됨
#  9월부터 소비가 증가해 11월 최대치, 12월은 떨어짐(연간 비슷한 추세를 보임)
#  2011년 12월의 소비를 예측하는 것이기에, 전체 월별실적을 변수로 하는 것 보다는 
#    "고객별 2009-12 ~ 2010-11을 변수 / 2010-12 total 계가 300이 넘는지를 y로 모델 학습 및 테스트"
#    "해당 모델을 동일기간인 2010-12 ~ 2011-11을 변수로 하는 데이터에 적용하여 2011-12를 예측"
#     하여 year, month컬럼은 만들 필요가 없는고 yyyy-mm의 형태 데이터 필요
# ####################### 데이터 전처리(?) 시작 #################################
# df를 df0로 복사 후 필요없는 컬럼 삭제
df0 = df.copy() 
df0 = df0.drop(columns = ['order_id', 'product_id', 'description', 'quantity', 'price'])
# print(df0)
# # yyyy-mm 항목 생성
df0['yyyy-mm'] = df['order_date'].dt.strftime('%Y-%m')
# print(df0['yyyy-mm'])
#  -> 40개 국가.. 추구하는게 고객id별 12월 예측이기에 국가가 큰 영향을 미치지 않을것이라고 추정
#  -> 우선 분석에서 제외 후 모델정확도가 떨어지면 그때 다시 고려
# 2009-12 ~ 2011-11 id별 total계를 구성하는 새로운 데이터셋 df1 생성(pivot활용)
df1 = df0.pivot_table(values='total', index='customer_id', columns='yyyy-mm', aggfunc=sum)
# print(df1)    # [780502 rows x 5 columns] -> [5914 rows x 24 columns]
df1 = df1.fillna(0)     # NaN값은 의사결정트리 분석이 작동하지 않기에 0으로 채움
# #생성한 모델을 적용할 데이터셋(x : 2009-12 ~ 2010-11 / y : 2010-12)
df_t = df1.iloc[:,:13]  
df_t['y'] = np.where(df_t['2010-12']>300, 1, 0)     # 0 : 300이하 / 1: 300초과
df_t=df_t.drop(columns = '2010-12')
# # print(df_t)
df_t.columns = ['12m_before','11m_before','10m_before','09m_before','08m_before','07m_before','06m_before','05m_before','04m_before','03m_before','02m_before','01m_before','y']
# print(df_t)
# #model생성을 위한 데이터셋(x : 2010-12 ~ 2011-11 / y : 아직없음)
df_p = df1.iloc[:,12:]  
# print(df_p)
df_p.columns = ['12m_before','11m_before','10m_before','09m_before','08m_before','07m_before','06m_before','05m_before','04m_before','03m_before','02m_before','01m_before']
#Machine Learning Algorithm (MLA) Selection and Initialization
from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
#Common Model Helpers
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn import feature_selection
from sklearn import model_selection
from sklearn import metrics
X = df_t.drop(['y'], axis=1)
y = df_t['y']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42) 
MLA = [
    #Ensemble Methods
    ensemble.AdaBoostClassifier(),
    ensemble.BaggingClassifier(),
    ensemble.ExtraTreesClassifier(),
    ensemble.GradientBoostingClassifier(),
    ensemble.RandomForestClassifier(),
    ensemble.AdaBoostClassifier(),
    #GLM
    linear_model.LogisticRegressionCV(),
    linear_model.RidgeClassifierCV(),
    linear_model.SGDClassifier(),
    #Nearest Neighbor
    KNeighborsClassifier(),
    #Trees    
    tree.DecisionTreeClassifier(),
    tree.ExtraTreeClassifier(),
    XGBClassifier(),
    LGBMClassifier(),
    CatBoostClassifier()]
#split dataset in cross-validation with this splitter class: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.ShuffleSplit.html#sklearn.model_selection.ShuffleSplit
#note: this is an alternative to train_test_split
cv_split = model_selection.ShuffleSplit(n_splits = 10, test_size = .3, train_size = .6, random_state = 0 ) # run model 10x with 60/30 split intentionally leaving out 10%
#create table to compare MLA metrics
MLA_columns = ['MLA Name', 'MLA Parameters','MLA Train Accuracy Mean', 'MLA Test Accuracy Mean', 'MLA Test Accuracy 3*STD' ,'MLA Time']
MLA_compare = pd.DataFrame(columns = MLA_columns)
#create table to compare MLA predictions
MLA_predict = y
print('모델 작동전',X.shape,y.shape)
#index through MLA and save performance to table
row_index = 0
for alg in MLA:
    print('for 안에서모델 작동전',X.shape,y.shape)
    #set name and parameters
    MLA_name = alg.__class__.__name__
    MLA_compare.loc[row_index, 'MLA Name'] = MLA_name
    MLA_compare.loc[row_index, 'MLA Parameters'] = str(alg.get_params())
    print('for 안에서모델 작동전1',X.shape,y.shape)
    print('MLA 작동전 shape',MLA_predict.shape)
#     #score model with cross validation: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html#sklearn.model_selection.cross_validate
    cv_results = model_selection.cross_validate(alg, X_train, y_train, cv  = cv_split,return_train_score=True)
    print(alg)
    print('for 안에서모델 작동후',X.shape,y.shape)
    print('MLA 작동후 shape', MLA_predict.shape)
    MLA_compare.loc[row_index, 'MLA Time'] = cv_results['fit_time'].mean()
    MLA_compare.loc[row_index, 'MLA Train Accuracy Mean'] = cv_results['train_score'].mean()
    MLA_compare.loc[row_index, 'MLA Test Accuracy Mean'] = cv_results['test_score'].mean()   
    #if this is a non-bias random sample, then +/-3 standard deviations (std) from the mean, should statistically capture 99.7% of the subsets
    MLA_compare.loc[row_index, 'MLA Test Accuracy 3*STD'] = cv_results['test_score'].std()*3   #let's know the worst that can happen!
    #save MLA predictions - see section 6 for usage
    alg.fit(X_train,y_train)
    MLA_predict[MLA_name] = alg.predict(X_test)
    row_index+=1
MLA_compare.sort_values(by = ['MLA Test Accuracy Mean'], ascending = False, inplace = True)
print(MLA_compare)