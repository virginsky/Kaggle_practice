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

# print(df.order_date)
df['order_date'] = pd.to_datetime(df['order_date'], format='%Y-%m', errors='raise')
# print(df.info())
# df['year'] = df['order_date'].dt.year
# df['month'] = df['order_date'].dt.month
# print(df.year)
# print(df.month)

df0 = df.copy() 
df0=df0.drop(columns = 'order_id')
df0=df0.drop(columns = 'product_id')
df0=df0.drop(columns = 'description')
df0=df0.drop(columns = 'quantity')
# df0=df0.drop(columns = 'order_date')
df0=df0.drop(columns = 'price')
# print(df0)

# # 연도별,월별 total 합계 그래프 -> 9월부터 소비가 증가해 11월 최대치, 12월은 떨어짐(그룹화는 필요없을듯)
# accsum_total = df0.pivot_table(values='total', index='month', columns='year', aggfunc=sum)
# ax=accsum_total.plot(marker = 'o')
# plt.show()
# print(df0.pivot_table(values='total', index='month', columns='year', aggfunc=sum))

# # year-month 묶어서(yyyy-mm) 그래프 확인
df0['yyyy-mm'] = df['order_date'].dt.strftime('%Y-%m')
'''
print(df0['yyyy-mm'])
# yyyy-mm & country total 합계 그래프 -> 40개 국가.. 생각보다 많네
accsum_total = df0.pivot_table(values='total', index='yyyy-mm', columns='country', aggfunc=sum)
ax=accsum_total.plot(marker = 'o')
plt.show()
print(df0.pivot_table(values='total', index='yyyy-mm', columns='country', aggfunc=sum))

y = 2011-12 id별 total합계가 300을 넘냐(1), 안넘냐(0)
x = country, 전년도 12월 total계(2009-12 / 2010-12), 1개월전 total계, 2개월전 total계, 3개월전 total계 ... 10개월전 total계 (이것도 다 300넘냐안넘냐로 단순화?)
그럼 데이터구성이 이걸 한번 만들고
id  country     yyyy-mm   t.total

12월 데이터를 예측하는 모델(x : 2009-12 ~ 2010-11, country // y : 2010-12)을 만들고
그 모델을 2011년 데이터를 적용해서 2011년 12월 예측해보자
id  country     1m.total    2m.total    3m.total    4m.total    ... 12m.total? 단순 ox로? 아님 숫자로?
'''
# print(df0)

# id별 yyyy-mm total계로 데이터 rows를 줄여야 함 -> 24개 yyyy-mm을 컬럼으로 만들고 그 밑에 id별 합계
# print(df0.pivot_table(values='total', index='customer_id', columns='yyyy-mm', aggfunc=sum))
df1 = df0.pivot_table(values='total', index='customer_id', columns='yyyy-mm', aggfunc=sum)
df1 = df1.fillna(0)
df_p = df1.iloc[:,12:]
df_t = df1.iloc[:,:13]

df_t['y'] = np.where(df_t['2010-12']>300, 1, 0)        # 0 : 300이하 / 1: 300초과
df_t=df_t.drop(columns = '2010-12')

df_t.columns = ['12m_before','11m_before','10m_before','09m_before','08m_before','07m_before','06m_before','05m_before','04m_before','03m_before','02m_before','01m_before','y']
df_p.columns = ['12m_before','11m_before','10m_before','09m_before','08m_before','07m_before','06m_before','05m_before','04m_before','03m_before','02m_before','01m_before']
# print(df_t)     # y = 0 or 1 / x = 나머지컬럼 전부
# print(df_p)     # 현재 x만 있음, 이를 토대로 y컬럼 새로 만들어서 300 이상,이하 맞춰야함




##################################### 분석시작 ###################################

X = df_t.drop(['y'], axis=1)
y = df_t['y']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42) 
# print(X_train.shape)
# print(X_test.shape)
# print(y_train.shape)
# print(y_test.shape)


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
ada_model = abc.fit(X_train, y_train)
# Predict the response for test dataset
y_pred = ada_model.predict(X_test)
# Model Accuracy, how often is the classifier correct?
print("Ada boosting Accuracy:",metrics.accuracy_score(y_test, y_pred))




'''
# XG boosting (XGBClassfier?)
      # https://wooono.tistory.com/97     - 설명참고
      # https://injo.tistory.com/44       - 모델세팅 참고

from xgboost import XGBClassifier
import xgboost as xgb

# 넘파이 형태의 학습 데이터 세트와 테스트 데이터를 DMatrix로 변환하는 예제
dtrain = xgb.DMatrix(data=X_train, label = y_train)
dtest = xgb.DMatrix(data=X_test, label=y_test)

# max_depth = 3, 학습률은 0.1, 예제가 이진분류이므로 목적함수(objective)는 binary:logistic(이진 로지스틱)
# 오류함수의 평가성능지표는 logloss
# 부스팅 반복횟수는 400
# 조기중단을 위한 최소 반복횟수는 100

params = {'booster' : 'gbtree', 'max_depth' : 6,
         'eta' : 0.1, 
         'objective' : 'binary:logistic',
         'eval_metric' : 'logloss',
         'early_stoppings' : 100, 'lambda':10, 'colsample_bytree' : 0.5}

num_rounds = 400

# train 데이터 세트는 'train', evaluation(test) 데이터 세트는 'eval' 로 명기
wlist = [(dtrain, 'train'), (dtest,'eval')]
# 하이퍼 파라미터와 early stopping 파라미터를 train() 함수의 파라미터로 전달
xgb_model = xgb.train(params = params, dtrain=dtrain, num_boost_round=num_rounds, evals=wlist)

pred_probs = xgb_model.predict(dtest)
print('predict() 수행 결과값을 10개만 표시, 예측 확률 값으로 표시됨')
print(np.round(pred_probs[:10], 3))

# 예측 확률이 0.5보다 크면 1, 그렇지 않으면 0으로 예측값 결정해 리스트 객체인 preds에 저장
preds = [ 1 if x > 0.5 else 0 for x in pred_probs]
print('예측값 10개만 표시: ', preds[:10])

# 혼동행렬, 정확도, 정밀도, 재현율, F1, AUC 불러오기
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
import warnings
def get_clf_eval(y_test, y_pred):
    confusion = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    F1 = f1_score(y_test, y_pred)
    AUC = roc_auc_score(y_test, y_pred)
    print('오차행렬:\n', confusion)
    print('\n정확도: {:.4f}'.format(accuracy))
    print('정밀도: {:.4f}'.format(precision))
    print('재현율: {:.4f}'.format(recall))
    print('F1: {:.4f}'.format(F1))
    print('AUC: {:.4f}'.format(AUC))

get_clf_eval(y_test, preds)

from xgboost import plot_importance
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 12))
plot_importance(xgb_model, ax=ax)

xgb.plot_tree(xgb_model, num_trees=0, rankdir='LR')
fig = plt.gcf()
fig.set_size_inches(150, 100)

plt.show()

'''





################################### 모델 적용 #####################################3

# adaboosting 모델 적용 (ada_model : Ada boosting Accuracy: 0.9046653144016227) 

# Finalize model
import pickle


# Save model to disk
filename = 'Final_Model.sav'
pickle.dump(ada_model, open(filename, 'wb'))

# Load model from disk and use it to make new predictions
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(X_test, y_test)
print(result)

# Load test dataset
final_predict = df_p.copy()
X_train = final_predict
pred = ada_model.predict(X_train)
print(pred)

# 예측값 채우기

# print(df_p)



