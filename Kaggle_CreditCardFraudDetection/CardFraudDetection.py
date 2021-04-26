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
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
import sys
import os
import statistics

dir = os.path.dirname(os.path.realpath(__file__))
df = pd.read_csv(dir+'./creditcard.csv')

# print(df)               # 
# print(df.info())        # 
# print(df.describe())    # 
# print(df.isna().sum())  # 공란 없음
print(df['Class'].value_counts())   
# # 0 : 284,315개 / 1 : 492개 -> 예측을 0이라고 하면 99.8%의 Accurancy 획득
# # 원하는건 Class가 1(사기거래)인 것을 탐지 -> 이 데이터를 불려서 모델에 적용해야 할 듯







# # 일단 이대로 돌려보자(df / xgboost결과)
# # 오차행렬:
# #  [[71082     7]
# #  [   26    87]]

# # 정확도: 0.9995
# # 정밀도: 0.9255
# # 재현율: 0.7699
# # F1: 0.8406
# # AUC: 0.8849
# # 전반적인 모델성능자체는 나쁘지않지만, 이 분석의 핵심인 사기거래 탐지 성능(재현율)은 77%로 떨어짐
# # 정확도는 조금 더 떨어지더라도 실제 사기를 더 잘 예측할 수 있는 모델학습이 필요





########################################### Autoencoder ########################################
df1 = df.copy()
X = df1.drop(['Class'], axis=1)
y = df1['Class']

shuffle_index = np.random.permutation(len(df1))
X = X.values[shuffle_index]
y = y.values[shuffle_index]

n_train = int(len(X) * 0.3)


x_train = X[:n_train]
y_train = y[:n_train]
x_test = X[n_train:]
y_test = y[n_train:]

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)


# 오토인코더의 구조를 각 레이어(28 > 100 > 50 > 100 > 28)가 되도록 구성하였습니다. 
# 그럼 28개의 Feature 정보가 레이어를 통해 다시 원복되는 패턴을 학습하게 됩니다.
# 여기서 정상 데이터만을 넣어 학습시키게 합니다.
import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.keras.models as models

n_inputs = x_train.shape[1]
n_outputs = 2
n_latent = 50

inputs = tf.keras.layers.Input(shape=(n_inputs, ))
x = tf.keras.layers.Dense(100, activation='tanh')(inputs)
latent = tf.keras.layers.Dense(n_latent, activation='tanh')(x)

# Encoder
encoder = tf.keras.models.Model(inputs, latent, name='encoder')
encoder.summary()

latent_inputs = tf.keras.layers.Input(shape=(n_latent, ))
x = tf.keras.layers.Dense(100, activation='tanh')(latent_inputs)
outputs = tf.keras.layers.Dense(n_inputs, activation='sigmoid')(x)

# Decoder
decoder = tf.keras.models.Model(latent_inputs, outputs, name='decoder')
decoder.summary()

# 정상 데이터 만을 학습
x_train_norm = x_train[y_train == 0]

autoencoder = tf.keras.models.Model(inputs, decoder(encoder(inputs)))
autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.fit(x_train_norm, x_train_norm, epochs=15, batch_size = 100, validation_data=(x_test, x_test))





# 오토인코더의 인코더에 train 데이터를 넣은 결과로 나온 Latent Vector로 정상 거래, ddd
# 사기 거래 중 무엇인지 분류하도록 학습합니다.

encoded = encoder.predict(x_train)

classifier = tf.keras.Sequential([
    layers.Dense(32, input_dim=n_latent, activation='tanh'),
    layers.Dense(16, activation='relu'),
    layers.Dense(n_outputs, activation ='softmax')
])
classifier.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
classifier.summary()

classifier.fit(encoded, y_train, batch_size=100, epochs=10)



# Predict !
from sklearn.metrics import classification_report
pred_y = classifier.predict(encoder.predict(x_test)).argmax(axis=1)
y = y_test

print(classification_report(y, pred_y))




# confusion matrix 
cm1=confusion_matrix(y, pred_y)
plt.figure(figsize=(10,7))
sns.heatmap(cm1,annot=True,fmt='d')
plt.xlabel('predicted by model')
plt.ylabel('Truth')
plt.show()

# 최종 결론
accuracy = np.mean(np.equal(y,pred_y))
right = np.sum(y * pred_y == 1)
precision = right / np.sum(pred_y)
recall = right / np.sum(y)
f1 = 2 * precision*recall/(precision+recall)
print('Final apply model : ?')
print('accuracy : ',accuracy)
print('precision : ', precision)
print('recall : ', recall)
print('f1 : ', f1)
print('AUC : ', roc_auc_score(y,pred_y))

# 망..









# X = df.drop(['Class'], axis=1)
# y = df['Class']
# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42) 

# # DecisionTree Classifier 생성 및 학습
# clf = DecisionTreeClassifier(max_depth=5,random_state=0)
# clf_model = clf.fit(X_train , y_train)
# clf_y_pred = clf.predict(X_test)
# print("DecisionTree Accuracy on train set: {:.3f}".format(clf.score(X_train, y_train)))
# print("DecisionTree Accuracy on test set: {:.3f}".format(clf.score(X_test, y_test)))


# # randomforest 분석
# rf = RandomForestClassifier(n_estimators=100, max_depth=4, min_samples_split=5)
# rf_model = rf.fit(X_train, y_train)
# rf_y_pred = rf.predict(X_test)
# print("RandomForest Train score: ",rf.score(X_train, y_train))
# print("RandomForest Test score: ",rf.score(X_test, y_test))
# # print("RandomForest Test score: ",metrics.accuracy_score(y_test, rf_y_pred))  - 위에꺼랑 다른표현 같은결과값


# # AdaBoost classifier
# ada = AdaBoostClassifier(n_estimators=50,
#                          learning_rate=1,random_state = 7777)
# ada_model = ada.fit(X_train, y_train)
# ada_y_pred = ada_model.predict(X_test)  
# print("Ada boosting Train Accuracy:",ada.score(X_train, y_train))
# print("Ada boosting Test Accuracy:",metrics.accuracy_score(y_test, ada_y_pred))
# # print("Ada boosting Test Accuracy:",ada.score(X_test, y_test)) - 위에꺼랑 다른표현 같은결과값




# # XGBoost classifier
#       # https://wooono.tistory.com/97     - 설명참고
#       # https://injo.tistory.com/44       - 모델세팅 참고

# from xgboost import XGBClassifier
# import xgboost as xgb

# # 넘파이 형태의 학습 데이터 세트와 테스트 데이터를 DMatrix로 변환
# dtrain = xgb.DMatrix(data=X_train, label = y_train)
# dtest = xgb.DMatrix(data=X_test, label=y_test)

# # parameters 변경해가며 돌리기
# params = {'booster' : 'gbtree', 'max_depth' : 3,
#          'eta' : 0.1, 
#          'objective' : 'binary:logistic',
#          'eval_metric' : 'logloss',
#          'early_stoppings' : 100, 'lambda':10, 'colsample_bytree' : 0.5}
# num_rounds = 300

# # train 데이터 세트는 'train', evaluation(test) 데이터 세트는 'eval' 로 명기
# wlist = [(dtrain, 'train'), (dtest,'eval')]
# # 하이퍼 파라미터와 early stopping 파라미터를 train() 함수의 파라미터로 전달
# xgb_model = xgb.train(params = params, dtrain=dtrain, num_boost_round=num_rounds, evals=wlist)

# pred_probs = xgb_model.predict(dtest)
# print('predict() 수행 결과값을 10개만 표시, 예측 확률 값으로 표시됨')
# print(np.round(pred_probs[:10], 3))

# # 예측 확률이 0.5보다 크면 1, 그렇지 않으면 0으로 예측값 결정해 리스트 객체인 preds에 저장
# preds = [ 1 if x > 0.5 else 0 for x in pred_probs]
# print('예측값 10개만 표시: ', preds[:10])

# # 혼동행렬, 정확도, 정밀도, 재현율, F1, AUC 불러오기
# import warnings
# def get_clf_eval(y_test, y_pred):
#     confusion = confusion_matrix(y_test, y_pred)
#     accuracy = accuracy_score(y_test, y_pred)
#     precision = precision_score(y_test, y_pred)
#     recall = recall_score(y_test, y_pred)
#     F1 = f1_score(y_test, y_pred)
#     AUC = roc_auc_score(y_test, y_pred)
#     print('오차행렬:\n', confusion)
#     print('\n정확도: {:.4f}'.format(accuracy))
#     print('정밀도: {:.4f}'.format(precision))
#     print('재현율: {:.4f}'.format(recall))
#     print('F1: {:.4f}'.format(F1))
#     print('AUC: {:.4f}'.format(AUC))

# get_clf_eval(y_test, preds)
# '''
# from xgboost import plot_importance
# import matplotlib.pyplot as plt
# fig, ax = plt.subplots(figsize=(10, 12))
# plot_importance(xgb_model, ax=ax)

# xgb.plot_tree(xgb_model, num_trees=0, rankdir='LR')
# fig = plt.gcf()
# fig.set_size_inches(150, 100)
# # plt.show()
# '''


# # Light GBM
# from lightgbm import LGBMClassifier
# from lightgbm import plot_importance
# # from sklearn.datasets import load_breast_cancer
# # from sklearn.model_selection import train_test_split

# lgbm_wrapper = LGBMClassifier(n_estimators=400)
# evals = [(X_test, y_test)]
# lgbm_wrapper.fit(X_train, y_train, early_stopping_rounds=100, eval_metric='logloss', eval_set=evals, verbose=True)
# pred = lgbm_wrapper.predict(X_test)
# pred_proba = lgbm_wrapper.predict_proba(X_test)[:1]

# # fig, ax = plt.subplots(figsize=(10,12))
# # plot_importance(lgbm_wrapper, ax=ax)
# # plt.show()

# lgbm_wrapper_model = lgbm_wrapper.fit(X_train, y_train)
# lgbm_wrapper_y_pred = lgbm_wrapper.predict(X_test)
# print("Light GBM Train score: ",lgbm_wrapper.score(X_train, y_train))
# print("Light GBM Test score: ",lgbm_wrapper.score(X_test, y_test))



# # Cat Boost Classifier
# import catboost
# from catboost import CatBoostClassifier, Pool

# params = {'loss_function':'Logloss', # objective function
#           'eval_metric':'AUC', # metric
#           'verbose': 200, # output to stdout info about training process every 200 iterations
#           }
# cbc = CatBoostClassifier(**params)
# cbc.fit(X_train, y_train, # data to train on (required parameters, unless we provide X as a pool object, will be shown below)
#           eval_set=(X_test, y_test), # data to validate on
#           use_best_model=True, # True if we don't want to save trees created after iteration with the best validation score
#           );
# print("CatBoost Train score: ",cbc.score(X_train, y_train))
# print("CatBoost Test score: ",cbc.score(X_test, y_test))
# feature_importance_df = pd.DataFrame(cbc.get_feature_importance(prettified=True))
# print(feature_importance_df)
# plt.figure(figsize=(12, 6));
# sns.barplot(x="Importances", y="Feature Id", data=feature_importance_df);
# plt.title('CatBoost features Importances:');
# plt.show()