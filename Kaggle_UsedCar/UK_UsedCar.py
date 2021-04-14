#  ------------------------------- Day 21 - 210408 실습 -------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn 
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

#편의상 unclean.csv는 분석에서 제외
import sys
import os
dir = os.path.dirname(os.path.realpath(__file__))

audi = pd.read_csv(dir+'./data/audi.csv')
bmw = pd.read_csv(dir+'./data/bmw.csv')
cclass = pd.read_csv(dir+'./data/cclass.csv')
focus = pd.read_csv(dir+'./data/focus.csv')
ford = pd.read_csv(dir+'./data/ford.csv')
hyundi = pd.read_csv(dir+'./data/hyundi.csv')
merc = pd.read_csv(dir+'./data/merc.csv')
skoda = pd.read_csv(dir+'./data/skoda.csv')
toyota = pd.read_csv(dir+'./data/toyota.csv')
vauxhall = pd.read_csv(dir+'./data/vauxhall.csv')
vw = pd.read_csv(dir+'./data/vw.csv')

#브랜드별 파일 합치기 전 행 추가
audi['brand'] = 'audi'
bmw['brand'] = 'bmw'
cclass['brand'] = 'cclass'
focus['brand'] = 'focus'
ford['brand'] = 'ford'
hyundi['brand'] = 'hyundi'
merc['brand'] = 'merc'
skoda['brand'] = 'skoda'
toyota['brand'] = 'toyota'
vauxhall['brand'] = 'vauxhall'
vw['brand'] = 'vw'

#각 브랜드별 파일 df0으로 합치기
df0 = pd.concat([audi,bmw,cclass,focus,ford,hyundi,merc,skoda,toyota,vauxhall,vw])
print(int(len(df0)) - (    int(len(audi))+int(len(bmw))+int(len(cclass))+int(len(focus))+int(len(ford))+int(len(hyundi))+int(len(merc))+int(len(skoda))+int(len(toyota))+int(len(vauxhall))+int(len(vw))   )   )
print(df0)
print(df0.info())          # y = price / x = 10개
print(df0.describe())
print(df0.isna().sum())       #tax, tax(£), mpg 공란데이터 있음

#df0(원본)을 df로 복사 후 데이터 전처리 시작
df = df0.copy()

# 공란 데이터 처리
#tax, tax(£)    // tax0으로 통일 : tax는 그대로, tax(£) = x1.38 (환율고려)
df = df.fillna({'tax(£)':0})
df = df.fillna({'tax':0})
df.rename(columns={'tax(£)':'tax1'}, inplace=True)
df['tax0'] = df.apply(lambda x : x.tax + (x.tax1 * 1.38), axis='columns')
df = df.drop(columns = 'tax')
df = df.drop(columns = 'tax1')
#mpg    : 108540 row 중 9353개가 mpg데이터 없음(약10%) -> 분석에서 제외(mpg 결측치 있는 행 제거)
print(df.isna().sum())
df = df.dropna(axis=0)
print(df.isna().sum())        # 더이상 공란 없음

# # 컬럼별 값 점검
print(df['year'].value_counts())              #2060년 1개 데이터 삭제요망
print(df['transmission'].value_counts())      #Other 9개 데이터 삭제요망
print(df['fuelType'].value_counts())          #Other, Electric 253개 데이터 삭제요망

# #특정 데이터 삭제
df = df[df.year != 2060]
df = df[df.transmission != 'Other']
df = df[df.fuelType != 'Other']
df = df[df.fuelType != 'Electric']
df.tail()

# year데이터를 car_age로 변경
df['car_age'] = 2021 - df['year']
df = df.drop(columns=['year'])
print(df)       #분석대상 데이터 갯수 총 98925 row x 10columns (y = price /// x = 9개)



#########데이터 점검 끝##########



# object type 변수별 갯수 확인
figure, ((ax1, ax2),(ax3,ax4)) = plt.subplots(nrows=2, ncols=2)
figure.set_size_inches(30,30)
sns.countplot(df['model'], ax=ax1)              
sns.countplot(df['transmission'], ax=ax2)       
sns.countplot(df['fuelType'], ax=ax3)           
sns.countplot(df['brand'], ax=ax4)              
plt.show()


#상관관계 heatmap 분석 
corrMatt = df.corr()
print(corrMatt)
mask = np.array(corrMatt)
mask[np.tril_indices_from(mask)] = False
fig,ax = plt.subplots()
fig.set_size_inches(20,10)
sns.heatmap(corrMatt, mask=mask, vmax=.8,square=True, annot=True)
plt.show()  # mileage & car_age 높은 상관관계


# # 모든 변수에 대해 그래프 출력(전반적인 내용 확인)
figure, ( (ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9) ) = plt.subplots(nrows=3, ncols=3)
figure.set_size_inches(30,30)
sns.barplot(data=df, x="model", y="price", ax=ax1)            #모델명-price는 관계없어보임
sns.lineplot(data=df, x="car_age", y="price", ax=ax2)             #최신일수록 가격이 비싼 경향을 보이나 1970년대 모델 가격이 높은 점 확인필요(classic car라서 비쌀가능성도 있음)
sns.barplot(data=df, x="transmission", y="price", ax=ax3)     #가격이 manual < Auto < Semiauto 일수록 비싸짐
sns.lineplot(data=df, x="mileage", y="price", ax=ax4)          #mile(주행거리)이 클수록 가격 낮아짐
sns.barplot(data=df, x="fuelType", y="price", ax=ax5)         #디젤=하이브리드 비슷, 가솔린은 낮은 경향
sns.lineplot(data=df, x="mpg", y="price", ax=ax6)              #연비와 fueltype 관계있겠지?
sns.lineplot(data=df, x="engineSize", y="price", ax=ax7)
sns.barplot(data=df, x="brand", y="price", ax=ax8)
sns.lineplot(data=df, x="tax0", y="price", ax=ax9)
plt.show()


#mileage & car_age -> price 그래프
print(df['car_age'].value_counts())
plt.figure(figsize=(15,10),facecolor='w') 
sns.scatterplot(df["mileage"], df["price"], hue = df["car_age"])
plt.show()

sns.pairplot(df)
plt.show()

sns.scatterplot(df=df, x='', y='', hue='', size='price', sizes=(50,330))





################ Pre-processing for modeling ####################

df_expanded = pd.get_dummies(df)
print(df_expanded)

std = StandardScaler()
df_expanded_std = std.fit_transform(df_expanded)
df_expanded_std = pd.DataFrame(df_expanded_std, columns = df_expanded.columns)
print(df_expanded_std.shape)
print(df_expanded_std.head())

X_train, X_test, y_train, y_test = train_test_split(df_expanded_std.drop(columns = ['price']), df_expanded_std[['price']])
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)




















####################  여기부터 모르겠다  ###########################










from sklearn.feature_selection import SelectKBest, f_regression
#SelectKBest 모듈은 target 변수와 그외 변수 사이의 상관관계를 계산하여 가장 상관관계가 높은 변수 k개를 선정할 수 있는 모듈임
# f_regression 참고 : (https://woolulu.tistory.com/63)
#Linear model for testing the individual effect of each of many regressors. 
#This is a scoring function to be used in a feature selection procedure, not a free standing feature selection procedure.

column_names = df_expanded.drop(columns = ['price']).columns

no_of_features = []
r_squared_train = []
r_squared_test = []

for k in range(3, 40, 2):
    selector = SelectKBest(f_regression, k = k)
    X_train_transformed = selector.fit_transform(X_train, y_train)
    X_test_transformed = selector.transform(X_test)
    regressor = LinearRegression()
    regressor.fit(X_train_transformed, y_train)
    no_of_features.append(k)
    r_squared_train.append(regressor.score(X_train_transformed, y_train))
    r_squared_test.append(regressor.score(X_test_transformed, y_test))
    
sns.lineplot(x = no_of_features, y = r_squared_train, legend = 'full')
sns.lineplot(x = no_of_features, y = r_squared_test, legend = 'full')
# plt.show()

selector = SelectKBest(f_regression, k = 23)
X_train_transformed = selector.fit_transform(X_train, y_train)
X_test_transformed = selector.transform(X_test)
column_names[selector.get_support()]

def regression_model(model):
    """
    Will fit the regression model passed and will return the regressor object and the score
    """
    regressor = model
    regressor.fit(X_train_transformed, y_train)
    score = regressor.score(X_test_transformed, y_test)
    return regressor, score

model_performance = pd.DataFrame(columns = ["Features", "Model", "Score"])

models_to_evaluate = [LinearRegression()]

for model in models_to_evaluate:
    regressor, score = regression_model(model)
    model_performance = model_performance.append({"Features": "Linear","Model": model, "Score": score}, ignore_index=True)

model_performance

regressor = sm.OLS(y_train, X_train).fit()
print(regressor.summary())

X_train_dropped = X_train.copy()


while True:
    if max(regressor.pvalues) > 0.05:
        drop_variable = regressor.pvalues[regressor.pvalues == max(regressor.pvalues)]
        print("Dropping " + drop_variable.index[0] + " and running regression again because pvalue is: " + str(drop_variable[0]))
        X_train_dropped = X_train_dropped.drop(columns = [drop_variable.index[0]])
        regressor = sm.OLS(y_train, X_train_dropped).fit()
    else:
        print("All p values less than 0.05")
        break


print(regressor.summary())





############## Fitting on polynomial features
# 여기부터 공부안함