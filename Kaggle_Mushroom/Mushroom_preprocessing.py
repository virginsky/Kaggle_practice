
# import numpy as np
import pandas as pd
# import sklearn 
# import statsmodels.api as sm
# from sklearn.linear_model import LinearRegression,Ridge,Lasso
# from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
# import sys
import os
dir = os.path.dirname(os.path.realpath(__file__))

df = pd.read_csv(dir+'./mushrooms.csv')
# print(df)
# print(df.info())        # y = class (e=edible, p=poisonous) / 22개의 변수
# print(df.describe())
# print(df.isna().sum())  # 공란데이터 없음


''' 의미없는 분석이었음 - 이런 문자열데이터는 상관관계x

# 상관관계 확인(heatmap)
corrMatt = df.corr()
print(corrMatt)
mask = np.array(corrMatt)
mask[np.tril_indices_from(mask)] = False
fig,ax = plt.subplots()
fig.set_size_inches(30,10)
sns.heatmap(corrMatt, mask=mask, vmax=1.0,square=True, annot=True)
plt.show()      # gill-attachment  and   veil-color 이 매우 높은 상관관계를 보임(0.89)
                머리 안쪽 갓? : gill-attachment: attached=a,descending=d,free=f,notched=n
                버섯 머리(?) 컬러 : veil-color: brown=n,orange=o,white=w,yellow=y
일반적인 상식?에서 컬러가 독버섯을 구분하는데 중요하다고 하기에 gill-attachment는 분석에서 제외
'''




# # # 모든 변수에 대해 그래프 출력(전반적인 내용 확인)
# import matplotlib.pyplot as plt
# import seaborn as sns
# figure, ( (ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9) ) = plt.subplots(nrows=3, ncols=3)
# figure.set_size_inches(30,30)
# sns.countplot(data=df, hue="class", x="cap-shape", ax=ax1)            
# sns.countplot(data=df, hue="class", x="cap-surface", ax=ax2)             
# sns.countplot(data=df, hue="class", x="cap-color", ax=ax3)     
# sns.countplot(data=df, hue="class", x="bruises", ax=ax4)          
# sns.countplot(data=df, hue="class", x="odor", ax=ax5)         
# sns.countplot(data=df, hue="class", x="gill-attachment", ax=ax6)       # 삭제예정
# sns.countplot(data=df, hue="class", x="gill-spacing", ax=ax7)
# sns.countplot(data=df, hue="class", x="gill-size", ax=ax8)
# sns.countplot(data=df, hue="class", x="gill-color", ax=ax9)
# plt.show()

# figure, ( (ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9) ) = plt.subplots(nrows=3, ncols=3)
# figure.set_size_inches(30,30)
# sns.countplot(data=df, hue="class", x="stalk-shape", ax=ax1)           # 삭제예정
# sns.countplot(data=df, hue="class", x="stalk-root", ax=ax2)            # 삭제예정 8414개 중 2480의 missing data 존재 (? 로 표기되어 not-null에 안뜸)
# print(df['stalk-root'].value_counts()) 
# sns.countplot(data=df, hue="class", x="stalk-surface-above-ring", ax=ax3)     
# sns.countplot(data=df, hue="class", x="stalk-surface-below-ring", ax=ax4)          
# sns.countplot(data=df, hue="class", x="stalk-color-above-ring", ax=ax5)         
# sns.countplot(data=df, hue="class", x="stalk-color-below-ring", ax=ax6)       
# sns.countplot(data=df, hue="class", x="veil-type", ax=ax7)              # 삭제예정
# sns.countplot(data=df, hue="class", x="veil-color", ax=ax8)             # 삭제예정
# sns.countplot(data=df, hue="class", x="ring-number", ax=ax9)
# plt.show()

# figure, ( (ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9) ) = plt.subplots(nrows=3, ncols=3)
# figure.set_size_inches(30,30)
# sns.countplot(data=df, hue="class", x="ring-type", ax=ax1)            
# sns.countplot(data=df, hue="class", x="spore-print-color", ax=ax2)             
# sns.countplot(data=df, hue="class", x="population", ax=ax3)     
# sns.countplot(data=df, hue="class", x="habitat", ax=ax4)          
# plt.show()



# 변수 그래프 확인결과 영향이 낮을것으로 판단되는 변수 제외 
df = df.copy()
df = df.drop(columns = 'gill-attachment')
df = df.drop(columns = 'stalk-shape')
df = df.drop(columns = 'veil-type')
df = df.drop(columns = 'veil-color')
df = df.drop(columns = 'stalk-root')
# print(df.info()) # 17개 변수.. 여전히 많지만.. ㄱㄱ


# # 라벨인코딩 
# from sklearn.preprocessing import LabelEncoder
# labelencoder=LabelEncoder()
# for columns in df.columns:
#     df[columns] = labelencoder.fit_transform(df[columns])
# # print(df)   



# 분석 시작
