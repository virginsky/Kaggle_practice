import numpy as np
import pandas as pd
import sklearn 
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans, DBSCAN
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, IsolationForest
from sklearn.naive_bayes import GaussianNB
from sklearn.covariance import EllipticEnvelope
# An object for detecting outliers in a Gaussian distributed dataset.
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets
from sklearn import metrics
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, plot_confusion_matrix
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
# from IPython.display import display
import sys
import os
import statistics

dir = os.path.dirname(os.path.realpath(__file__))
heart = pd.read_csv(dir+'./heart.csv')

# print(heart)               # 
# print(heart.info())        # 
# print(heart.describe())    # 
# print(heart.isna().sum())  # 공란데이터 없음

# # print(heart.describe().columns)
# heart['sex'] = heart['sex'].map({0:'female',1:'male'})
# heart['chest_pain_type'] = heart['cp'].map({3:'asymptomatic', 1:'atypical_angina', 2:'non_anginal_pain', 0:'typical_angina'})
# heart['fbs'] = heart['fbs'].map({0:'less_than_120mg/ml',1:'greater_than_120mg/ml'})
# heart['restecg'] = heart['restecg'].map({0:'normal',1:'ST-T_wave_abnormality',2:'left_ventricular_hypertrophy'})
# heart['exang'] = heart['exang'].map({0:'no',1:'yes'})
# heart['slope'] = heart['slope'].map({0:'upsloping',1:'flat',2:'downsloping'})
# heart['thal'] = heart['thal'].map({1:'fixed_defect',0:'normal',2:'reversable_defect'})
# heart['target'] = heart['target'].map({0:'no_disease', 1:'has_disease'})
# # print(heart)  
# # print(heart.isna().sum())

# categorical = [i for i in heart.loc[:,heart.nunique()<=10]]
# # print(categorical)
# continuous = [i for i in heart.loc[:,heart.nunique()>=10]]
# print(continuous)





'''
# Styling - customcolor
cust_palt = [
    '#111d5e', '#c70039', '#f37121', '#ffbd69', '#ffc93c'
]
plt.style.use('ggplot')


def dist(df, cols, hue = None, row=3,columns=3):
    fig,axes = plt.subplots(row,columns,figsize=(16,12))
    axes = axes.flatten()

    for i,j in zip(df[cols].columns, axes):
        sns.countplot(x = i, data = df, hue = hue, ax = j, orient = df[i].value_counts().index)
        j.set_title(f'{str(i).capitalize()} Distribution')
        total = float(len(df[i]))

        for p in j.patches:
            height = p.get_height()
            j.text(p.get_x()+p.get_width() / 2, height/2, '{:1.2f}%'.format((height/total)*100),ha = 'center')
        
        plt.tight_layout()

# dist(heart,categorical)
# plt.show()
# # 그래프로 알 수 있는 것
# #     남성이 여성보다 더 많이 관찰 됨
# #     Cp(가슴통증)이 있는 경우가 50%이상 관측이 됨.
# #     fbs(공복시 혈당) 이 연관이 없어 보임
# #     resting electrocardiographic results (values 0,1,2) (안정 심선도 결과) -> 좌심실 비대 사이에 고르게 분포.
# #     oldpeak = ST depression induced by exercise - relative to rest (비교적 안정되기까지 운동으로 유발되는 ST) -> 환자의 약 67%는 운동으로 인한 협심증이 없음.
# #     the slope of the peak exercise ST segment number of major vessels (0-3) colored by flourosopy (최대 운동 ST segment의 기울이(0=하강,1 = 평면, 2= 상승) -> 운동 경사는 주로 오르막과 평평한 것으로 나눠짐



import matplotlib.gridspec as gridspec
fig = plt.figure(constrained_layout=True,figsize = (16,12))
#A grid layout to place subplots within a figure.
grid = gridspec.GridSpec(ncols = 6, nrows= 3, figure = fig)

ax1 = fig.add_subplot(grid[0, :2])
#trestbps : 안정시 혈압
ax1.set_title('Trestbps Distribution')

sns.distplot(heart[continuous[1]], hist_kws={'rwidth':0.85, 'edgecolor':'blue','alpha':0.7},color = cust_palt[0])

ax12 = fig.add_subplot(grid[0, 2:3])

ax12.set_title('Trestbps')

sns.boxplot(heart[continuous[1]], orient='v')

ax2 = fig.add_subplot(grid[0,3:5])
#chol : 콜레스테롤 수치
ax2.set_title('Cholr Distribution')

sns.distplot(heart[continuous[2]], hist_kws= {'rwidth':0.85, 'edgecolor':'blue','alpha':0.7},color = cust_palt[1])

ax22 = fig.add_subplot(grid[0,5:])

ax22.set_title('Cholr')

sns.boxplot(heart[continuous[2]], orient='v',color = cust_palt[1])

ax3 = fig.add_subplot(grid[0,:2])
#thalach : 최대 심박수
ax3.set_title('thalach Distribution')

sns.distplot(heart[continuous[3]], hist_kws= {'rwidth':0.85, 'edgecolor':'blue','alpha':0.7},color = cust_palt[2])

ax32 = fig.add_subplot(grid[0,2:3])

ax32.set_title('thalach')

sns.boxplot(heart[continuous[3]], orient='v',color = cust_palt[2])


ax4 = fig.add_subplot(grid[0,3:5])
#oldpeak : 휴식에 비해 운동으로 인해 유발된 ST 우울증 ? 
ax4.set_title('oldpeark Distribution')

sns.distplot(heart[continuous[4]], hist_kws= {'rwidth':0.85, 'edgecolor':'blue','alpha':0.7},color = cust_palt[3])

ax42 = fig.add_subplot(grid[0,5:])

ax42.set_title('thalach')

sns.boxplot(heart[continuous[4]], orient='v',color = cust_palt[3])


ax5 = fig.add_subplot(grid[2, :4])

ax5.set_title('Age Distribution')

sns.distplot(heart[continuous[0]],
                 hist_kws={
                 'rwidth': 0.95,
                 'edgecolor': 'black',
                 'alpha': 0.8},
                 color=cust_palt[4])

ax55 = fig.add_subplot(grid[2, 4:])

ax55.set_title('Age')

sns.boxplot(heart[continuous[0]], orient='h', color=cust_palt[4])

# plt.show()


# 단변량(univariate) analysis
# Categoical VS Target
dist(heart,categorical[:-1],'target',4,2)
# plt.show()


fig = plt.figure(constrained_layout=True,figsize = (16,12))
#A grid layout to place subplots within a figure.
grid = gridspec.GridSpec(ncols = 6, nrows= 3, figure = fig)

ax1 = fig.add_subplot(grid[0, :2])

ax1.set_title('trestbps Distribution')

sns.boxplot(x = 'target',y = 'trestbps',data = heart,palette = cust_palt[2:],ax = ax1)

sns.swarmplot(x = 'target',y = 'trestbps',data = heart, palette = cust_palt[:2],ax = ax1)


ax2 = fig.add_subplot(grid[0,2:])

ax2.set_title('chol Distribution')

sns.boxplot(x = 'target', y = 'chol',data = heart, palette = cust_palt[:2],ax = ax2)

sns.swarmplot(x = 'target', y = 'chol',data = heart, palette = cust_palt[:2],ax = ax2)

ax3 = fig.add_subplot(grid[1,2:])

ax3.set_title('thalach Distribution')


sns.boxplot(x = 'target', y = 'thalach',data = heart, palette = cust_palt[:2],ax = ax3)

sns.swarmplot(x = 'target', y = 'thalach',data = heart, palette = cust_palt[:2],ax = ax3)

ax4 = fig.add_subplot(grid[1,2:])

ax4.set_title('st_depression Distribution')

sns.boxplot(x = 'target', y = 'oldpeak',data = heart, palette = cust_palt[:2],ax = ax4)

sns.swarmplot(x = 'target', y = 'oldpeak',data = heart, palette = cust_palt[:2],ax = ax4)

ax5 = fig.add_subplot(grid[2,:])

ax5.set_title('age Distribution')


sns.boxplot(x = 'target', y = 'age',data = heart, palette = cust_palt[2:],ax = ax5)

sns.swarmplot(x = 'target', y = 'age',data = heart, palette = cust_palt[:2],ax = ax5)


# plt.show()



plt.figure(figsize = (16,10))
sns.pairplot(heart[['trestbps','chol','thalach','oldpeak','age','target']],markers=['o','D'])
plt.show()


#3D scatter plot
import plotly.express as px
fig = px.scatter_3d(heart, x = 'chol',y = 'thalach',z = 'age',size = 'oldpeak',color = 'target',opacity=0.8)
# reference : https://plotly.com/python/reference/layout/
fig.update_layout(margin = dict(l=0,r=0,b=0,t=0))
fig.show()


def freq(df,cols, xi, hue = None, row=4, col = 1):
    fig,axes = plt.subplots(row,col,figsize = (16,12),sharex = True)
    axes = axes.flatten()

    for i,j in zip(df[cols].columns, axes):
        sns.pointplot(x = xi,y=i,data = df, palette = cust_palt[:2],hue = hue,ax =j)
        plt.tight_layout()

freq(heart, ['trestbps','chol','thalach','oldpeak'],'age',hue = 'target', row =4, col=1)
plt.show()


#corr matirx
corrlation_matrix = heart.corr()
# mask = np.triu(corrlation_matrix.corr())
plt.figure(figsize = (20,12))
sns.heatmap(corrlation_matrix, annot = True, fmt = '.2f',cmap = 'summer',linewidths=1,cbar = True)
plt.show()
'''



X = heart.drop('target',axis=1)
y = heart['target']
# print(X)
# print(y)

# Selecting some sklearn classifiers:
gradclass = GradientBoostingClassifier(random_state=42)
knclass = KNeighborsClassifier()
dectree = DecisionTreeClassifier(random_state=42)
svc = SVC()
randfclass = RandomForestClassifier(random_state=42)
adaclass = AdaBoostClassifier(random_state=42)
gsclass = GaussianNB()
xgbclass = XGBClassifier()
ligthgbmsclass = LGBMClassifier()
catboostclass = CatBoostClassifier()

from sklearn.model_selection import KFold
cv = KFold(5, shuffle=True, random_state=42)
classifiers = [gradclass, knclass, dectree, svc, randfclass, adaclass,gsclass,xgbclass,ligthgbmsclass,catboostclass]

def model_check(X, y, classifiers, cv):
    
    ''' A function for testing multiple classifiers and return several metrics. '''
    
    model_table = pd.DataFrame()

    row_index = 0
    for cls in classifiers:

        MLA_name = cls.__class__.__name__
        model_table.loc[row_index, 'Model Name'] = MLA_name
        
        cv_results = cross_validate(cls, X, y, cv=cv,scoring=('accuracy','f1','roc_auc'),return_train_score=True,n_jobs=-1)
        model_table.loc[row_index, 'Train Roc/AUC Mean'] = cv_results['train_roc_auc'].mean()
        model_table.loc[row_index, 'Test Roc/AUC Mean'] = cv_results['test_roc_auc'].mean()
        model_table.loc[row_index, 'Test Roc/AUC Std'] = cv_results['test_roc_auc'].std()
        model_table.loc[row_index, 'Train Accuracy Mean'] = cv_results['train_accuracy'].mean()
        model_table.loc[row_index, 'Test Accuracy Mean'] = cv_results[
            'test_accuracy'].mean()
        model_table.loc[row_index, 'Test Acc Std'] = cv_results['test_accuracy'].std()
        model_table.loc[row_index, 'Train F1 Mean'] = cv_results[
            'train_f1'].mean()
        model_table.loc[row_index, 'Test F1 Mean'] = cv_results[
            'test_f1'].mean()
        model_table.loc[row_index, 'Test F1 Std'] = cv_results['test_f1'].std()
        model_table.loc[row_index, 'Time'] = cv_results['fit_time'].mean()

        row_index += 1        

    model_table.sort_values(by=['Test F1 Mean'],
                            ascending=False,
                            inplace=True)

    return model_table

raw_models = model_check(X, y, classifiers, cv)
print(raw_models)




def f_imp(classifiers, X, y, bins):
    
    ''' A function for displaying feature importances'''
    
    fig, axes = plt.subplots(1, 2, figsize=(20, 6))
    axes = axes.flatten()

    for ax, classifier in zip(axes, classifiers):

        try:
            classifier.fit(X, y)
            feature_imp = pd.DataFrame(sorted(
                zip(classifier.feature_importances_, X.columns)),
                                       columns=['Value', 'Feature'])

            sns.barplot(x="Value",
                        y="Feature",
                        data=feature_imp.sort_values(by="Value",
                                                     ascending=False),
                        ax=ax,
                        palette='plasma')
            plt.title('Features')
            plt.tight_layout()
            ax.set(title=f'{classifier.__class__.__name__} Feature Impotances')
            ax.xaxis.set_major_locator(MaxNLocator(nbins=bins))
        except:
            continue
    plt.show()

f_imp([randfclass,dectree], X, y, 6)