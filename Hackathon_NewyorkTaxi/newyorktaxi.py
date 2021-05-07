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

print(df)               # 
print(df.info())        # 
print(df.describe())    # 
print(df.isna().sum())  # 공란데이터 없음