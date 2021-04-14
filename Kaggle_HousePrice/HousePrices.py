# Kaggle 숙제 House advanced regression problem
# (https://www.kaggle.com/c/house-prices-advanced-regression-techniques)

#  ------------------------------- Day 22 - 210409  -------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn 
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

