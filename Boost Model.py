import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
import time
import xgboost as xgb
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

#Open Files
data_train = pd.read_csv("Updated Inputs Unscaled.csv", low_memory=False, index_col=0)

#Set Up Model
boost = XGBRegressor(max_depth=4, n_estimators=600,
    min_child_weight=0.6,
    subsample=0.6,
    eta=0.1,
    seed=42)

#Fit Data
boost.fit(data_train.iloc[653023:,15:], data_train.iloc[653023:,10])

#Save Model
boost.save_model('trained_model.json')

#Export and Graph Feature Importance
feat_importances = pd.Series(boost.feature_importances_, index=boost.feature_names_in_)
feat_importances.sort_values().plot.bar()
plt.title("Feature Importance")
plt.xlabel("Feature")
plt.ylabel("Importance")
plt.tight_layout()
plt.show()
feat_importances.to_csv("Features.csv")