#import packages needed for RFR model
import pandas as pd
import xgboost as xgb
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

#read cleaned data
data = pd.read_csv("Updated Inputs Unscaled.csv", low_memory=False, index_col=0)

data_train = data[(data['Year'] < 2022) | ((data['Year'] == 2022) & (data['month'] < 8))]
data_test = data[(data['Year'] == 2022) & (data['month'] == 8)]

#feature selection
rf_feat = xgb.XGBRFRegressor()
rf_feat.fit(data_train.iloc[650000:,15:], data_train.iloc[650000:,10])
select = SelectFromModel(rf_feat, threshold=0.02, prefit=True)
X_train = select.transform(data_train.iloc[650000:,15:])
X_test = select.transform(data_test.iloc[:,15:])

#define model
boost_mod = xgb.XGBRFRegressor()

#train model
boost_mod.fit(data_train.iloc[650000:,15:], data_train.iloc[650000:,10])
y_predict = boost_mod.predict(data_test.iloc[:,15:])

#evaluating performance of model using MSE
MAE = mean_absolute_error(data_test.iloc[:,10], y_predict)
MSE = mean_squared_error(data_test.iloc[:,10], y_predict)
R2 = r2_score(data_test.iloc[:,10], y_predict)

print("Mean Absolute Error:", MAE)
print("Mean Squared Error:", MSE)
print("R^2 Score:", R2)