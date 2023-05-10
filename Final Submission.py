import pandas as pd
import numpy as np
from datetime import datetime
from statsmodels.tsa.seasonal import seasonal_decompose
import xgboost as xgb
import time
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

#Start Timer
start = time.perf_counter()

#Open Data Sets
data = pd.read_csv("Hackathon Data.csv", low_memory=False)
data_Vald = pd.read_csv("Validation_Data.csv", low_memory=False)

#Data Cleaning Validation
data_Vald.dropna(axis=0, how='all', inplace=True)
data_Vald.dropna(axis=1, how='all', inplace=True)
data_Vald['SALES_DATE'] = pd.to_datetime(data_Vald['SALES_DATE'])
data_Vald['Encoded_SKU_ID'] = data_Vald['Encoded_SKU_ID'].astype(int)
data_Vald['Week'] = data_Vald['SALES_DATE'].dt.week
data_Vald['Year'] = data_Vald['SALES_DATE'].dt.year
data_Vald['quarter'] = data_Vald["SALES_DATE"].dt.quarter
data_Vald['DAILY_UNITS'] = data_Vald['DAILY_UNITS'].astype(int)
data_Vald['month'] = data_Vald["SALES_DATE"].dt.month
data_Vald['RETAIL_PRICE'] = data_Vald['RETAIL_PRICE'].str.replace("?", "0")
data_Vald['RETAIL_PRICE'] = data_Vald['RETAIL_PRICE'].str.replace(",", "")
data_Vald['RETAIL_PRICE'] = data_Vald['RETAIL_PRICE'].astype(float)

#Data Cleaning Main Set
data.dropna(axis=0, how='all', inplace=True)
data.dropna(axis=1, how='all', inplace=True)
data['SALES_DATE'] = pd.to_datetime(data['SALES_DATE'])
data['Encoded_SKU_ID'] = data['Encoded_SKU_ID'].astype(int)
data['Week'] = data['SALES_DATE'].dt.week
data['Year'] = data['SALES_DATE'].dt.year
data['quarter'] = data["SALES_DATE"].dt.quarter
data['DAILY_UNITS'] = data['DAILY_UNITS'].str.replace(',', '')
data['DAILY_UNITS'] = data['DAILY_UNITS'].astype(int)
data['month'] = data["SALES_DATE"].dt.month

#Dummies
dummies_Inventory = pd.get_dummies(data_Vald["Inventory"], prefix = 'INV')
data_Vald = pd.concat([data_Vald, dummies_Inventory], axis = 1)

#Thanksgiving Holidays
data_Vald['IS_THANKSGIVING'] = np.where((data_Vald['SALES_DATE'] == pd.to_datetime("11/23/2017")) |
                                    (data_Vald['SALES_DATE'] == pd.to_datetime("11/22/2018")) |
                                    (data_Vald['SALES_DATE'] == pd.to_datetime("11/28/2019")) |
                                    (data_Vald['SALES_DATE'] == pd.to_datetime("11/26/2020")) |
                                    (data_Vald['SALES_DATE'] == pd.to_datetime("11/25/2021")) |
                                    (data_Vald['SALES_DATE'] == pd.to_datetime("11/24/2022")), 1, 0)

data_Vald['IS_BLACK_FRIDAY'] = np.where((data_Vald['SALES_DATE'] == pd.to_datetime("11/24/2017")) |
                                    (data_Vald['SALES_DATE'] == pd.to_datetime("11/23/2018")) |
                                    (data_Vald['SALES_DATE'] == pd.to_datetime("11/29/2019")) |
                                    (data_Vald['SALES_DATE'] == pd.to_datetime("11/27/2020")) |
                                    (data_Vald['SALES_DATE'] == pd.to_datetime("11/26/2021")) |
                                    (data_Vald['SALES_DATE'] == pd.to_datetime("11/25/2022")), 1, 0)

data_Vald['IS_BLACK_SATURDAY'] = np.where((data_Vald['SALES_DATE'] == pd.to_datetime("11/25/2017")) |
                                    (data_Vald['SALES_DATE'] == pd.to_datetime("11/24/2018")) |
                                    (data_Vald['SALES_DATE'] == pd.to_datetime("11/30/2019")) |
                                    (data_Vald['SALES_DATE'] == pd.to_datetime("11/28/2020")) |
                                    (data_Vald['SALES_DATE'] == pd.to_datetime("11/27/2021")) |
                                    (data_Vald['SALES_DATE'] == pd.to_datetime("11/26/2022")), 1, 0)

data_Vald['IS_BLACK_SUNDAY'] = np.where((data_Vald['SALES_DATE'] == pd.to_datetime("11/26/2017")) |
                                    (data_Vald['SALES_DATE'] == pd.to_datetime("11/25/2018")) |
                                    (data_Vald['SALES_DATE'] == pd.to_datetime("12/1/2019")) |
                                    (data_Vald['SALES_DATE'] == pd.to_datetime("11/29/2020")) |
                                    (data_Vald['SALES_DATE'] == pd.to_datetime("11/28/2021")) |
                                    (data_Vald['SALES_DATE'] == pd.to_datetime("11/27/2022")), 1, 0)

data_Vald['IS_CYBER_MONDAY'] = np.where((data_Vald['SALES_DATE'] == pd.to_datetime("11/27/2017")) |
                                    (data_Vald['SALES_DATE'] == pd.to_datetime("11/26/2018")) |
                                    (data_Vald['SALES_DATE'] == pd.to_datetime("12/2/2019")) |
                                    (data_Vald['SALES_DATE'] == pd.to_datetime("11/30/2020")) |
                                    (data_Vald['SALES_DATE'] == pd.to_datetime("11/29/2021")) |
                                    (data_Vald['SALES_DATE'] == pd.to_datetime("11/28/2022")), 1, 0)

#Get Difference between Regular Price and Promo Price
data_Vald['PROMO_PRICE'] = data_Vald['PROMO_PRICE'].str.replace("?", "0")
data_Vald['PROMO_PRICE'] = data_Vald['PROMO_PRICE'].str.replace(",", "")
data_Vald['PROMO_PRICE'] = data_Vald['PROMO_PRICE'].astype(float)
data_Vald.loc[data_Vald['PROMO_PRICE'] == 0, 'PROMO_PRICE'] = data_Vald['RETAIL_PRICE']
data_Vald['PROMO_DIFF'] = data_Vald['RETAIL_PRICE'] - data_Vald['PROMO_PRICE']

#SKU Sales Totals in Different Time Periods

data_Vald['WeekPrior'] = data_Vald['SALES_DATE'] - pd.Timedelta(7, 'd')
sameDayLastWeekSameSKU = data.groupby(['Encoded_SKU_ID', 'SALES_DATE'])['DAILY_UNITS'].sum()
data_Vald = pd.merge(data_Vald, sameDayLastWeekSameSKU, how='left', left_on=['Encoded_SKU_ID', 'WeekPrior'], right_index=True)
data_Vald.rename(columns={"DAILY_UNITS_y": "sameDayLastWeekSameSKU"}, inplace=True)
data_Vald.rename(columns={"DAILY_UNITS_x": "DAILY_UNITS"}, inplace=True)
data_Vald.drop(['WeekPrior'], axis=1, inplace=True)

data_Vald['LastWeek'] = data_Vald['Week'] - 1
totalLastWeekSameSKU = data.groupby(['Encoded_SKU_ID', 'Week', 'Year'])['DAILY_UNITS'].sum()
data_Vald = pd.merge(data_Vald, totalLastWeekSameSKU, how='left', left_on=['Encoded_SKU_ID', 'LastWeek', 'Year'], right_index=True)
data_Vald.rename(columns={"DAILY_UNITS_y": "totalLastWeekSameSKU"}, inplace=True)
data_Vald.rename(columns={"DAILY_UNITS_x": "DAILY_UNITS"}, inplace=True)
data_Vald.drop(['LastWeek'], axis=1, inplace=True)

data_Vald['FourWeeks'] = data_Vald['Week'] - 4
totalFourWeeksSameSKU = data.groupby(['Encoded_SKU_ID', 'Week', 'Year'])['DAILY_UNITS'].sum()
data_Vald = pd.merge(data_Vald, totalFourWeeksSameSKU, how='left', left_on=['Encoded_SKU_ID', 'FourWeeks', 'Year'], right_index=True)
data_Vald.rename(columns={"DAILY_UNITS_y": "fourWeeksAgoSameSKU"}, inplace=True)
data_Vald.rename(columns={"DAILY_UNITS_x": "DAILY_UNITS"}, inplace=True)
data_Vald.drop(['FourWeeks'], axis=1, inplace=True)

data_Vald['LastQuarter'] = data_Vald['quarter'] - 1
totalLastQuarterSameSKU = data.groupby(['Encoded_SKU_ID', 'quarter', 'Year'])['DAILY_UNITS'].sum()
data_Vald = pd.merge(data_Vald, totalLastQuarterSameSKU, how='left', left_on=['Encoded_SKU_ID', 'LastQuarter', 'Year'], right_index=True)
data_Vald.rename(columns={"DAILY_UNITS_y": "lastQuarterSameSKU"}, inplace=True)
data_Vald.rename(columns={"DAILY_UNITS_x": "DAILY_UNITS"}, inplace=True)
data_Vald.drop(['LastQuarter'], axis=1, inplace=True)

data_Vald['LastYear'] = data_Vald['Year'] - 1
totalLastYearSameSKU = data.groupby(['Encoded_SKU_ID', 'Year'])['DAILY_UNITS'].sum()
data_Vald = pd.merge(data_Vald, totalLastYearSameSKU, how='left', left_on=['Encoded_SKU_ID', 'LastYear'], right_index=True)
data_Vald.rename(columns={"DAILY_UNITS_y": "lastYearSameSKU"}, inplace=True)
data_Vald.rename(columns={"DAILY_UNITS_x": "DAILY_UNITS"}, inplace=True)
data_Vald.drop(['LastYear'], axis=1, inplace=True)

#ML Total Sales in Different Time Periods

data_Vald['ThreeWeeks'] = data_Vald['Week'] - 3
totalThreeWeeksML = data.groupby(['ML_NAME', 'Week', 'Year'])['DAILY_UNITS'].sum()
data_Vald = pd.merge(data_Vald, totalThreeWeeksML, how='left', left_on=['ML_NAME', 'ThreeWeeks', 'Year'], right_index=True)
data_Vald.rename(columns={"DAILY_UNITS_y": "threeWeeksAgoML"}, inplace=True)
data_Vald.rename(columns={"DAILY_UNITS_x": "DAILY_UNITS"}, inplace=True)
data_Vald.drop(['ThreeWeeks'], axis=1, inplace=True)

data_Vald['FourWeeks'] = data_Vald['Week'] - 4
totalFourWeeksML = data.groupby(['ML_NAME', 'Week', 'Year'])['DAILY_UNITS'].sum()
data_Vald = pd.merge(data_Vald, totalFourWeeksML, how='left', left_on=['ML_NAME', 'FourWeeks', 'Year'], right_index=True)
data_Vald.rename(columns={"DAILY_UNITS_y": "fourWeeksAgoML"}, inplace=True)
data_Vald.rename(columns={"DAILY_UNITS_x": "DAILY_UNITS"}, inplace=True)
data_Vald.drop(['FourWeeks'], axis=1, inplace=True)

#Category Seasonality Calculations
categories = data['CATEGORY_NAME'].unique()
cat_df_day = []
cat_df_week = []

for x in categories:
    dayOfWeek = seasonal_decompose(data[data['CATEGORY_NAME'] == x].groupby(['SALES_DATE', 'CATEGORY_NAME'])['DAILY_UNITS'].sum(), period=7)
    weekOfYear = seasonal_decompose(data[data['CATEGORY_NAME'] == x].groupby(['Year', 'Week', 'CATEGORY_NAME'])['DAILY_UNITS'].sum(), period=52)
    cat_df_day.append(dayOfWeek.seasonal)
    cat_df_week.append(weekOfYear.seasonal)

cat_df_day = pd.concat(cat_df_day, ignore_index=False, sort=False)
cat_df_week = pd.concat(cat_df_week, ignore_index=False, sort=False)
data_Vald['WeekPrior'] = data_Vald['SALES_DATE'] - pd.Timedelta(7, 'd')
data_Vald = pd.merge(data_Vald, cat_df_day, how='left', left_on=['WeekPrior', 'CATEGORY_NAME'], right_index=True)
data_Vald.rename(columns={'seasonal': 'day_effect_category'}, inplace=True)
data_Vald['YearPrior'] = data_Vald['Year'] - 1
data_Vald = pd.merge(data_Vald, cat_df_week, how='left', left_on=['YearPrior', 'Week', 'CATEGORY_NAME'], right_index=True)
data_Vald.rename(columns={'seasonal': 'week_effect_category'}, inplace=True)

#Class Seasonality Calculations
classes = data['CLASS_NAME'].unique()
class_df_day = []
class_df_week = []

for x in classes:
    dayOfWeek = seasonal_decompose(data[data['CLASS_NAME'] == x].groupby(['SALES_DATE', 'CLASS_NAME'])['DAILY_UNITS'].sum(), period=7)
    weekOfYear = seasonal_decompose(data[data['CLASS_NAME'] == x].groupby(['Year', 'Week', 'CLASS_NAME'])['DAILY_UNITS'].sum(), period=52)
    class_df_day.append(dayOfWeek.seasonal)
    class_df_week.append(weekOfYear.seasonal)

class_df_day = pd.concat(class_df_day, ignore_index=False, sort=False)
class_df_week = pd.concat(class_df_week, ignore_index=False, sort=False)
data_Vald = pd.merge(data_Vald, class_df_day, how='left', left_on=['WeekPrior', 'CLASS_NAME'], right_index=True)
data_Vald.rename(columns={'seasonal': 'day_effect_class'}, inplace=True)
data_Vald = pd.merge(data_Vald, class_df_week, how='left', left_on=['YearPrior', 'Week', 'CLASS_NAME'], right_index=True)
data_Vald.rename(columns={'seasonal': 'week_effect_class'}, inplace=True)

data_Vald.drop(['WeekPrior', 'YearPrior'], axis=1, inplace=True)

#Loading File
model = xgb.XGBRegressor()

model.load_model('trained_model.json')

boost_predict = model.predict(data_Vald.iloc[:,15:]).round().astype(int)

rSqBoostTest = r2_score(data_Vald.iloc[:,10], boost_predict)
RMSEBoostTest = mean_squared_error(data_Vald.iloc[:,10], boost_predict, squared=False, )

#Stop Timer
stop = time.perf_counter()

#Score
print("R Squared = ", rSqBoostTest)
print("RMSEBoostTest = ", RMSEBoostTest)
print("Seconds to Run = ", (stop - start))

#Calculate Total Sales Numbers
data_Vald['Forecasted_Sales'] = boost_predict
data_Vald['Forecasted_Sales_$'] = data_Vald['Forecasted_Sales'] * data_Vald['PROMO_PRICE']
data_Vald['Actual_Sales_$'] = data_Vald['DAILY_UNITS'] * data_Vald['PROMO_PRICE']

print(data_Vald['Forecasted_Sales_$'].sum())
print(data_Vald['Actual_Sales_$'].sum())

export = pd.concat([data_Vald.iloc[:,:11],data_Vald['Forecasted_Sales']], axis=1)

#Export Final Results
export.to_csv("Final_Results.csv")
