import pandas as pd
import numpy as np
from datetime import datetime
from statsmodels.tsa.seasonal import seasonal_decompose

#Open File
data = pd.read_csv("Hackathon Data.csv", low_memory=False)

#Data Cleaning
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
data['RETAIL_PRICE'] = data['RETAIL_PRICE'].str.replace("?", "0")
data['RETAIL_PRICE'] = data['RETAIL_PRICE'].str.replace(",", "")
data['RETAIL_PRICE'] = data['RETAIL_PRICE'].astype(float)

#Dummies
dummies_Inventory = pd.get_dummies(data["Inventory"], prefix = 'INV')
data = pd.concat([data, dummies_Inventory], axis = 1)

#Thanksgiving Holidays
data['IS_THANKSGIVING'] = np.where((data['SALES_DATE'] == pd.to_datetime("11/23/2017")) |
                                    (data['SALES_DATE'] == pd.to_datetime("11/22/2018")) |
                                    (data['SALES_DATE'] == pd.to_datetime("11/28/2019")) |
                                    (data['SALES_DATE'] == pd.to_datetime("11/26/2020")) |
                                    (data['SALES_DATE'] == pd.to_datetime("11/25/2021")) |
                                    (data['SALES_DATE'] == pd.to_datetime("11/24/2022")), 1, 0)

data['IS_BLACK_FRIDAY'] = np.where((data['SALES_DATE'] == pd.to_datetime("11/24/2017")) |
                                    (data['SALES_DATE'] == pd.to_datetime("11/23/2018")) |
                                    (data['SALES_DATE'] == pd.to_datetime("11/29/2019")) |
                                    (data['SALES_DATE'] == pd.to_datetime("11/27/2020")) |
                                    (data['SALES_DATE'] == pd.to_datetime("11/26/2021")) |
                                    (data['SALES_DATE'] == pd.to_datetime("11/25/2022")), 1, 0)

data['IS_BLACK_SATURDAY'] = np.where((data['SALES_DATE'] == pd.to_datetime("11/25/2017")) |
                                    (data['SALES_DATE'] == pd.to_datetime("11/24/2018")) |
                                    (data['SALES_DATE'] == pd.to_datetime("11/30/2019")) |
                                    (data['SALES_DATE'] == pd.to_datetime("11/28/2020")) |
                                    (data['SALES_DATE'] == pd.to_datetime("11/27/2021")) |
                                    (data['SALES_DATE'] == pd.to_datetime("11/26/2022")), 1, 0)

data['IS_BLACK_SUNDAY'] = np.where((data['SALES_DATE'] == pd.to_datetime("11/26/2017")) |
                                    (data['SALES_DATE'] == pd.to_datetime("11/25/2018")) |
                                    (data['SALES_DATE'] == pd.to_datetime("12/1/2019")) |
                                    (data['SALES_DATE'] == pd.to_datetime("11/29/2020")) |
                                    (data['SALES_DATE'] == pd.to_datetime("11/28/2021")) |
                                    (data['SALES_DATE'] == pd.to_datetime("11/27/2022")), 1, 0)

data['IS_CYBER_MONDAY'] = np.where((data['SALES_DATE'] == pd.to_datetime("11/27/2017")) |
                                    (data['SALES_DATE'] == pd.to_datetime("11/26/2018")) |
                                    (data['SALES_DATE'] == pd.to_datetime("12/2/2019")) |
                                    (data['SALES_DATE'] == pd.to_datetime("11/30/2020")) |
                                    (data['SALES_DATE'] == pd.to_datetime("11/29/2021")) |
                                    (data['SALES_DATE'] == pd.to_datetime("11/28/2022")), 1, 0)

#Get Difference between Regular Price and Promo Price
data['PROMO_PRICE'] = data['PROMO_PRICE'].str.replace("?", "0")
data['PROMO_PRICE'] = data['PROMO_PRICE'].str.replace(",", "")
data['PROMO_PRICE'] = data['PROMO_PRICE'].astype(float)
data.loc[data['PROMO_PRICE'] == 0, 'PROMO_PRICE'] = data['RETAIL_PRICE']
data['PROMO_DIFF'] = data['RETAIL_PRICE'] - data['PROMO_PRICE']

#SKU Total Sales for Time Periods

data['WeekPrior'] = data['SALES_DATE'] - pd.Timedelta(7, 'd')
sameDayLastWeekSameSKU = data.groupby(['Encoded_SKU_ID', 'SALES_DATE'])['DAILY_UNITS'].sum()
data = pd.merge(data, sameDayLastWeekSameSKU, how='left', left_on=['Encoded_SKU_ID', 'WeekPrior'], right_index=True)
data.rename(columns={"DAILY_UNITS_y": "sameDayLastWeekSameSKU"}, inplace=True)
data.rename(columns={"DAILY_UNITS_x": "DAILY_UNITS"}, inplace=True)
data.drop(['WeekPrior'], axis=1, inplace=True)

data['LastWeek'] = data['Week'] - 1
totalLastWeekSameSKU = data.groupby(['Encoded_SKU_ID', 'Week', 'Year'])['DAILY_UNITS'].sum()
data = pd.merge(data, totalLastWeekSameSKU, how='left', left_on=['Encoded_SKU_ID', 'LastWeek', 'Year'], right_index=True)
data.rename(columns={"DAILY_UNITS_y": "totalLastWeekSameSKU"}, inplace=True)
data.rename(columns={"DAILY_UNITS_x": "DAILY_UNITS"}, inplace=True)
data.drop(['LastWeek'], axis=1, inplace=True)

data['FourWeeks'] = data['Week'] - 4
totalFourWeeksSameSKU = data.groupby(['Encoded_SKU_ID', 'Week', 'Year'])['DAILY_UNITS'].sum()
data = pd.merge(data, totalFourWeeksSameSKU, how='left', left_on=['Encoded_SKU_ID', 'FourWeeks', 'Year'], right_index=True)
data.rename(columns={"DAILY_UNITS_y": "fourWeeksAgoSameSKU"}, inplace=True)
data.rename(columns={"DAILY_UNITS_x": "DAILY_UNITS"}, inplace=True)
data.drop(['FourWeeks'], axis=1, inplace=True)

data['LastQuarter'] = data['quarter'] - 1
totalLastQuarterSameSKU = data.groupby(['Encoded_SKU_ID', 'quarter', 'Year'])['DAILY_UNITS'].sum()
data = pd.merge(data, totalLastQuarterSameSKU, how='left', left_on=['Encoded_SKU_ID', 'LastQuarter', 'Year'], right_index=True)
data.rename(columns={"DAILY_UNITS_y": "lastQuarterSameSKU"}, inplace=True)
data.rename(columns={"DAILY_UNITS_x": "DAILY_UNITS"}, inplace=True)
data.drop(['LastQuarter'], axis=1, inplace=True)

data['LastYear'] = data['Year'] - 1
totalLastYearSameSKU = data.groupby(['Encoded_SKU_ID', 'Year'])['DAILY_UNITS'].sum()
data = pd.merge(data, totalLastYearSameSKU, how='left', left_on=['Encoded_SKU_ID', 'LastYear'], right_index=True)
data.rename(columns={"DAILY_UNITS_y": "lastYearSameSKU"}, inplace=True)
data.rename(columns={"DAILY_UNITS_x": "DAILY_UNITS"}, inplace=True)
data.drop(['LastYear'], axis=1, inplace=True)


#ML Total Sales During Time Periods

data['ThreeWeeks'] = data['Week'] - 3
totalThreeWeeksML = data.groupby(['ML_NAME', 'Week', 'Year'])['DAILY_UNITS'].sum()
data = pd.merge(data, totalThreeWeeksML, how='left', left_on=['ML_NAME', 'ThreeWeeks', 'Year'], right_index=True)
data.rename(columns={"DAILY_UNITS_y": "threeWeeksAgoML"}, inplace=True)
data.rename(columns={"DAILY_UNITS_x": "DAILY_UNITS"}, inplace=True)
data.drop(['ThreeWeeks'], axis=1, inplace=True)

data['FourWeeks'] = data['Week'] - 4
totalFourWeeksML = data.groupby(['ML_NAME', 'Week', 'Year'])['DAILY_UNITS'].sum()
data = pd.merge(data, totalFourWeeksML, how='left', left_on=['ML_NAME', 'FourWeeks', 'Year'], right_index=True)
data.rename(columns={"DAILY_UNITS_y": "fourWeeksAgoML"}, inplace=True)
data.rename(columns={"DAILY_UNITS_x": "DAILY_UNITS"}, inplace=True)
data.drop(['FourWeeks'], axis=1, inplace=True)

#Category Seasonality
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
data = pd.merge(data, cat_df_day, how='left', left_on=['SALES_DATE', 'CATEGORY_NAME'], right_index=True)
data.rename(columns={'seasonal': 'day_effect_category'}, inplace=True)
data = pd.merge(data, cat_df_week, how='left', left_on=['Year', 'Week', 'CATEGORY_NAME'], right_index=True)
data.rename(columns={'seasonal': 'week_effect_category'}, inplace=True)

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
data = pd.merge(data, class_df_day, how='left', left_on=['SALES_DATE', 'CLASS_NAME'], right_index=True)
data.rename(columns={'seasonal': 'day_effect_class'}, inplace=True)
data = pd.merge(data, class_df_week, how='left', left_on=['Year', 'Week', 'CLASS_NAME'], right_index=True)
data.rename(columns={'seasonal': 'week_effect_class'}, inplace=True)

#Save File
data.to_csv("Updated Inputs Unscaled.csv")