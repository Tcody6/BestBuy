import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

data = pd.read_csv("Hackathon Data.csv", low_memory=False)

data.dropna(axis=0, how='all', inplace=True)
data.dropna(axis=1, how='all', inplace=True)
data['SALES_DATE'] = pd.to_datetime(data['SALES_DATE'])
data['Encoded_SKU_ID'] = data['Encoded_SKU_ID'].astype(int)
data['Week'] = data['SALES_DATE'].dt.week
data['Year'] = data['SALES_DATE'].dt.year
data['quarter'] = data["SALES_DATE"].dt.quarter
data['DAILY_UNITS'] = data['DAILY_UNITS'].str.replace(',', '')
data['DAILY_UNITS'] = data['DAILY_UNITS'].astype(int)

#Total Sales
dayOfWeekAll = seasonal_decompose(data.groupby(['SALES_DATE'])['DAILY_UNITS'].sum(), period=7)
weekOfYearAll = seasonal_decompose(data.groupby(['Year', 'Week'])['DAILY_UNITS'].sum(), period=52)
data = pd.merge(data, dayOfWeekAll.seasonal, how='left', left_on=['SALES_DATE'], right_index=True)
data.rename(columns={'seasonal': 'day_effect_all'}, inplace=True)
data = pd.merge(data, weekOfYearAll.seasonal, how='left', left_on=['Year', 'Week'], right_index=True)
data.rename(columns={'seasonal': 'week_effect_all'}, inplace=True)

#Category Sales
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

#ML Sales
MLs = data['ML_NAME'].unique()
ml_df_day = []
ml_df_week = []

for x in MLs:
    dayOfWeek = seasonal_decompose(data[data['ML_NAME'] == x].groupby(['SALES_DATE', 'ML_NAME'])['DAILY_UNITS'].sum(), period=7)
    weekOfYear = seasonal_decompose(data[data['ML_NAME'] == x].groupby(['Year', 'Week', 'ML_NAME'])['DAILY_UNITS'].sum(), period=52)
    ml_df_day.append(dayOfWeek.seasonal)
    ml_df_week.append(weekOfYear.seasonal)

ml_df_day = pd.concat(ml_df_day, ignore_index=False, sort=False)
ml_df_week = pd.concat(ml_df_week, ignore_index=False, sort=False)
data = pd.merge(data, ml_df_day, how='left', left_on=['SALES_DATE', 'ML_NAME'], right_index=True)
data.rename(columns={'seasonal': 'day_effect_ML'}, inplace=True)
data = pd.merge(data, ml_df_week, how='left', left_on=['Year', 'Week', 'ML_NAME'], right_index=True)
data.rename(columns={'seasonal': 'week_effect_ML'}, inplace=True)

#Class Sales
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

#SubClass Sales
subclasses = data['SUBCLASS_NAME'].unique()
subclass_df_day = []
subclass_df_week = []

for x in subclasses:
    dayOfWeek = seasonal_decompose(data[data['SUBCLASS_NAME'] == x].groupby(['SALES_DATE', 'SUBCLASS_NAME'])['DAILY_UNITS'].sum(), period=7)
    subclass_df_day.append(dayOfWeek.seasonal)
    try:
        weekOfYear = seasonal_decompose(data[data['SUBCLASS_NAME'] == x].groupby(['Year', 'Week', 'SUBCLASS_NAME'])['DAILY_UNITS'].sum(), period=52)
        subclass_df_week.append(weekOfYear.seasonal)
    except:
        pass
subclass_df_day = pd.concat(subclass_df_day, ignore_index=False, sort=False)
subclass_df_week = pd.concat(subclass_df_week, ignore_index=False, sort=False)
data = pd.merge(data, subclass_df_day, how='left', left_on=['SALES_DATE', 'SUBCLASS_NAME'], right_index=True)
data.rename(columns={'seasonal': 'day_effect_subclass'}, inplace=True)
data = pd.merge(data, subclass_df_week, how='left', left_on=['Year', 'Week', 'SUBCLASS_NAME'], right_index=True)
data.rename(columns={'seasonal': 'week_effect_subclass'}, inplace=True)