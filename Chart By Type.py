import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("Hackathon Data.csv", low_memory=False)

data.dropna(axis=0, how='all', inplace=True)
data.dropna(axis=1, how='all', inplace=True)
data['SALES_DATE'] = pd.to_datetime(data['SALES_DATE'])
data['Encoded_SKU_ID'] = data['Encoded_SKU_ID'].astype(int)
data['Week'] = data['SALES_DATE'].dt.week
data['Year'] = data['SALES_DATE'].dt.year
data['Month'] = data['SALES_DATE'].dt.month
data['quarter'] = data["SALES_DATE"].dt.quarter
data['DAILY_UNITS'] = data['DAILY_UNITS'].str.replace(',', '')
data['DAILY_UNITS'] = data['DAILY_UNITS'].astype(int)

all = data.groupby(['SALES_DATE'])['DAILY_UNITS'].sum()

plt.plot(all)
plt.title("All Products")
plt.xlabel("Date")
plt.ylabel("Total Units Sold")
plt.show()

categories = data['CATEGORY_NAME'].unique()

for x in categories:
    subset = data[data['CATEGORY_NAME'] == x].groupby(['SALES_DATE'])['DAILY_UNITS'].sum()
    plt.plot(subset)
    plt.title(x)
    plt.xlabel("Date")
    plt.ylabel("Total Units Sold")
    plt.show()

ML = data['ML_NAME'].unique()

for x in ML:
    subset = data[data['ML_NAME'] == x].groupby(['SALES_DATE'])['DAILY_UNITS'].sum()
    plt.plot(subset)
    plt.title(x)
    plt.xlabel("Date")
    plt.ylabel("Total Units Sold")
    plt.show()

Class = data['CLASS_NAME'].unique()

for x in Class:
    subset = data[data['CLASS_NAME'] == x].groupby(['SALES_DATE'])['DAILY_UNITS'].sum()
    plt.plot(subset)
    plt.title(x)
    plt.xlabel("Date")
    plt.ylabel("Total Units Sold")
    plt.show()

Subclass = data['SUBCLASS_NAME'].unique()

for x in Subclass:
    subset = data[data['SUBCLASS_NAME'] == x].groupby(['SALES_DATE'])['DAILY_UNITS'].sum()
    plt.plot(subset)
    plt.title(x)
    plt.xlabel("Date")
    plt.ylabel("Total Units Sold")
    plt.show()

