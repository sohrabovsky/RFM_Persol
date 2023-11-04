from functools import reduce
import requests
import pandas as pd
import numpy as np
import jalali_pandas
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import seaborn as sns
import pyodbc
import matplotlib.pyplot as plt
from scipy import stats
import plotly.express as px
from datetime import date
from operator import attrgetter


def mean_purchase_widnow(df):
    customers = df.CustomerCode.unique()
    dff = pd.DataFrame()
    for customer in customers:
        temp_df = df[df['CustomerCode'] == customer].sort_values(
            by='jdate').reset_index(drop=True)
        diff_dates = []
        for i in range(len(temp_df) - 1):
            diff = temp_df.loc[i+1, 'jdate'] - temp_df.loc[i, 'jdate']
            diff_dates.append(diff)
            mean_diff = np.mean(diff_dates)
            dff.loc[customer, 'mean_diff'] = mean_diff
    return dff.mean().apply(attrgetter('days'))[0]


cluster_names = ['Champion', 'Loyal', 'Promising',
                 'Needing Attention 1', 'Needing Attention 2']


def cluster_naming(df):
    dff = df.groupby('cluster')['recency', 'frequency', 'monetary'].mean().drop(
        index='New Customers').sort_values(by=['frequency', 'monetary'], ascending=False).sort_values(by='recency')
    cluster_orders = dff.index
    number_of_cluster = df.cluster.nunique()
    i = 0
    for cluster in cluster_orders:
        index = df[df['cluster'] == cluster].index
        df.loc[index, 'cluster'] = cluster_names[i]
        i += 1
    return df


# Specifying the ODBC driver, server name, database, etc. directly
cnxn = pyodbc.connect(
    'DRIVER={ODBC Driver 17 for SQL Server};SERVER=192.168.10.41;DATABASE=ReportDB;UID=SelfServiceBI;PWD=PL@PS@SLFSRVbi10')

df_sales = pd.read_sql("SELECT * FROM dbo.FullInvoiceReport", cnxn)
df_category = pd.read_sql('SELECT * FROM dbo.ProductCategory', cnxn)

df = df_sales.merge(right=df_category, how='left', on='ProductCode')

df.rename(columns={'G1': 'Category'}, inplace=True)

df.drop(
    columns=[i for i in df.columns[df.columns.str.contains('G')]], inplace=True)

path = r"C:\Users\mk-05\Documents\Python Scripts\rfm"


df_wood = df[df['Category'] == 'Wood Products'].reset_index(drop=True)
df_Cellulosic = df[df['Category'] ==
                   'Cellulosic Products'].reset_index(drop=True)
df_chemical_polymer = df[df['Category'] ==
                         'Chemical & Polymer Products'].reset_index(drop=True)

shayan_customer_code = df_wood[df_wood['Customer']
                               == 'شایان حقیقی']['CustomerCode'].unique()[0]
syamak_file = pd.read_excel(
    r'C:\Users\mk-05\Documents\Python Scripts\syamak.xlsx', usecols=['shayan'])
syamak_file = syamak_file.drop(index=15).reset_index(drop=True)
syamak_file['family'] = syamak_file['shayan'].apply(lambda x: x.split()[-1])
df_wood['family'] = df_wood['Customer'].apply(lambda x: x.split()[-1])

index = df_wood[df_wood['family'].isin(syamak_file['family'].unique())].index
df_wood.loc[index, 'Customer'] = 'شایان حقیقی'
df_wood.loc[index, 'CustomerCode'] = shayan_customer_code

# RFM Wood

wood_monetary = df_wood.groupby('CustomerCode')['NetAmount'].sum()
wood_monetary = wood_monetary.reset_index()

df_wood["jdate"] = df_wood["InvoiceDate"].jalali.parse_jalali("%Y/%m/%d")
for index in range(len(df_wood)):
    df_wood.loc[index, 'Diff'] = (
        df_wood['jdate'].max() - df_wood.loc[index, 'jdate']).days
wood_recency = df_wood.groupby(['CustomerCode'])['Diff'].min()
wood_latency = df_wood.groupby(['CustomerCode'])['Diff'].max()

wood_recency = wood_recency.reset_index()
wood_latency = wood_latency.reset_index()

wood_frequency = df_wood.groupby('CustomerCode')['InvoiceNumber'].count()
wood_frequency = wood_frequency.reset_index()

dataframes = [wood_recency, wood_frequency, wood_monetary, wood_latency]
rfm_wood = reduce(lambda df1, df2: pd.merge(
    left=df1, right=df2, how='inner', on='CustomerCode'), dataframes)


rfm_wood.rename(columns={
    'Diff_x': 'recency',
    'InvoiceNumber': 'frequency',
    'NetAmount': 'monetary',
    'Diff_y': 'latency'
}, inplace=True)

wood_retCustomers_index = rfm_wood[rfm_wood['latency']
                                   > mean_purchase_widnow(df_wood)].index

scaler = StandardScaler()

rfm_wood_norm = pd.DataFrame(scaler.fit_transform(
    rfm_wood[['recency', 'frequency', 'monetary']]), columns=['recency', 'frequency', 'monetary'])

rfm_wood_norm = rfm_wood_norm.loc[wood_retCustomers_index, :]

kmeans = KMeans(n_clusters=4, max_iter=50)
kmeans.fit(rfm_wood_norm[['recency', 'frequency', 'monetary']])

rfm_wood_norm['cluster'] = kmeans.labels_
rfm_wood.loc[wood_retCustomers_index,
             'cluster'] = rfm_wood_norm.loc[wood_retCustomers_index, 'cluster']
rfm_wood.cluster.fillna('New Customers', inplace=True)

rfm_wood.loc[:, 'date'] = date.today()

cluster_naming(rfm_wood)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(projection='3d')
for cluster in rfm_wood.cluster.unique():
    x = rfm_wood[rfm_wood['cluster'] == cluster]['recency']
    y = rfm_wood[rfm_wood['cluster'] == cluster]['frequency']
    z = rfm_wood[rfm_wood['cluster'] == cluster]['monetary']
    ax.scatter(x, y, z, label=cluster)
    ax.set_xlabel('Recency')
    ax.set_ylabel('Frequency')
    ax.set_zlabel("Monetary", rotation=90)
    ax.zaxis.labelpad = -0.2  # <- change the value here
plt.legend()
plt.title('RFM Segmentation for Wood Product')
plt.savefig(path + '\\' + 'rfm_wood.jpeg')


# RFM Celluluse
# df_Cellulosic
cellulosic_monetary = df_Cellulosic.groupby('CustomerCode')['NetAmount'].sum()
cellulosic_monetary = cellulosic_monetary.reset_index()

df_Cellulosic["jdate"] = df_Cellulosic["InvoiceDate"].jalali.parse_jalali(
    "%Y/%m/%d")
for index in range(len(df_Cellulosic)):
    df_Cellulosic.loc[index, 'Diff'] = (
        df_Cellulosic['jdate'].max() - df_Cellulosic.loc[index, 'jdate']).days
cellulosic_recency = df_Cellulosic.groupby(['CustomerCode'])['Diff'].min()
cellulosic_latency = df_Cellulosic.groupby(['CustomerCode'])['Diff'].max()
cellulosic_recency = cellulosic_recency.reset_index()
cellulosic_latency = cellulosic_latency.reset_index()


cellulosic_frequency = df_Cellulosic.groupby(
    'CustomerCode')['InvoiceNumber'].count()
cellulosic_frequency = cellulosic_frequency.reset_index()

dataframes = [cellulosic_recency, cellulosic_frequency,
              cellulosic_monetary, cellulosic_latency]
rfm_cellulosic = reduce(lambda df1, df2: pd.merge(
    left=df1, right=df2, how='inner', on='CustomerCode'), dataframes)


rfm_cellulosic.rename(columns={
    'Diff_x': 'recency',
    'InvoiceNumber': 'frequency',
    'NetAmount': 'monetary',
    'Diff_y': 'latency'
}, inplace=True)

cellulosic_retCustomers_index = rfm_cellulosic[rfm_cellulosic['latency'] > mean_purchase_widnow(
    df_Cellulosic)].index

scaler = StandardScaler()
rfm_cellulosic_norm = pd.DataFrame(scaler.fit_transform(rfm_cellulosic[[
                                   'recency', 'frequency', 'monetary']]), columns=['recency', 'frequency', 'monetary'])
rfm_cellulosic_norm = rfm_cellulosic_norm.loc[cellulosic_retCustomers_index, :]

kmeans = KMeans(n_clusters=5, max_iter=300)
kmeans.fit(rfm_cellulosic_norm[['recency', 'frequency', 'monetary']])

rfm_cellulosic_norm['cluster'] = kmeans.labels_
rfm_cellulosic.loc[cellulosic_retCustomers_index,
                   'cluster'] = rfm_cellulosic_norm.loc[cellulosic_retCustomers_index, 'cluster']
rfm_cellulosic.cluster.fillna('New Customers', inplace=True)

rfm_cellulosic.loc[:, 'date'] = date.today()
cluster_naming(rfm_cellulosic)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(projection='3d')

for cluster in rfm_cellulosic.cluster.unique():
    x = rfm_cellulosic[rfm_cellulosic['cluster']
                       == cluster]['recency']
    y = rfm_cellulosic[rfm_cellulosic['cluster']
                       == cluster]['frequency']
    z = rfm_cellulosic[rfm_cellulosic['cluster']
                       == cluster]['monetary']
    ax.scatter(x, y, z, label=cluster)
    ax.set_xlabel('Recency')
    ax.set_ylabel('Frequency')
    ax.set_zlabel("Monetary", rotation=90)
    ax.zaxis.labelpad = -0.2  # <- change the value here
plt.title('RFM Segmentation for Cellulosic Product')
plt.legend()
plt.savefig(path + '\\' + 'rfm_cellulosic.jpeg')

# RFM chemical - polymer

chemical_monetary = df_chemical_polymer.groupby('CustomerCode')[
    'NetAmount'].sum()
chemical_monetary = chemical_monetary.reset_index()

df_chemical_polymer["jdate"] = df_chemical_polymer["InvoiceDate"].jalali.parse_jalali(
    "%Y/%m/%d")
for index in range(len(df_chemical_polymer)):
    df_chemical_polymer.loc[index, 'Diff'] = (
        df_chemical_polymer['jdate'].max() - df_chemical_polymer.loc[index, 'jdate']).days
chemical_recency = df_chemical_polymer.groupby(['CustomerCode'])['Diff'].min()
chemical_latency = df_chemical_polymer.groupby(['CustomerCode'])['Diff'].max()

chemical_recency = chemical_recency.reset_index()
chemical_latency = chemical_latency.reset_index()


chemical_frequency = df_chemical_polymer.groupby(
    'CustomerCode')['InvoiceNumber'].count()
chemical_frequency = chemical_frequency.reset_index()

dataframes = [chemical_recency, chemical_frequency,
              chemical_monetary, chemical_latency]
rfm_chemical = reduce(lambda df1, df2: pd.merge(
    left=df1, right=df2, how='inner', on='CustomerCode'), dataframes)


rfm_chemical.rename(columns={
    'Diff_x': 'recency',
    'InvoiceNumber': 'frequency',
    'NetAmount': 'monetary',
    'Diff_y': 'latency'
}, inplace=True)

chemical_retCustomers_index = rfm_chemical[rfm_chemical['latency'] > mean_purchase_widnow(
    df_chemical_polymer)].index

scaler = StandardScaler()

rfm_chemical_norm = pd.DataFrame(scaler.fit_transform(
    rfm_chemical[['recency', 'frequency', 'monetary']]), columns=['recency', 'frequency', 'monetary'])

rfm_chemical_norm = rfm_chemical_norm.loc[chemical_retCustomers_index, :]

kmeans = KMeans(n_clusters=4, max_iter=50)
kmeans.fit(rfm_chemical_norm[['recency', 'frequency', 'monetary']])

rfm_chemical_norm['cluster'] = kmeans.labels_
rfm_chemical.loc[chemical_retCustomers_index,
                 'cluster'] = rfm_chemical_norm.loc[chemical_retCustomers_index, 'cluster']
rfm_chemical.cluster.fillna('New Customers', inplace=True)

rfm_chemical.loc[:, 'date'] = date.today()

cluster_naming(rfm_chemical)


fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(projection='3d')

for cluster in rfm_chemical.cluster.unique():
    x = rfm_chemical[rfm_chemical['cluster'] == cluster]['recency']
    y = rfm_chemical[rfm_chemical['cluster'] == cluster]['frequency']
    z = rfm_chemical[rfm_chemical['cluster'] == cluster]['monetary']
    ax.scatter(x, y, z, label=cluster)
    ax.set_xlabel('Recency')
    ax.set_ylabel('Frequency')
    ax.set_zlabel("Monetary", rotation=90)
    ax.zaxis.labelpad = -0.2  # <- change the value here
plt.title('RFM Segmentation for Chemical Product')
plt.legend()
plt.savefig(path + '\\' + 'rfm_chemical.jpeg')

rfm_wood = rfm_wood.merge(right=df_wood[[
                          'CustomerCode', 'Customer']].drop_duplicates(), how='left', on='CustomerCode')
rfm_cellulosic = rfm_cellulosic.merge(right=df_Cellulosic[[
                                      'CustomerCode', 'Customer']].drop_duplicates(), how='left', on='CustomerCode')
rfm_chemical = rfm_chemical.merge(right=df_chemical_polymer[[
                                  'CustomerCode', 'Customer']].drop_duplicates(), how='left', on='CustomerCode')


with pd.ExcelWriter(r'C:\Users\mk-05\Documents\Python Scripts\rfm\rfm.xlsx') as writer:

    rfm_wood.to_excel(writer, sheet_name='wood', index=False)
    rfm_cellulosic.to_excel(writer, sheet_name='cellulosic', index=False)
    rfm_chemical.to_excel(writer, sheet_name='chemical', index=False)
