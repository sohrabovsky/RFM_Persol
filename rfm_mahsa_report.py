from functools import reduce
import requests
import pandas as pd
import numpy as np
import jalali_pandas
from persiantools.jdatetime import JalaliDate
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import seaborn as sns
import pyodbc
import matplotlib.pyplot as plt
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
from datetime import date
from operator import attrgetter

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

df_cellulosic = df[df['Category'] ==
                   'Cellulosic Products'].reset_index(drop=True)

# Eliminating products cleaning (Cellulosic):
df_cellulosic = df_cellulosic[
    ~(df_cellulosic['ProductName'].str.contains('GB')) | (df_cellulosic['ProductName'].str.contains('پشت طوسی'))]

# If we want to separate Cellulosic departments
mahsa_customers = pd.read_excel('MRB cellulose customers.xlsx')
mahsa_customers = mahsa_customers[mahsa_customers['is_mahsa'] == True].customer.unique()
df_cellulosic_mahsa = df_cellulosic[df_cellulosic['Customer'].isin(mahsa_customers)]
df_cellulosic_alireza = df_cellulosic[~df_cellulosic['Customer'].isin(mahsa_customers)]

df_cellulosic_mahsa = df_cellulosic_mahsa.reset_index(drop=True)
df_cellulosic_mahsa = df_cellulosic_mahsa[
    df_cellulosic_mahsa['InvoiceDate'] >= '1402/01/01']  # In case we want customers after 1402
df_cellulosic_alireza = df_cellulosic_alireza.reset_index(drop=True)

cluster_names = ['Champion', 'Loyal', 'Promising', 'Needing Attention']

df= df_cellulosic_mahsa
quantity = df.groupby('CustomerCode')['Quantity'].sum()  # Monetary here is Quantity because of inflation effect
quantity = quantity.reset_index()

quantity_fbb= df[df['ProductName'].str.contains('FBB', case= True)].groupby('CustomerCode')['Quantity'].sum()
quantity_fbb.rename('FBB_quantity', inplace= True)
quantity_fbb= quantity_fbb.reset_index()

quantity_paper= df[df['ProductName'].str.contains('Paper', case= True)].groupby('CustomerCode')['Quantity'].sum()
quantity_paper.rename('paper_quantity', inplace= True)
quantity_paper= quantity_paper.reset_index()

quantity_pulp= df[df['ProductName'].str.contains('Pulp', case= True)].groupby('CustomerCode')['Quantity'].sum()
quantity_pulp.rename('pulp_quantity', inplace= True)
quantity_pulp= quantity_pulp.reset_index()

df["jdate"] = df["InvoiceDate"].jalali.parse_jalali("%Y/%m/%d")
for index in range(len(df)):
    df.loc[index, 'Diff'] = (
            df['jdate'].max() - df.loc[index, 'jdate']).days
recency = df.groupby(['CustomerCode'])['Diff'].min()
latency = df.groupby(['CustomerCode'])['Diff'].max()

recency = recency.reset_index()
latency = latency.reset_index()

frequency = df.groupby('CustomerCode')['InvoiceNumber'].count()
frequency = frequency.reset_index()

dataframes = [recency, frequency, quantity, latency]
rfm = reduce(lambda df1, df2: pd.merge(
    left=df1, right=df2, how='inner', on='CustomerCode'), dataframes)

rfm.rename(columns={
    'Diff_x': 'recency',
    'InvoiceNumber': 'frequency',
    'Quantity': 'quantity',
    'Diff_y': 'latency'
}, inplace=True)

def mean_purchase_widnow(df):
    customers = df.CustomerCode.unique()
    dff = pd.DataFrame()
    for customer in customers:
        temp_df = df[df['CustomerCode'] == customer].sort_values(
            by='jdate').reset_index(drop=True)
        diff_dates = []
        for i in range(len(temp_df) - 1):
            diff = temp_df.loc[i + 1, 'jdate'] - temp_df.loc[i, 'jdate']
            diff_dates.append(diff)
            mean_diff = np.mean(diff_dates)
            dff.loc[customer, 'mean_diff'] = mean_diff

    return dff.mean().apply(attrgetter('days'))[0]

new_customer_window_size = mean_purchase_widnow(df)
retCustomers_index = rfm[rfm['latency'] > new_customer_window_size].index
print(f'New Customer is defined as who purchased for the first time from Persol in {new_customer_window_size} days')

scaler = StandardScaler()

rfm_norm = pd.DataFrame(scaler.fit_transform(
    rfm[['recency', 'frequency', 'quantity']]), columns=['recency', 'frequency', 'quantity'])

rfm_norm = rfm_norm.loc[retCustomers_index, :]

kmeans = KMeans(n_clusters=4, max_iter=50)
kmeans.fit(rfm_norm[['recency', 'frequency', 'quantity']])
rfm_norm['cluster'] = kmeans.labels_

### Scoring
rfm_norm['RFM_score'] = -rfm_norm['recency'] + rfm_norm['frequency'] + rfm_norm['quantity']

rfm.loc[retCustomers_index, 'cluster'] = rfm_norm.loc[retCustomers_index, 'cluster']
rfm.loc[retCustomers_index, 'RFM_score'] = rfm_norm.loc[retCustomers_index, 'RFM_score']
rfm.cluster.fillna('New Customers', inplace=True)

def cluster_naming(rfm_norm):
    df = rfm_norm.groupby('cluster')['RFM_score'].max().sort_values(ascending=False)
    cluster_orders = df.index
    i = 0
    for cluster in cluster_orders:
        index = rfm_norm[rfm_norm['cluster'] == cluster].index
        rfm_norm.loc[index, 'cluster'] = cluster_names[i]
        i += 1
    return rfm_norm

rfm_norm = cluster_naming(rfm_norm)
rfm.loc[retCustomers_index, 'cluster'] = rfm_norm.loc[retCustomers_index, 'cluster']
rfm.loc[:, 'date'] = str(date.today())
rfm = rfm.merge(right=df[[
    'CustomerCode', 'Customer']].drop_duplicates(), how='left', on='CustomerCode')
rfm= rfm.merge(right= quantity_fbb, how= 'left', on= 'CustomerCode')
rfm= rfm.merge(right= quantity_paper, how= 'left', on= 'CustomerCode')
rfm= rfm.merge(right= quantity_pulp, how= 'left', on= 'CustomerCode')
rfm.fillna(0, inplace= True)
rfm= rfm.sort_values(by= 'RFM_score', ascending= False)
rfm.to_excel('rfm_Mahsa_report_product_splits_1402.xlsx', index= False)
