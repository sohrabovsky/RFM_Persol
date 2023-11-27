from functools import reduce
import requests
import pandas as pd
import numpy as np
import jalali_pandas
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


df_wood = df[df['Category'] == 'Wood Products'].reset_index(drop=True)

# Eliminating products cleaning (Wood):
wood_eliminated_products= ['تخته بریوزا (غان) ', 'چوب نیم گرد', 'چوب گرده بینه (Debarked & cut wood Log)']
df_wood[~df_wood['ProductName'].isin(wood_eliminated_products)]



df_cellulosic = df[df['Category'] ==
                   'Cellulosic Products'].reset_index(drop=True)
df_chemical_polymer = df[df['Category'] ==
                         'Chemical & Polymer Products'].reset_index(drop=True)

# Eliminating products cleaning (Cellulosic):
df_cellulosic= df_cellulosic[~(df_cellulosic['ProductName'].str.contains('GB')) | (df_cellulosic['ProductName'].str.contains('پشت طوسی'))]

# If we want to separate Cellulosic departments
mahsa_customers= pd.read_excel('MRB cellulose customers.xlsx')
mahsa_customers= mahsa_customers[mahsa_customers['is_mahsa'] == True].customer.unique()

df_cellulosic_mahsa= df_cellulosic[df_cellulosic['Customer'].isin(mahsa_customers)]

df_cellulosic_alireza= df_cellulosic[~df_cellulosic['Customer'].isin(mahsa_customers)]

shayan_customer_code = df_wood[df_wood['Customer']
                               == 'شایان حقیقی']['CustomerCode'].unique()[0]
syamak_file = pd.read_excel(
    r'C:\Users\mk-05\Documents\Python Scripts\syamak.xlsx', usecols=['shayan'])
syamak_file = syamak_file.drop(index=15).reset_index(drop=True)
syamak_file['family'] = syamak_file['shayan'].apply(lambda x: x.split()[-1])
df_wood['family'] = df_wood['Customer'].apply(lambda x: x.split()[-1])

index = df_wood[df_wood['family'].isin(syamak_file['family'].unique())].index

df_wood.drop(index= index, inplace= True)
df_wood= df_wood.reset_index(drop= True)
df_cellulosic_mahsa= df_cellulosic_mahsa.reset_index(drop= True)
df_cellulosic_alireza= df_cellulosic_alireza.reset_index(drop= True)


cluster_names = ['Champion', 'Loyal', 'Promising','Needing Attention']

def rfm_calculations(df):
    quantity = df.groupby('CustomerCode')['Quantity'].sum()    # Monetary here is Quantity because of inflation effect
    quantity = quantity.reset_index()

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
                diff = temp_df.loc[i+1, 'jdate'] - temp_df.loc[i, 'jdate']
                diff_dates.append(diff)
                mean_diff = np.mean(diff_dates)
                dff.loc[customer, 'mean_diff'] = mean_diff
        
        return dff.mean().apply(attrgetter('days'))[0]
    new_customer_window_size= mean_purchase_widnow(df)
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
    rfm_norm['RFM_score']= -rfm_norm['recency'] + rfm_norm['frequency'] + rfm_norm['quantity']

    
    rfm.loc[retCustomers_index,'cluster'] = rfm_norm.loc[retCustomers_index, 'cluster']
    rfm.loc[retCustomers_index,'RFM_score'] = rfm_norm.loc[retCustomers_index, 'RFM_score']
    rfm.cluster.fillna('New Customers', inplace=True)

    def cluster_naming(rfm_norm):
        df = rfm_norm.groupby('cluster')['RFM_score'].max().sort_values(ascending= False)
        cluster_orders = df.index
        i = 0
        for cluster in cluster_orders:
            index = rfm_norm[rfm_norm['cluster'] == cluster].index
            rfm_norm.loc[index, 'cluster'] = cluster_names[i]
            i += 1
        return rfm_norm
    rfm_norm= cluster_naming(rfm_norm)
    rfm.loc[retCustomers_index,'cluster'] = rfm_norm.loc[retCustomers_index, 'cluster']
    rfm.loc[:, 'date'] = str(date.today())
    rfm = rfm.merge(right=df[[
                          'CustomerCode', 'Customer']].drop_duplicates(), how='left', on='CustomerCode')
    return(rfm)

# RFM Wood
rfm_wood= rfm_calculations(df_wood)
rfm_wood.loc[:, 'vertical']= 'wood'
rfm_wood= rfm_wood.sort_values(by= 'RFM_score', ascending= False)

# RFM Cellulosic
rfm_cellulosic_mahsa= rfm_calculations(df_cellulosic_mahsa)
rfm_cellulosic_mahsa.loc[:, 'vertical']= 'cellulosic_mahsa'
rfm_cellulosic_mahsa= rfm_cellulosic_mahsa.sort_values(by= 'RFM_score', ascending= False)

rfm_cellulosic_alireza= rfm_calculations(df_cellulosic_alireza)
rfm_cellulosic_alireza.loc[:, 'vertical']= 'cellulosic_alireza'
rfm_cellulosic_alireza= rfm_cellulosic_alireza.sort_values(by= 'RFM_score', ascending= False)

# RFM Chemical - Polymer
rfm_chemical= rfm_calculations(df_chemical_polymer)
rfm_chemical.loc[:, 'vertical']= 'chemical'
rfm_chemical= rfm_chemical.sort_values(by= 'RFM_score', ascending= False)

def plotting(rfm, name):
    fig = plt.figure(figsize=(8, 6))                
    ax = fig.add_subplot(projection='3d')
    for cluster in cluster_names:
        x = rfm[rfm['cluster'] == cluster]['recency']
        y = rfm[rfm['cluster'] == cluster]['frequency']
        z = rfm[rfm['cluster'] == cluster]['quantity']/1000
        ax.scatter(x, y, z, label=cluster)
        ax.set_xlabel('Recency (days)')
        ax.set_ylabel('Frequency')
        ax.set_zlabel("Quantity (ton)", rotation=90)
        ax.zaxis.labelpad = -0.2 

    plt.legend()
    plt.title(f'RFM Segmentation for {name} Product')
    plt.savefig(path + '\\' + f'rfm_{name}.jpeg')

plotting(rfm_wood, 'Wood')
plotting(rfm_cellulosic_mahsa, 'Cellulosic_Mahsa')
plotting(rfm_cellulosic_alireza, 'Cellulosic_Alireza')
plotting(rfm_chemical, 'Chemical')
df= pd.concat([rfm_wood, rfm_cellulosic_mahsa, rfm_cellulosic_alireza, rfm_chemical])
df.to_excel('rfm_segmentation.xlsx', index= False)