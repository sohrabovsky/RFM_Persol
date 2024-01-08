from functools import reduce
import pandas as pd
import numpy as np
import jalali_pandas
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import pyodbc
from datetime import date
from operator import attrgetter

# Specifying the ODBC driver, server name, database, etc. directly
cnxn = pyodbc.connect(
    'DRIVER={ODBC Driver 17 for SQL Server};SERVER=192.168.10.41;DATABASE=ReportDB;UID=SelfServiceBI;PWD=MobbMobb66!')
df_sales = pd.read_sql("SELECT * FROM dbo.FullInvoiceReport", cnxn)
df_category = pd.read_sql('SELECT * FROM dbo.ProductCategory', cnxn)
df_debts = pd.read_sql("SELECT * FROM dbo.ReceiptRequewstAndInvoiceDebits", cnxn)
df_cheques= pd.read_sql("SELECT * FROM dbo.ReceivableCheques", cnxn)
df = df_sales.merge(right=df_category, how='left', on='ProductCode')
df.rename(columns={'G1': 'Category'}, inplace=True)
df.drop(
    columns=[i for i in df.columns[df.columns.str.contains('G')]], inplace=True)
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
shayan_customer_code = df_wood[df_wood['Customer']
                               == 'شایان حقیقی']['CustomerCode'].unique()[0]
syamak_file = pd.read_excel(r"C:\Users\mk-05\Documents\Python Scripts\rfm\syamak.xlsx", usecols=['shayan'])
syamak_file = syamak_file.drop(index=15).reset_index(drop=True)
syamak_file['family'] = syamak_file['shayan'].apply(lambda x: x.split()[-1])
df_wood['family'] = df_wood['Customer'].apply(lambda x: x.split()[-1])
index = df_wood[df_wood['family'].isin(syamak_file['family'].unique())].index
df_wood.drop(index= index, inplace= True)
df_wood= df_wood.reset_index(drop= True)
df_cellulosic= df_cellulosic.reset_index(drop= True)
# If we want to separate Cellulosic departments
mahsa_customers= pd.read_excel('MRB cellulose customers.xlsx')
mahsa_customers= mahsa_customers[mahsa_customers['is_mahsa'] == True].customer.unique()
df_cellulosic_mahsa= df_cellulosic[df_cellulosic['Customer'].isin(mahsa_customers)]
df_cellulosic_alireza= df_cellulosic[~df_cellulosic['Customer'].isin(mahsa_customers)]
df_cellulosic_mahsa= df_cellulosic_mahsa.reset_index(drop= True)
df_cellulosic_alireza= df_cellulosic_alireza.reset_index(drop= True)
# Clusters Definitions
cluster_names = ['champion', 'loyal', 'promising','needing_attention']
# Credits per cluster
credit={'champion': 25000000000,
        'loyal': 20000000000,
        'promising': 7500000000,
        'needing_attention': 0}
first_month_penalty= 1.05
second_month_penalty= 1.1
# Debits
df_debts_positive= df_debts[df_debts['FinalElapsedDays'] < 0]
index= df_debts_positive[df_debts_positive['docType'] == 'اسناد دريافتني'].index
df_debts_positive.loc[index, 'Title']= df_debts_positive[df_debts_positive['docType'] == 'اسناد دريافتني']['Title'].str[10:]
# Cheques
df_cheques= pd.read_sql("SELECT * FROM dbo.ReceivableCheques", cnxn)
df_cheques_positive= df_cheques[df_cheques['ChequeStatus'] == 'واخواست شده']
df_cheques_positive_customers= pd.DataFrame(columns= ['cheque', 'returned'])
df_cheques_positive_customers.loc[:,'cheque']= df_cheques_positive['trimmedCustomer'].unique()
df_cheques_positive_customers.loc[:,'returned']= True
df_debts_positive_grouped= df_debts_positive.groupby(['Title'])['Debit', 'FinalElapsedDays'].sum()
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
def cluster_naming(df):
    dff = df.groupby('cluster')['RFM_score'].max().sort_values(ascending= False)
    cluster_orders = dff.index
    i = 0
    for cluster in cluster_orders:
        index = df[df['cluster'] == cluster].index
        df.loc[index, 'cluster'] = cluster_names[i]
        i += 1
    return df
def credit_system(df):
    dff= df.merge(df_debts_positive_grouped, how= 'left', left_on= 'Customer', right_on= 'Title')
    dfff= dff.merge(df_cheques_positive_customers, how= 'left', left_on= 'Customer', right_on= 'cheque')
    dfff.returned.fillna(False, inplace= True)
    dfff['FinalElapsedDays'].fillna(0, inplace= True)
    dfff['Debit'].fillna(0, inplace= True)
    for cluster in cluster_names:
        index= dfff[(dfff['cluster'] == cluster) & (dfff['FinalElapsedDays'] >= -30)].index
        dfff.loc[index, 'credit'] = credit[cluster]- dfff.loc[index, 'Debit']
        index= dfff[(dfff['cluster'] == cluster) & (dfff['FinalElapsedDays'] >= -60) & (dfff['FinalElapsedDays'] < -30)].index
        dfff.loc[index, 'credit'] = credit[cluster]- dfff.loc[index, 'Debit'] * first_month_penalty
        index= dfff[(dfff['cluster'] == cluster) & (dfff['FinalElapsedDays'] >= -90) & (dfff['FinalElapsedDays'] < -60)].index
        dfff.loc[index, 'credit'] = credit[cluster]- dfff.loc[index, 'Debit'] * second_month_penalty
        index= dfff[(dfff['cluster'] == cluster) & (dfff['FinalElapsedDays'] < -90)].index
        dfff.loc[index, 'credit'] = 0
    index= dfff[dfff['returned'] == True].index
    dfff.loc[index, 'credit'] = 0
    return dfff
def rfm_calculations(df):
    monetary = df.groupby('CustomerCode')['NetAmount'].sum()
    monetary = monetary.reset_index()
    quantity = df.groupby('CustomerCode')['Quantity'].sum()
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
    dataframes = [recency, frequency, quantity, latency, monetary]
    rfm = reduce(lambda df1, df2: pd.merge(
    left=df1, right=df2, how='inner', on='CustomerCode'), dataframes)
    rfm.rename(columns={
        'Diff_x': 'recency',
        'InvoiceNumber': 'frequency',
        'Quantity': 'quantity',
        'Diff_y': 'latency',
        'NetAmount': 'monetary'
    }, inplace=True)
    new_customer_window_size= mean_purchase_widnow(df)
    retCustomers_index = rfm[rfm['latency'] > new_customer_window_size].index
    scaler = StandardScaler()
    rfm_norm = pd.DataFrame(scaler.fit_transform(rfm[['recency', 'frequency', 'quantity']]), columns=['recency', 'frequency', 'quantity'])
    rfm_norm = rfm_norm.loc[retCustomers_index, :]
    kmeans = KMeans(n_clusters=4, max_iter=50)
    kmeans.fit(rfm_norm[['recency', 'frequency', 'quantity']])
    rfm_norm['cluster'] = kmeans.labels_
    ### Scoring 
    rfm_norm['RFM_score']= -rfm_norm['recency'] + rfm_norm['frequency'] + rfm_norm['quantity']
    rfm.loc[retCustomers_index,'cluster'] = rfm_norm.loc[retCustomers_index, 'cluster']
    rfm.loc[retCustomers_index,'RFM_score'] = rfm_norm.loc[retCustomers_index, 'RFM_score']
    rfm.cluster.fillna('New Customers', inplace=True)
    rfm.loc[:, 'new_customer_window_size']= new_customer_window_size
    rfm_norm= cluster_naming(rfm_norm)
    rfm.loc[retCustomers_index,'cluster'] = rfm_norm.loc[retCustomers_index, 'cluster']
    rfm = rfm.merge(right=df[[
                          'CustomerCode', 'Customer']].drop_duplicates(), how='left', on='CustomerCode')
    rfm.rename(columns= {
        'recency' : 'recency_days'
    }, inplace= True)
    rfm_with_credit= credit_system(rfm)
    rfm_with_credit.drop(columns= ['cheque'], inplace= True)
    return rfm_with_credit
# RFM Wood
rfm_wood= rfm_calculations(df_wood)
rfm_wood= rfm_wood.sort_values(by= 'RFM_score', ascending= False)
# RFM Cellulosic
rfm_cellulosic_mahsa= rfm_calculations(df_cellulosic_mahsa)
rfm_cellulosic_mahsa= rfm_cellulosic_mahsa.sort_values(by= 'RFM_score', ascending= False)
rfm_cellulosic_alireza= rfm_calculations(df_cellulosic_alireza)
rfm_cellulosic_alireza= rfm_cellulosic_alireza.sort_values(by= 'RFM_score', ascending= False)
# RFM Chemical - Polymer
rfm_chemical= rfm_calculations(df_chemical_polymer)
rfm_chemical= rfm_chemical.sort_values(by= 'RFM_score', ascending= False)
# Writing to DataBase
import psycopg2
from sqlalchemy import create_engine
conn= psycopg2.connect(
    user= "postgres",
    password= "&rXK(6(N4(6)=e{b",
    host= 'localhost',
    port= '5432',
    database= 'customerDB'
)
engine = create_engine('postgresql://postgres:&rXK(6(N4(6)=e{b@localhost:5432/customerDB')
# RFM with credit for wood:
rfm_wood.to_sql("wood_rfm_with_credit", con= engine, if_exists= 'replace', index= False)
rfm_cellulosic_mahsa.to_sql("cellulosic1_rfm_with_credit", con= engine, if_exists= 'replace', index= False)
rfm_cellulosic_alireza.to_sql("cellulosic2_rfm_with_credit", con= engine, if_exists= 'replace', index= False)
rfm_chemical.to_sql("chemical_rfm_with_credit", con= engine, if_exists= 'replace', index= False)
engine.dispose()
conn.close()
