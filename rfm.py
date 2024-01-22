from functools import reduce
import pandas as pd
import numpy as np
import jalali_pandas
import jdatetime
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import pyodbc
from datetime import date
from operator import attrgetter
import warnings
warnings.filterwarnings("ignore")
# Specifying the ODBC driver, server name, database, etc. directly
cnxn = pyodbc.connect(
    'DRIVER={ODBC Driver 17 for SQL Server};SERVER=192.168.10.41;DATABASE=ReportDB;UID=SelfServiceBI;PWD=MobbMobb66!')
df_sales = pd.read_sql("SELECT * FROM dbo.FullInvoiceReport", cnxn)
df_category = pd.read_sql('SELECT * FROM dbo.ProductCategory', cnxn)
df_debts = pd.read_sql("SELECT * FROM dbo.ReceiptRequewstAndInvoiceDebits", cnxn)
df_cheques = pd.read_sql("SELECT * FROM dbo.ReceivableCheques", cnxn)
df = df_sales.merge(right=df_category, how='left', on='ProductCode')
df.rename(columns={'G1': 'Category'}, inplace=True)
df.drop(
    columns=[i for i in df.columns[df.columns.str.contains('G')]], inplace=True)
df_wood = df[df['Category'] == 'Wood Products'].reset_index(drop=True)
# Eliminating products cleaning (Wood):
wood_eliminated_products = ['تخته بریوزا (غان) ', 'چوب نیم گرد', 'چوب گرده بینه (Debarked & cut wood Log)']
df_wood[~df_wood['ProductName'].isin(wood_eliminated_products)]
df_cellulosic = df[df['Category'] ==
                   'Cellulosic Products'].reset_index(drop=True)
df_chemical_polymer = df[df['Category'] ==
                         'Chemical & Polymer Products'].reset_index(drop=True)
# Eliminating products cleaning (Cellulosic):
df_cellulosic = df_cellulosic[
    ~(df_cellulosic['ProductName'].str.contains('GB')) | (df_cellulosic['ProductName'].str.contains('پشت طوسی'))]
shayan_customer_code = df_wood[df_wood['Customer']
                               == 'شایان حقیقی']['CustomerCode'].unique()[0]
syamak_file = pd.read_excel(r"C:\Users\mk-05\Documents\Python Scripts\rfm\syamak.xlsx", usecols=['shayan'])
syamak_file = syamak_file.drop(index=15).reset_index(drop=True)
syamak_file['family'] = syamak_file['shayan'].apply(lambda x: x.split()[-1])
df_wood['family'] = df_wood['Customer'].apply(lambda x: x.split()[-1])
index = df_wood[df_wood['family'].isin(syamak_file['family'].unique())].index
df_wood= df_wood[~df_wood['CustomerCode'].isin(['2022091', '2021617', '2022112', '2021954', '3022613'])] # Deleting پالت سازان و سانا تجارت و چوب و کاغذ مازندران, هادی واقفی، سیامک اشکان فر
df_wood.drop(index=index, inplace=True)
df_wood = df_wood.reset_index(drop=True)
df_cellulosic = df_cellulosic.reset_index(drop=True)
# If we want to separate Cellulosic departments
mahsa_team_employeeID = ['16134', '16114'] # Mahsa RB and Elmira Codes
alireza_team_employeeID = ['16020', '16108'] # Alireza and Mahsa GH
df_cellulosic_mahsa = df_cellulosic[df_cellulosic['SalesEmployeeID'].isin(mahsa_team_employeeID)]
df_cellulosic_mahsa = df_cellulosic_mahsa[df_cellulosic_mahsa['InvoiceDate'] >= '1402/01/01']   # Mahsa for recent year
df_cellulosic_alireza = df_cellulosic[df_cellulosic['SalesEmployeeID'].isin(alireza_team_employeeID)]
df_cellulosic_alireza= df_cellulosic_alireza[df_cellulosic_alireza['CustomerCode'] != '2010262']
df_cellulosic_mahsa = df_cellulosic_mahsa.reset_index(drop=True)
df_cellulosic_alireza = df_cellulosic_alireza.reset_index(drop=True)
# Clusters Definitions
cluster_names = ['champion', 'loyal', 'promising', 'needing_attention']
# Credits per cluster
mahsa_credit_cellulosic = {'champion': 25000000000,
                'loyal': 15000000000,
                'promising': 2000000000,
                'needing_attention': 0,
                'New Customers': 0}
alireza_credit_cellulosic = {'champion': 10000000000,
                           'loyal': 7000000000,
                           'promising': 1000000000,
                           'needing_attention': 0,
                            'New Customers': 0}
credit_wood = {'champion': 40000000000,
               'loyal': 5000000000,
               'promising': 1000000000,
               'needing_attention': 0,
               'New Customers' : 0}
credit_chemical = {'champion': 13000000000,
                   'loyal': 6000000000,
                   'promising': 1000000000,
                   'needing_attention': 0,
                   'New Customers': 0}
first_month_penalty = 1.05
second_month_penalty = 1.1
# Debits
df_debts_positive = df_debts[df_debts['FinalElapsedDays'] < 0]
index = df_debts_positive[df_debts_positive['docType'] == 'اسناد دريافتني'].index
df_debts_positive.loc[index, 'Title'] = df_debts_positive[df_debts_positive['docType'] == 'اسناد دريافتني'][
                                            'Title'].str[10:]
# Cheques
df_cheques = pd.read_sql("SELECT * FROM dbo.ReceivableCheques", cnxn)
df_cheques_positive = df_cheques[df_cheques['ChequeStatus'] == 'واخواست شده']
df_cheques_positive_customers = pd.DataFrame(columns=['cheque', 'bouncedCheck'])
df_cheques_positive_customers.loc[:, 'cheque'] = df_cheques_positive['trimmedCustomer'].unique()
df_cheques_positive_customers.loc[:, 'bouncedCheck'] = True
df_debts_positive_grouped = df_debts_positive.groupby(['Title'])['Debit', 'FinalElapsedDays'].sum()


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


def cluster_naming(df):
    dff = df.groupby('cluster')['RFM_score'].max().sort_values(ascending=False)
    cluster_orders = dff.index
    i = 0
    for cluster in cluster_orders:
        index = df[df['cluster'] == cluster].index
        df.loc[index, 'cluster'] = cluster_names[i]
        i += 1
    return df


def credit_system(df, credit_dictionary):
    dff = df.merge(df_debts_positive_grouped, how='left', left_on='Customer', right_on='Title')
    dfff = dff.merge(df_cheques_positive_customers, how='left', left_on='Customer', right_on='cheque')
    dfff.bouncedCheck.fillna(False, inplace=True)
    dfff['FinalElapsedDays'].fillna(0, inplace=True)
    dfff['Debit'].fillna(0, inplace=True)
    for cluster in cluster_names:
        index = dfff[(dfff['cluster'] == cluster) & (dfff['FinalElapsedDays'] >= -30)].index
        dfff.loc[index, 'credit (IRR)'] = credit_dictionary[cluster] - dfff.loc[index, 'Debit']
        dfff.loc[index, "Base Credit"]= credit_dictionary[cluster]

        index = dfff[
            (dfff['cluster'] == cluster) & (dfff['FinalElapsedDays'] >= -60) & (dfff['FinalElapsedDays'] < -30)].index
        dfff.loc[index, 'credit (IRR)'] = credit_dictionary[cluster] - dfff.loc[index, 'Debit'] * first_month_penalty
        dfff.loc[index, "Base Credit"]= credit_dictionary[cluster]

        index = dfff[
            (dfff['cluster'] == cluster) & (dfff['FinalElapsedDays'] >= -90) & (dfff['FinalElapsedDays'] < -60)].index
        dfff.loc[index, 'credit (IRR)'] = credit_dictionary[cluster] - dfff.loc[index, 'Debit'] * second_month_penalty
        dfff.loc[index, "Base Credit"]= credit_dictionary[cluster]

        index = dfff[(dfff['cluster'] == cluster) & (dfff['FinalElapsedDays'] < -90)].index
        dfff.loc[index, 'credit (IRR)'] = 0
        dfff.loc[index, "Base Credit"]= credit_dictionary[cluster]

    index = dfff[dfff['bouncedCheck'] == True].index
    dfff.loc[index, 'credit (IRR)'] = 0

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
        'NetAmount': 'monetary (IRR)'
    }, inplace=True)
    new_customer_window_size = mean_purchase_widnow(df)
    retCustomers_index = rfm[rfm['latency'] > new_customer_window_size].index
    scaler = StandardScaler()
    rfm_norm = pd.DataFrame(scaler.fit_transform(rfm[['recency', 'frequency', 'quantity']]),
                            columns=['recency', 'frequency', 'quantity'])
    rfm_norm = rfm_norm.loc[retCustomers_index, :]
    kmeans = KMeans(n_clusters=4, max_iter=50)
    kmeans.fit(rfm_norm[['recency', 'frequency', 'quantity']])
    rfm_norm['cluster'] = kmeans.labels_
    ### Scoring 
    rfm_norm['RFM_score'] = -rfm_norm['recency'] + rfm_norm['frequency'] + rfm_norm['quantity']
    rfm.loc[retCustomers_index, 'cluster'] = rfm_norm.loc[retCustomers_index, 'cluster']
    rfm.loc[retCustomers_index, 'RFM_score'] = rfm_norm.loc[retCustomers_index, 'RFM_score']
    rfm.cluster.fillna('New Customers', inplace=True)
    rfm.loc[:, 'new_customer_window_size (days)'] = new_customer_window_size
    rfm_norm = cluster_naming(rfm_norm)
    rfm.loc[retCustomers_index, 'cluster'] = rfm_norm.loc[retCustomers_index, 'cluster']
    index= rfm[rfm['cluster'] == 'New Customers'].index
    rfm.loc[index, 'RFM_score']= -100
    rfm.loc[:, 'date'] = str(jdatetime.date.today())
    rfm = rfm.merge(right=df[[
        'CustomerCode', 'Customer']].drop_duplicates(), how='left', on='CustomerCode')
    rfm.rename(columns={
        'recency': 'recency (days)'
    }, inplace=True)
    return rfm


# RFM Wood
rfm_wood = rfm_calculations(df_wood)
rfm_wood_with_credit = credit_system(rfm_wood, credit_wood)
rfm_wood_with_credit.drop(columns=['cheque'], inplace=True)
rfm_wood_with_credit = rfm_wood_with_credit.sort_values(by='RFM_score', ascending=False)
# RFM Cellulosic
rfm_cellulosic_mahsa = rfm_calculations(df_cellulosic_mahsa)
rfm_cellulosic_mahsa_with_credit = credit_system(rfm_cellulosic_mahsa, mahsa_credit_cellulosic)
rfm_cellulosic_mahsa_with_credit.drop(columns=['cheque'], inplace=True)
rfm_cellulosic_mahsa_with_credit = rfm_cellulosic_mahsa_with_credit.sort_values(by='RFM_score', ascending=False)
rfm_cellulosic_alireza = rfm_calculations(df_cellulosic_alireza)
rfm_cellulosic_alireza_with_credit = credit_system(rfm_cellulosic_alireza, alireza_credit_cellulosic)
rfm_cellulosic_alireza_with_credit.drop(columns=['cheque'], inplace=True)
rfm_cellulosic_alireza_with_credit = rfm_cellulosic_alireza_with_credit.sort_values(by='RFM_score', ascending=False)
# RFM Chemical - Polymer
rfm_chemical = rfm_calculations(df_chemical_polymer)
rfm_chemical_with_credit = credit_system(rfm_chemical, credit_chemical)
rfm_chemical_with_credit.drop(columns=['cheque'], inplace=True)
rfm_chemical_with_credit = rfm_chemical_with_credit.sort_values(by='RFM_score', ascending=False)

