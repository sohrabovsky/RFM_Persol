import pandas as pd
from rfm_segmentation.rfm_functools import import_from_rahkaran, rfm_calculations, mean_purchase_widnow, export_to_db, cluster_naming
df= import_from_rahkaran()
df_wood = df[df['Category'] == 'Wood Products'].reset_index(drop=True)
# Eliminating products cleaning (Wood):
wood_eliminated_products= ['تخته بریوزا (غان) ', 'چوب نیم گرد', 'چوب گرده بینه (Debarked & cut wood Log)']
df_wood[~df_wood['ProductName'].isin(wood_eliminated_products)]
shayan_customer_code = df_wood[df_wood['Customer'] == 'شایان حقیقی']['CustomerCode'].unique()[0]
syamak_file = pd.read_excel('syamak.xlsx', usecols=['shayan'])
syamak_file = syamak_file.drop(index=15).reset_index(drop=True)
syamak_file['family'] = syamak_file['shayan'].apply(lambda x: x.split()[-1])
df_wood['family'] = df_wood['Customer'].apply(lambda x: x.split()[-1])
index = df_wood[df_wood['family'].isin(syamak_file['family'].unique())].index
df_wood.drop(index= index, inplace= True)
df_wood= df_wood.reset_index(drop= True)
cluster_names = ['Champion', 'Loyal', 'Promising','Needing Attention']
rfm_wood= rfm_calculations(df_wood)
rfm_wood= rfm_wood.sort_values(by= 'RFM_score', ascending= False)
export_to_db(rfm_wood, "RFM_Table")