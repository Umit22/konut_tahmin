# Kütüphaneleri ekleme

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
pd.options.display.float_format = '{:.2f}'.format

# Verileri pandas dataframe aktarma

df = pd.read_csv('Bengaluru_House_Data.csv')

#Gereksiz kolonları temizleme

df = df.drop(['area_type', 'availability', 'society'], axis='columns')

# Null değerleri belirleyip gerekli işlemleri yapma

df.isna().sum()

df = df.dropna()
df.isna().sum()

df['size'].apply(lambda x: int(x.split(' ')[0]))

df['oda'] = df['size'].apply(lambda x: int(x.split(' ')[0]))

df = df.drop(['size'], axis='columns')



def sayi_mi(x):
    try:
        float(x)
    except:
        return False
    return True

df[~df['total_sqft'].apply(sayi_mi)]

df2 = df[df['total_sqft'].apply(sayi_mi)==False]

df2[df2['total_sqft'].str.contains("Meter")]

def convert_range_to_num(x):
    tokens = str(x).split('-')
    if len(tokens) == 2:
        res = (float(tokens[0]) + float(tokens[1])) / 2
    else:
        try:
            res = float(x)
        except:
            res = None
    return res

df['total_sqft'] = df['total_sqft'].apply(convert_range_to_num)

df = df.dropna()

print(df.head())

df.to_csv('cleaned_data.csv', index=False)
