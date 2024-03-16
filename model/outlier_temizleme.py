import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv("feature_engineering_data.csv")

# Oda metrekaresi düşük olanları silme
df = df[~(df['total_sqft']/df['oda'] < 300)]
df.shape

def remove_ppm_outliers(df):
    df_out = pd.DataFrame()
    for key, subdf in df.groupby('location'):
        lower_limit = subdf.price_per_m2.mean() - ( 1 * subdf.price_per_m2.std() ) # Bu grubun ortalamasını buluyor. Daha sonra standart sapmayı hesaplıyor. Bir standart sapma çıkarıyor.
        upper_limit = subdf.price_per_m2.mean() + ( 1 * subdf.price_per_m2.std() )

        reduced_df = subdf[ ( subdf.price_per_m2 > lower_limit) & (subdf.price_per_m2 < upper_limit) ]
        df_out = pd.concat([df_out, reduced_df], ignore_index=True)

    return df_out

df3 = remove_ppm_outliers(df)
# print(df3)

def plot_scatter_chart(df,location):
    bhk2 = df[(df.location == location) & (df.oda==2)]
    bhk3 = df[(df.location == location) & (df.oda==3)]
    #matplotlib.rcParams['figure.figsize'] = (15, 10)
    plt.scatter(bhk2.total_sqft,bhk2.price, color='blue', label='2 BHK', s=50)
    plt.scatter(bhk3.total_sqft, bhk3.price, marker='+', color='green', label='3 BHK', s=50)
    plt.xlabel("Total Square Feet Area")
    plt.ylabel("Price (Lakh Indian Rupees)")
    plt.title(location)
    plt.legend()

#plot_scatter_chart(df3, "Rajaji Nagar") grafiklerin karışmaması için yorum satırına alındı.
#plt.show()

#plot_scatter_chart(df3, "Hebbal") grafiklerin karışmaması için yorum satırına alındı.
#plt.show()

"""
Aynı konum için 3 yatak odalı bir dairenin fiyatının, aynı metrekare alana sahip 2 yatak odalı bir daireden
daha düşük olduğu özellikleri de kaldırmalıyız. Yapacağımız şey, belirli bir konum için bir BHK (yatak odası sayısı)
başına bir istatistikler sözlüğü oluşturmaktır.

{
    {
    '1' : {
        'mean': 4000,
        'std: 2000,
        'count': 34
    },
    '2' : {
        'mean': 4300,
        'std: 2300,
        'count': 22
    },    
}

Şimdi, 1 yatak odalı dairelerin ortalama metrekare başına fiyatının altında olan metrekare başına fiyatı daha düşük olan 2 yatak odalı daireleri de kaldırabiliriz.
"""

def remove_oda_outliers(df):
    exclude_indices = np.array([])
    for location, location_df in df.groupby('location'):
        oda_stats = {}
        for oda, oda_df in location_df.groupby('oda'):
            oda_stats[oda] = {
                'mean': np.mean(oda_df.price_per_m2),
                'std': np.std(oda_df.price_per_m2),
                'count': oda_df.shape[0]
            }
        for oda, oda_df in location_df.groupby('oda'):
            stats = oda_stats.get(oda-1)
            if stats and stats['count']>5:
                exclude_indices = np.append(exclude_indices, oda_df[oda_df.price_per_m2<(stats['mean'])].index.values)
    return df.drop(exclude_indices,axis='index')

df4 = remove_oda_outliers(df3)
#print(df4.shape)

"""
import matplotlib
matplotlib.rcParams["figure.figsize"] = (20,10)
plt.hist(df4.price_per_m2,rwidth=0.8)
plt.xlabel("Price Per Square Feet")
plt.ylabel("Count")
plt.show()
Daha sonra ki grafiklere etki etmemesi için yorum satırına aldım. 
"""
df5 = df4[df4.bath<df4.oda+2]

df6 = df5.drop(['balcony', 'price_per_m2', 'metrekare'],axis='columns')

df6.rename(columns={'oda':'bhk'}, inplace=True, errors='raise')
df6.head(3)

df6.to_csv('outlier_temizleme.csv', index=False)











