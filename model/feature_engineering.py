import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

# veri_temizleme' de temizlenen data'yı dosyaya ekleme.
df = pd.read_csv('cleaned_data.csv')

# total_sqft'u bildiğimiz değere yani metrekareye çevirme.
df['metrekare'] = df['total_sqft'] * 0.092903

# metrekare başına düşen fiyatı hesaplama
df['price_per_m2'] = df['price'] * 1_000 / df['metrekare']

# strip fonksiyonu ile başında ki ve sonunda ki boşlukları kaldırma.
df.location = df.location.apply(lambda x: x.strip())

# locationların hepsini bir df'ye çekmek. Analiz yapabilmek için.
loc_stats = df.groupby(['location'])['location'].size().reset_index(name='counts').sort_values(by=['counts'],ascending = False)

#histogramını çizdirme
matplotlib.rcParams['figure.figsize'] = (20,10)

plt.xlabel("lokasyonlar counts değeri")
plt.ylabel("kaç lokasyon bu durumda")

bins = 10
data = loc_stats.counts
arr = plt.hist(data, bins=bins, range=(0,100))
for i in range(bins):
    plt.text(arr[1][i],arr[0][i],str(arr[0][i]))
# plt.show() Grafiği görüntülemek

# 1-10 arası lokasyonları diğer olarak birleştirerek model eğitimini kolaylaştırmak gerekiyor.
df.groupby(['location'])['location'].size().reset_index(name='counts').sort_values(by=['counts'],ascending =False)

# print(len(loc_stats[loc_stats.counts<=10])) #bir adreste 10dan daha az daire olan lokasyonların değeri

#10'dan az olanları listeleme
less_than_10 = loc_stats[loc_stats.counts<=10]
# print(less_than_10.location.to_list())

# eğer x değeri bu liste içerisinde ise bunu other olarak değiştir. Yoksa x olduğu değerde kalsın.
df.location = df.location.apply(lambda x: 'other' if x in less_than_10.location.to_list() else x)
# print(df[df.location == 'other'])

# print(df.head(30)) (other değerleri kontrol edildi.)

df.to_csv('feature_engineering_data.csv', index=False)