# -*- coding: utf-8 -*-

# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# %matplotlib inline
import seaborn as sns

from google.colab import files

"""# Kaggle setup"""

# Install kaggle
!pip install -q kaggle

# Unggah file json yang diunduh dari akun kaggle
uploaded = files.upload()

# Buat direktori kaggle dan pindahkan file yang diunggah ke folder baru
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle

# Izin baca agar dapat diakses di Google Colab
!chmod 600 /root/.kaggle/kaggle.json

"""## Mengunduh dan Menyiapkan Dataset

![image](https://user-images.githubusercontent.com/87566521/139109868-1ef63ec2-d447-468a-926c-18691e8bd070.png)

####**Informasi Dataset :**

Jenis | Informasi
--- | ---
Sumber | [Kaggle Dataset : Anime Recommendations Database](https://www.kaggle.com/CooperUnion/anime-recommendations-database)
Lisensi | CC0: Public Domain
Kategori | Anime, Manga
Rating Pengunaan | 8.2 (Gold)
Jenis dan Ukuran Berkas | CSV (112 MB)
"""

# Unduh dataset yang akan digunakan
!kaggle datasets download -d CooperUnion/anime-recommendations-database

# Mengekstrak berkas zip ke direktori
!unzip -q /content/anime-recommendations-database.zip -d .

"""## Univariate Exploratory Data Analysis"""

# Memuat data pada sebuah dataframe menggunakan pandas
df_anime = pd.read_csv('/content/anime.csv')
df_rating = pd.read_csv('/content/rating.csv')

print ('Bentuk data (baris, kolom):'+ str(df_anime.shape))
print ('Bentuk data (baris, kolom):'+ str(df_rating.shape))

"""#### **Informasi Kolom pada Dataset :**
> Dataset ini memiliki 2 file csv yaitu Anime.csv dan Rating.csv. Berikut penjelasannya :

**Anime.csv** 

Kolom | Keterangan
--- | ---
anime_id | ID unik yang mengidentifikasi anime
name | Judul anime
genre | Daftar genre pada anime yang dipisahkan dengan tanda koma
type | movie, TV, OVA, dll
episodes | Jumlah episode (1 jika movie)
rating | Rating rata-rata dari 10 untuk anime
member | Jumlah anggota komunitas yang ada di anime

**Rating.csv** 

Kolom | Keterangan
--- | ---
user_id | ID pengguna yang dibuat secara acak tidak dapat diidentifikasi
anime_id | Anime yang telah diberi peringkat oleh pengguna
rating | Rating dari 10 yang telah diberikan pengguna (-1 jika pengguna menontonnya tetapi tidak memberikan peringkat)

> *Catatan : kolom rating pada file Anime.csv adalah rating yang berasal dari ulasan pada situs web, dan kolom rating pada file Rating.csv adalah rating yang berasal dari ID pengguna.*

##### Variabel Anime
"""

df_anime.head(11)

# Memuat informasi dataframe
df_anime.info()

"""Berdasarkan output di atas, dapat diketahui bahwa file Anime.csv memiliki 12294 entri."""

# Memuat deskripsi setiap kolom dataframe
df_anime.describe()

"""Dari output di atas, dapat disimpulkan bahwa nilai maksimum rating adalah 10 dan nilai minimumnya adalah 1.027 (1). Artinya, skala rating berkisar antara 1 hingga 10."""

# Melihat jumlah data kosong pada setiap kolom
df_anime.isnull().sum()

#Drop data yang kosong pada setiap kolom
df_anime = df_anime.dropna(axis=0)
df_anime.info()

# mengubah tipe data pada kolom rating menjadi integer untuk menyamakan dengan kolom rating pada file Rating.csv
df_anime['rating'] = df_anime['rating'].astype(int)
df_anime.info()

print('Total jumlah ID Anime :', len(df_anime['anime_id'].unique()))
print('Total jumlah Genre :', len(df_anime['genre'].unique()))
print('Total jumlah Judul Anime :', len(df_anime['name'].unique()))
print('Total jumlah Rating :', len(df_anime['rating'].unique()))

"""##### Variabel Rating"""

df_rating.head(11)

# Memuat informasi dataframe
df_rating.info()

"""Berdasarkan output di atas, dapat diketahui bahwa file Rating.csv memiliki 7813737 entri."""

# Memuat deskripsi setiap kolom dataframe
df_rating.describe()

print('Total jumlah user :', len(df_rating['user_id'].unique()))
print('Total jumlah judul anime :', len(df_rating['anime_id'].unique()))
print('Total jumlah rating :', len(df_rating['rating'].unique()))

"""## Data Preprocessing"""

# Merge dataframe df_rating dan df_anime
anime_new = pd.merge(df_anime, df_rating, on='anime_id', suffixes=['', 'user'])
anime_new

# Ganti nama kolom agar dapat lebih mudah dipahami
anime_new = anime_new.rename(columns={'name' : 'title', 'ratinguser':'rating_user'})
anime_new.head(11)

"""## Data Preparation

Pada data preparation, saya akan memeriksa rating dengan nilai -1 yang kemudian akan dianggap sebagai outlier dan akan diubah menjadi NaN lalu akan saya bersihkan.
"""

anime_fix=anime_new.copy()
anime_fix["rating_user"].replace({-1: np.nan}, inplace=True)
anime_fix

# Melihat jumlah data kosong
anime_fix.isnull().sum()

# Drop data yang kosong
anime_fix = anime_fix.dropna(axis = 0, how ='any') 
anime_fix.isnull().sum()

# Membuat variabel fix_anime yang berisi dataframe anime_fix kemudian mengurutkan berdasarkan anime_id
fix_anime = anime_fix
fix_anime.sort_values('anime_id')

"""Saya hanya akan menggunakan data unik untuk dimasukkan ke dalam proses pemodelan. Oleh karena itu, data yang duplikat akan saya hapus dengan fungsi drop_duplicates(). Dalam hal ini, saya membuang data duplikat pada kolom ‘anime_id’."""

# Membuang data duplikat pada variabel fix_anime
fix_anime = fix_anime.drop_duplicates('anime_id')
fix_anime

"""Setelah menghapus data duplikat, jumlah data saat ini adalah 9892 baris dan 9 kolom. Selanjutnya saya akan membuat visualisasi data agar dapat dipahami dengan lebih mudah."""

# Visualisasi untuk kolom type
plt.figure(figsize=(10,6))
sns.countplot(fix_anime.type)

"""Dari hasil visualisasi di atas, dapat disimpulkan bahwa anime ditayangkan lebih banyak di TV dan di OVA."""

# Visualisasi untuk rating berdasarkan ulasan situs web
with sns.axes_style('dark'):
    g = sns.catplot('rating', data=fix_anime, aspect=2.0, kind='count')
    g.set_ylabels('Jumlah total rating berdasarkan ulasan situs web ')

"""Dari hasil visualisasi di atas, dapat disimpulkan bahwa rata-rata rating melalui ulasan website adalah kebanyakan diantara 6 dan 7."""

# Visualisasi untuk rating berdasarkan ID pengguna
with sns.axes_style('dark'):
    g = sns.catplot('rating_user', data=fix_anime, aspect=2.0, kind='count')
    g.set_ylabels('Jumlah total rating berdasarkan id pengguna ')

"""Berdasarkan visualisasi di atas, dapat disimpulkan bahwa rating berdasarkan ID Pengguna paling banyak adalah 7 dan diikuti dengan 6 dan 8.

Selanjutnya, saya akan menampilkan genre mana saja yang paling sering muncul dengan menggunakan word clouds plot.
"""

from wordcloud import WordCloud

def wordCloud(words):
    wordCloud = WordCloud(width=800, height=500, background_color='black', random_state=21, max_font_size=120).generate(words)
    
    plt.figure(figsize=(10, 7))
    plt.imshow(wordCloud, interpolation='bilinear')
    plt.axis('off')
 
all_words = ' '.join([text for text in fix_anime['genre']])
wordCloud(all_words)

"""Berdasarkan plot yang telah dibuat, dapat dilihat bahwa genre yang paling sering muncul adalah Sci Fi, Comedy, Adventure, Slice Life, dan Action.

Selanjutnya, saya akan melakukan konversi data series menjadi list. Dalam hal ini, saya menggunakan fungsi tolist() dari library numpy. Lalu saya akan melakukan persiapan data untuk menyandikan (encode) kolom ‘user_id’ dan ‘anime_id’ ke dalam indeks integer.
"""

# Mengubah user_id menjadi list
user_id = fix_anime['user_id'].unique().tolist()
print('list user_id: ', user_id)
 
# Encoding user_id
user_to_user_encoded = {x: i for i, x in enumerate(user_id)}
print('encoded user_id : ', user_to_user_encoded)
 
# Encoding angka ke ke user_id
user_encoded_to_user = {i: x for i, x in enumerate(user_id)}

print('encoded angka ke user_id: ', user_encoded_to_user)

# Mengubah anime_id menjadi list
anime_id = fix_anime['anime_id'].unique().tolist()
 
# Encoding anime_id
anime_to_anime_encoded = {x: i for i, x in enumerate(anime_id)}
 
# Encoding angka ke anime_id
anime_encoded_to_anime = {i: x for i, x in enumerate(anime_id)}

print('encoded angka ke anime_id: ', anime_encoded_to_anime)

"""# Model Development

Untuk tahap modeling, saya akan menggunakan Neural Network dan Cosine Simirality untuk sistem rekomendasi berbasis Collaborative Filtering dan Content-Based Filtering.

Model Deep Learning akan saya gunakan untuk Sistem Rekomendasi berbasis Collaborative Filtering yang mana model ini akan menghasilkan rekomendasi untuk satu pengguna.

Cosine Similarity akan saya gunakan untuk Sistem Rekomendasi berbasis Content-Based Filtering yang akan menghitung kemiripan antara satu film dengan lainnya berdasarkan fitur yang terdapat pada satu film.

##### Content Based Filtering
"""

from sklearn.feature_extraction.text import TfidfVectorizer

# Inisialisasi TfidfVectorizer
tf = TfidfVectorizer()
 
# Melakukan perhitungan idf pada data genre
tf.fit(fix_anime['genre']) 
 
# Mapping array dari fitur index integer ke fitur nama
tf.get_feature_names()

# Melakukan fit lalu ditransformasikan ke bentuk matrix
tfidf_matrix = tf.fit_transform(fix_anime['genre']) 
 
# Melihat ukuran matrix tfidf
tfidf_matrix.shape

from sklearn.metrics.pairwise import cosine_similarity

# Menghitung cosine similarity pada matrix tf-idf
cosine_sim = cosine_similarity(tfidf_matrix) 
cosine_sim

# Membuat dataframe dari variabel cosine_sim dengan baris dan kolom berupa judul anime
cosine_sim_df = pd.DataFrame(cosine_sim, index=fix_anime['title'],
                             columns=fix_anime['title'])
print('Shape:', cosine_sim_df.shape)
 
# Melihat similarity matrix tiap anime
cosine_sim_df.sample(10, axis=1).sample(10, axis=0)

"""
    Rekomendasi Anime berdasarkan kemiripan dataframe
 
    Parameter:
    ---
    title : tipe data string (str)
                Judul Anime (index kemiripan dataframe)
    similarity_data : tipe data pd.DataFrame (object)
                      Kesamaan dataframe, simetrik, dengan genre sebagai 
                      indeks dan kolom
    items : tipe data pd.DataFrame (object)
            Mengandung kedua nama dan fitur lainnya yang digunakan untuk mendefinisikan kemiripan
    k : tipe data integer (int)
        Banyaknya jumlah rekomendasi yang diberikan
    ---
 
 
    Pada index ini, kita mengambil k dengan nilai similarity terbesar 
    pada index matrix yang diberikan (i).
"""

def anime_recommendations(anime_title, similarity_data=cosine_sim_df, 
                         items=fix_anime[['title','genre','type','rating']], k=10):
 
 
    # Mengambil data dengan menggunakan argpartition untuk melakukan partisi secara tidak langsung sepanjang sumbu yang diberikan    
    # Dataframe diubah menjadi numpy
    # Range(start, stop, step)

    index = similarity_data.loc[:, anime_title].to_numpy().argpartition(
        range(-1, -k, -1)
    )

    # Mengambil data dengan similarity terbesar dari index yang ada
    closest = similarity_data.columns[index[-1:-(k+2):-1]]

    # Drop anime_title agar nama anime yang dicari tidak muncul dalam daftar rekomendasi
    closest = closest.drop(anime_title, errors='ignore')

    return pd.DataFrame(closest).merge(items).head(k)

"""Selanjutnya, saya akan menerapkan kode di atas untuk menemukan rekomendasi Anime yang mirip dengan Boruto: Naruto the Movie."""

fix_anime[fix_anime.title.eq('Boruto: Naruto the Movie')]

"""Berdasarkan output di atas, dapat dilihat bahwa Anime dengan judul Boruto: Naruto the Movie memiliki genre Action, Comedy, Martial Arts, Shounen, dan Super Power. Rekomendasi yang diharapkan adalah Anime dengan genre yang sama."""

# Mendapatkan rekomendasi anime yang mirip dengan Boruto: Naruto the Movie
anime_recommendations('Boruto: Naruto the Movie')

"""Model berhasil memberikan rekomendasi 10 judul Anime dengan Genre yang sama seperti yang diharapkan, yaitu Action, Comedy, Martial Arts, Shounen, dan Super Power.

##### Collaborative Filtering
"""

fix_anime.head()

# Drop kolom yang tidak digunakan
df_anime = fix_anime.drop(columns=['rating'])
df_anime

# Mapping user_id ke dataframe user
df_anime['user'] = df_anime['user_id'].map(user_to_user_encoded)
 
# Mapping anime_id ke dataframe anime
df_anime['anime'] = df_anime['anime_id'].map(anime_to_anime_encoded)

# Cek jumlah user, anime, dan mengubah nilai rating menjadi float.

# Mendapatkan jumlah user
num_user = len(user_to_user_encoded)
 
# Mendapatkan jumlah anime
num_anime = len(anime_encoded_to_anime)
 
# Mengubah rating menjadi nilai float
df_anime['rating_user'] = df_anime['rating_user'].values.astype(np.float32)
 
# Nilai minimum rating
min_rating = min(df_anime['rating_user'])
 
# Nilai maksimal rating
max_rating = max(df_anime['rating_user'])
 
print('Jumlah User: {}, Jumlah anime: {}, Min Rating: {}, Max Rating: {}'.format(
    num_user, num_anime, min_rating, max_rating
))

"""Saya akan membagi data untuk training dan validasi dengan komposisi 80:20. Namun sebelumnya, saya akan mengacak datanya terlebih dahulu agar distribusinya menjadi random."""

# Mengacak dataset
df_anime = df_anime.sample(frac=1, random_state=42)
df_anime

"""Saya akan memetakan (mapping) data user dan anime menjadi satu value terlebih dahulu. Kemudian membuat rating dalam skala 0 sampai 1 agar mudah dalam melakukan proses training."""

# Inisialisasi variable x untuk mencocokkan data user dan anime menjadi satu value
x = df_anime[['user', 'anime']].values
 
# Inisialisasi variable y untuk membuat rating dari hasil 
y = df_anime['rating_user'].apply(lambda x: (x - min_rating) / (max_rating - min_rating)).values
 
# Split data dengan komposisi 80% data train dan 20% data validasi
train_indices = int(0.8 * df_anime.shape[0])
x_train, x_val, y_train, y_val = (
    x[:train_indices],
    x[train_indices:],
    y[:train_indices],
    y[train_indices:]
)
 
print(x, y)

"""Saya akan melakukan proses embedding terhadap data user dan anime. Lalu melakukan operasi perkalian dot product antara embedding user dan anime. Selain itu, saya juga menambahkan bias untuk setiap user dan anime. Skor kecocokan ditetapkan dalam skala [0,1] dengan fungsi aktivasi sigmoid."""

# Import library yang akan kita gunakan

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class RecommenderNet(tf.keras.Model):
 
  # Insialisasi fungsi
  def __init__(self, num_users, num_anime, embedding_size, **kwargs):
    super(RecommenderNet, self).__init__(**kwargs)
    self.num_users = num_users
    self.num_anime = num_anime
    self.embedding_size = embedding_size
    self.user_embedding = layers.Embedding( # layer embedding user
        num_users,
        embedding_size,
        embeddings_initializer = 'he_normal',
        embeddings_regularizer = keras.regularizers.l2(1e-6)
    )
    self.user_bias = layers.Embedding(num_users, 1) # layer embedding user bias
    self.anime_embedding = layers.Embedding( # layer embeddings anime
        num_anime,
        embedding_size,
        embeddings_initializer = 'he_normal',
        embeddings_regularizer = keras.regularizers.l2(1e-6)
    )
    self.anime_bias = layers.Embedding(num_anime, 1) # layer embedding anime bias
 
  def call(self, inputs):
    user_vector = self.user_embedding(inputs[:,0]) # memanggil layer embedding 1
    user_bias = self.user_bias(inputs[:, 0]) # memanggil layer embedding 2
    anime_vector = self.anime_embedding(inputs[:, 1]) # memanggil layer embedding 3
    anime_bias = self.anime_bias(inputs[:, 1]) # memanggil layer embedding 4
 
    dot_user_anime = tf.tensordot(user_vector, anime_vector, 2) 
 
    x = dot_user_anime + user_bias + anime_bias
    
    return tf.nn.sigmoid(x) # activation sigmoid

# inisialisasi model RecommenderNet
model = RecommenderNet(num_user, num_anime, 50)

# compile model
model.compile(
    loss = tf.keras.losses.BinaryCrossentropy(),
    optimizer = keras.optimizers.Adam(learning_rate=0.001),
    metrics=[[tf.keras.metrics.MeanAbsoluteError(), tf.keras.metrics.RootMeanSquaredError()]]
)

"""Model ini menggunakan Binary Crossentropy untuk menghitung loss function, Adam (Adaptive Moment Estimation) sebagai optimizer, Mean Absolute Error dan Root Mean Squared Error (RMSE) sebagai metrics evaluation."""

# Menggunakan callback agar pengujian berhenti jika akurasi mencapai target
from keras.callbacks import  EarlyStopping

callbacks = EarlyStopping(
    min_delta=0.0001,
    patience=7,
    restore_best_weights=True,
)

history = model.fit(
    x = x_train,
    y = y_train,
    batch_size = 8,
    epochs = 100,
    validation_data = (x_val, y_val),
    callbacks=[callbacks]
)

"""Untuk mendapatkan rekomendasi anime, saya akan mengambil sampel user secara acak dan mendefinisikan variabel anime_not_watched yang merupakan daftar anime yang belum pernah ditonton oleh pengguna."""

# Mengambil sample user
user_id = df_anime.user_id.sample(1).iloc[0]
anime_watched_by_user = df_anime[df_anime.user_id == user_id]
 
# Operator bitwise
anime_not_watched = df_anime[~df_anime['anime_id'].isin(anime_watched_by_user.anime_id.values)]['anime_id'] 
anime_not_watched = list(
    set(anime_not_watched)
    .intersection(set(anime_to_anime_encoded.keys()))
)

anime_not_watched = [[anime_to_anime_encoded.get(x)] for x in anime_not_watched]
user_encoder = user_to_user_encoded.get(user_id)
user_anime_array = np.hstack(
    ([[user_encoder]] * len(anime_not_watched), anime_not_watched)
)

ratings = model.predict(user_anime_array).flatten()

# top rating
top_ratings_indices = ratings.argsort()[-10:][::-1]

# rekomendasi anime
recommended_anime_ids = [
    anime_encoded_to_anime.get(anime_not_watched[x][0]) for x in top_ratings_indices
]

print('Menampilkan rekomendasi untuk user: {}'.format(user_id))
print('=' * 9)
print('anime dengan peringkat tinggi dari user')
print('-' * 8)

# mencari rekomendasi anime berdasarkan rating yang diberikan user
top_anime_user = (
    anime_watched_by_user.sort_values(
        by = 'rating_user',
        ascending=False
    )
    .head(5)
    .anime_id.values
)
 
df_anime_rows = df_anime[df_anime['anime_id'].isin(top_anime_user)]
for row in df_anime_rows.itertuples():
    print(row.title, ':', row.genre)
 
print('-' * 8)
print('10 rekomendasi anime teratas')
print('-' * 8)

# rekomendasi anime
anime_top10 = df_anime[df_anime['anime_id'].isin(recommended_anime_ids)]

# fungsi perulangan untuk menampilkan rekomendasi anime dan genre sebanyak 10 buah
for row in anime_top10.itertuples():
    print(row.title, ':', row.genre)

"""Kita telah mendapatkan 10 rekomendasi Anime untuk user dengan id 4643.

# Model Evaluation

Mean Absolute Error (MAE) mengukur besarnya rata-rata kesalahan dalam serangkaian prediksi yang sudah di latih kepada data yang akan dites, tanpa mempertimbangkan arahnya. Semakin rendah nilai MAE (mean absolute error) maka semakin baik dan akurat model yang dibuat.

Berikut rumusnya :

![image](https://user-images.githubusercontent.com/87566521/139152819-30500f63-40a3-40ed-86fd-a62e517adbb4.png)
"""

plt.plot(history.history['mean_absolute_error'])
plt.plot(history.history['val_mean_absolute_error'])
plt.title('model_metrics')
plt.ylabel('mean_absolute_error')
plt.xlabel('epoch')
plt.legend(['mean_absolute_error', 'val_mean_absolute_error'], loc='upper left')
plt.show()

"""Root mean squared error (RMSE) adalah aturan penilaian kuadrat yang juga mengukur besarnya rata-rata kesalahan. Sama seperti MAE, semakin rendahnya nilai root mean square error juga menandakan semakin baik model tersebut dalam melakukan prediksi.

Berikut rumusnya :

![image](https://user-images.githubusercontent.com/87566521/139154262-7eca086f-2007-41e1-9737-5f9fe68a8f49.png)

"""

plt.plot(history.history['root_mean_squared_error'])
plt.plot(history.history['val_root_mean_squared_error'])
plt.title('model_metrics')
plt.ylabel('root_mean_squared_error')
plt.xlabel('epoch')
plt.legend(['root_mean_squared_error', 'val_root_mean_squared_error'], loc='upper left')
plt.show()

"""Berdasarkan plotting proses training di atas, dapat dilihat bahwa proses training model cukup smooth dan model konvergen pada epochs sekitar 90. Dari hasil model ini, Mean Absolute Error yang didapat adalah 0.0077 pada training dan 0.1375 pada test. Untuk Root Mean Squared Error, diperoleh nilai error akhir sebesar 0.0126 pada tranining dan 0.1823 pada test. Hal ini menunjukan bahwa model ini memiliki error dibawah 20% jika menggunakan MAE dan dibawah 20% jika menggunakan RMSE.

# Penutup :

Model Sistem rekomendasi Anime telah selesai dibuat dan model ini dapat digunakan untuk untuk merekomendasikan data yang sebenarnya. Berdasarkan model tersebut, dapat diketahui bahwa sistem rekomendasi berbasis Collaborative Filtering dan Content-Based Filtering dapat merekomendasikan anime kepada pengguna seperti yang diharapkan. Namun, beberapa pengembangan lain masih dapat dilakukan untuk membuat model yang memiliki akurasi lebih tinggi.
"""
