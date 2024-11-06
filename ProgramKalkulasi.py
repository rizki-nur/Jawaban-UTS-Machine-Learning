import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Load dataset
file_path = 'C:\Project\Jawaban Soal UTS Machine Learning\mobileprice_modified.csv'
df = pd.read_csv(file_path)

# Nomor 1: identifikasi atribut dan jenisnya
print("Identifikasi Atribut dan Jenisnya:\n")
for column in df.columns:
    unique_values = df[column].nunique()
    dtype = df[column].dtype
    print(f"Atribut: {column}, Jenis: {'Kategorik' if unique_values < 10 else 'Numerik'}, Tipe Data: {dtype}")
    if unique_values < 10:  # jika kategorik, tampilkan nilai unik
        print("Nilai Unik:", df[column].unique())
    print("\n")

# Nomor 2: Praproses Data
# 2a: Pisahkan atribut prediktor dan label
X = df.drop(columns='price_range')
y = df['price_range']

# 2b: Tangani missing values menggunakan simpleimputer dengan strategi rata-rata
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# simpan data sebelum dan sesudah imputasi untuk statistik deskriptif
X_before_impute = X.describe()
X_after_impute = pd.DataFrame(X_imputed, columns=X.columns).describe()

# 2c: terapkan standardscaler untuk standarisasi data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# simpan statistik deskriptif setelah standarisasi
X_after_scaling = pd.DataFrame(X_scaled, columns=X.columns).describe()

# tampilkan statistik deskriptif
print("Statistik Deskriptif Sebelum Imputasi:\n", X_before_impute)
print("\nStatistik Deskriptif Setelah Imputasi:\n", X_after_impute)
print("\nStatistik Deskriptif Setelah Standarisasi:\n", X_after_scaling)

# Nomor 3: Melakukan Klasifikasi
# 3b: Membagi dataset menjadi data training (85%) dan data testing (15%) dengan metode holdout
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.15, random_state=42)

# 3a: membangun model klasifikasi menggunakan Decision Tree
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# melakukan prediksi pada data testing
y_pred = model.predict(X_test)

# 3c: menghitung confusion matrix dan akurasi
conf_matrix = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

# tampilkan hasil evaluasi klasifikasi
print("\nConfusion Matrix:\n", conf_matrix)
print("Accuracy:", accuracy)

# Nomor 4: Melakukan Clustering
# 4a: membangun model clustering menggunakan K-Means dengan 4 cluster
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(X_scaled)

# 4b: menghitung silhouette score
silhouette_avg = silhouette_score(X_scaled, kmeans.labels_)

# tampilkan hasil silhouette score
print("\nSilhouette Score:", silhouette_avg)
