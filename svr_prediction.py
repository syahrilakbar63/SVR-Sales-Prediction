import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import joblib

# Memuat dataset dari file CSV lokal
file_path = "penjualan_barang.csv"
df = pd.read_csv(file_path)

# Menampilkan semua baris dari dataset
print("Data from the dataset:")
pd.set_option('display.max_rows', None)  # Set opsi untuk menampilkan semua baris
print(df)
pd.reset_option('display.max_rows')  # Reset opsi setelah menampilkan dataset


# Normalisasi tanggal dengan ordinal encoding
df['tanggal'] = pd.to_datetime(df['tanggal'])
df['ordinal_tanggal'] = df['tanggal'].apply(lambda x: x.toordinal())
X = np.array(df[['ordinal_tanggal', 'kuantum']])
X_scaled = StandardScaler().fit_transform(X)

# Mengambil kolom nominal sebagai target
y = np.array(df['nominal'])

# Membagi data menjadi set pelatihan dan pengujian (80:20)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Membuat model SVR
model = SVR(kernel='rbf', C=10, gamma=0.1, epsilon=0.1)

# Melatih model
model.fit(X_train, y_train)

# Melakukan prediksi
y_pred = model.predict(X_test)

# Mengevaluasi model
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print(f"\nSkor R2: {r2}")
print(f"Mean Squared Error: {mse}")
print(f"Mean Absolute Error: {mae}")

# Simpan model ke file dengan nama yang jelas
joblib.dump(model, 'svr_sales_prediction_model.joblib')

# Visualisasi hasil prediksi
plt.scatter(X_test[:, 0], y_test, color='black', label='Actual')
plt.scatter(X_test[:, 0], y_pred, color='red', label='Predicted')
plt.title('SVR Prediction of Sales')
plt.xlabel('Ordinal Tanggal')  # Sesuaikan label sumbu x
plt.ylabel('Nominal')
plt.legend()
plt.show()