# SVR-Sales-Prediction

Repository ini berisi kode untuk memprediksi penjualan menggunakan Support Vector Regression (SVR). Kode ini mengambil data penjualan dari file CSV, melakukan preprocessing data, melatih model SVR, dan mengevaluasi performa model. Hasil prediksi juga divisualisasikan dalam bentuk plot.

## Deskripsi

Program ini dibuat dengan bahasa pemrograman Python dan menggunakan library seperti pandas, numpy, sklearn, dan matplotlib. Program ini membaca data penjualan dari file CSV, melakukan normalisasi tanggal dengan ordinal encoding, dan membagi data menjadi set pelatihan dan pengujian.

Setelah itu, program ini melatih model SVR dan melakukan prediksi pada set pengujian. Performa model dievaluasi dengan skor R2, Mean Squared Error, dan Mean Absolute Error. Hasil prediksi divisualisasikan dalam bentuk plot.

## Cara Menjalankan

1. Pastikan Anda telah menginstal Python dan semua library yang diperlukan.
2. Download file `penjualan_barang.csv` dan letakkan di direktori yang sama dengan script Python.
3. Jalankan script Python.

## Hasil

Setelah menjalankan script, model SVR akan disimpan dalam file `svr_sales_prediction_model.joblib`. Anda juga akan melihat plot yang menunjukkan hasil prediksi model.
