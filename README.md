# Rainfall Prediction Using Ensemble SVR

## 📌 Overview
Proyek ini merupakan bagian dari skripsi berjudul **"Penerapan Metode Ensemble untuk Multivariate Time Series Berbasis Kernel pada Peramalan Curah Hujan (Studi Kasus: Stasiun Meteorologi Perak I)"**. Penelitian ini menggunakan metode **Ensemble Support Vector Regression (SVR)** untuk melakukan peramalan curah hujan berdasarkan data cuaca multivariate yang telah dinormalisasi.

## 🔍 Metode Penelitian
- **Data**: Data cuaca dari Stasiun Meteorologi Perak I dengan fitur **Temperatur, Kelembapan, Kecepatan Angin, dan Curah Hujan**.
- **Exploratory Data Analysis**: mengeksplorasi dan menganalisis kumpulan data dengan menggunakan teknik visualisasi dan statistik seperti:
  - **Statistik Deskriptif**: Melakukan perhitungan ringkasan statistik seperti mean, median, dan deviasi standar untuk menganalisis nilai-nilai pusat dan sebaran data.
  - **Visualisasi Data**: Menggunakan berbagai jenis grafik seperti histogram, diagram pencar (scatter plot), dan box plot untuk memvisualisasikan distribusi data dan hubungan antar variabel.
- **Preprocessing**:
  - **Interpolasi** untuk mengisi missing value.
  - **Normalisasi** untuk menyamakan skala fitur.
  - **Deteksi dan Penanganan Outlier** menggunakan Z-score.
  - **Sliding Window Penentuan Input dan Output** menggunakan ACF dan PACF.
- **Pembagian Data**: Pembagian ini dilakukan dengan rasio 80:20, di mana 80% dari data digunakan sebagai data latih dan 20% sisanya sebagai data uji.
- **Modeling**:
  - **Bootstrap Sampling** untuk membentuk beberapa subset data training sebanyak 5, 10, dan 20 estimator.
  - **Support Vector Regression (SVR) dengan kernel Linear, RBF dan Polynomial**.
  - **Optimasi Hyperparameter menggunakan GridSearch**.
- **Evaluasi**: Menggunakan metrik **MAE dan RMSE**.

## 📂 Struktur Repository
```
├── data/                      # Dataset curah hujan setelah preprocessing
├── models/                    # Model yang telah dilatih
├── notebooks/                 # Notebook Jupyter untuk eksplorasi dan eksperimen
├── scripts/                   # Script untuk preprocessing, training, dan evaluasi
├── results/                   # Hasil prediksi dan evaluasi model
├── README.md                  # Dokumentasi proyek
```

## 🚀 Instalasi dan Penggunaan
### 1. Clone Repository
```bash
git clone https://github.com/rizkyjisantt-dev/rainfall-prediction-using-ensemblesvr.git
cd rainfall-prediction-using-ensemblesvr
```
### 2. Install Dependencies
Gunakan Python 3.8+ dan install package yang diperlukan:
```bash
pip install -r requirements.txt
```

### 3. Jalankan Eksperimen
Gunakan Jupyter Notebook atau jalankan script Python untuk menjalankan eksperimen:
```bash
jupyter notebook
```
Atau jalankan model langsung:
```bash
python scripts/train_model.py
```

## 📊 Hasil dan Analisis
- Model terbaik dipilih berdasarkan nilai **MSE, RMSE, dan R²**.
- Diperbandingkan antara **SVR biasa dan Ensemble SVR** untuk melihat peningkatan performa.
- Analisis korelasi dilakukan untuk memahami hubungan antar fitur dengan curah hujan.

## 🏆 Kontribusi dan Lisensi
Proyek ini bersifat open-source. Jika ingin berkontribusi atau berdiskusi, silakan lakukan **pull request** atau hubungi melalui **[GitHub Issues](https://github.com/rizkyjisantt-dev/rainfall-prediction-using-ensemblesvr/issues)**.

---
✉️ **Dikembangkan oleh [Rizky Jisantt](https://github.com/rizkyjisantt-dev/)**

