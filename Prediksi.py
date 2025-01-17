import streamlit as st
import numpy as np  # Untuk operasi matematika dan manipulasi array
import pandas as pd  # Untuk manipulasi dan analisis data tabular (dataframe)
import matplotlib.pyplot as plt  # Untuk visualisasi data dalam berbagai grafik
from matplotlib import pyplot  # Untuk fungsi tambahan dalam visualisasi
import matplotlib.dates as mdates  # Untuk manipulasi dan format tanggal dalam grafik
import seaborn as sns  # Untuk visualisasi statistik yang lebih menarik dan informatif
from scipy import stats  # Untuk komputasi ilmiah dan analisis statistik
from scipy.stats import zscore  # Untuk menghitung nilai z-score (standarisasi data)
from sklearn.preprocessing import MinMaxScaler  # Untuk normalisasi data ke rentang [0, 1]
from sklearn.model_selection import train_test_split, GridSearchCV  # Untuk pemisahan data dan pencarian hyperparameter
from sklearn.utils import resample  # Untuk teknik sampling ulang seperti bootstrap
from sklearn.ensemble import BaggingRegressor  # Model ensemble yang menggunakan teknik bootstrap untuk regresi
from sklearn.svm import SVR  # Support Vector Regression untuk model regresi
import joblib  # Untuk menyimpan dan memuat model atau objek Python
import pickle

# Set Streamlit page configuration
st.set_page_config(page_title="Prediksi Curah Hujan", page_icon=":🌦️:", layout="wide")

# CSS untuk menambahkan gambar latar belakang
page_bg_img = '''
<style>
.stApp {
    background-image: url("https://storage.needpix.com/rsynced_images/lluvia-25761279462529L5Co.jpg");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: fixed;
</style>
'''
# Masukkan CSS ke dalam aplikasi
st.markdown(page_bg_img, unsafe_allow_html=True)

# Define navigation menu
menu = ["Homepage", "Dataset", "Preprocessing", "Pembagian Data", "Modelling", "Prediksi"]
choice = st.sidebar.selectbox("Pilih Halaman", menu)

# Load dataset from GitHub
url = "https://raw.githubusercontent.com/rizkyjisantt/skripsi/refs/heads/main/laporan_iklim_2018-2024_terbaru.csv"

def load_data():
    # Langkah 2: Ambil data dari GitHub dan masukkan ke dalam DataFrame
    data = pd.read_csv(url)
    # Langkah 3: Ubah nama format tanggal
    data['Tanggal'] = pd.to_datetime(data['Tanggal'], format='%d-%m-%Y')
    data.set_index(['Tanggal'],inplace=True)
    # Langkah 4: Ubah nama kolom sesuai dengan instruksi
    data.rename(columns={
        "Tavg": "temperatur",
        "RH_avg": "kelembapan",
        "ff_avg": "kecepatan_angin",
        "RR": "curah_hujan"
    }, inplace=True)
    return data

if choice == "Homepage":
    st.title("Prediksi Curah Hujan menggunakan Ensemble Support Vector Regression ")
    st.sidebar.success("Pilih halaman berikut")
    text = """<p>Selamat Datang Sistem Prediksi Curah Hujan menggunakan Ensemble Support Vector Regression 👋. Data yang digunakan merupakan data rekapan harian milik Badan Meteorologi Klimatologi dan Geofisika (BMKG) yang dapat diakses pada laman <a href='https://dataonline.bmkg.go.id/dataonline-home'>dataonline.bmkg.go.id/dataonline-home</a>.</p>"""
    st.markdown(text, unsafe_allow_html=True)
    image_url = "https://raw.githubusercontent.com/rizkyjisantt/skripsi/main/lamandataonlinebmkg.png" # Gunakan URL gambar langsung
    st.image(image_url, caption='Laman dataonline.bmkg.go.id/dataonline-home', width=None)


elif choice == "Dataset":
    st.title("Dataset")
    # Menambahkan penjelasan tentang dataset
    st.write("""
    **Penjelasan Dataset:**
    Data yang digunakan berasal dari hasil observasi dari Badan Meteorologi, Klimatologi, dan Geofisika (BMKG) Stasiun Meteorologi Perak I. 
    Data ini diambil dari website [http://dataonline.bmkg.go.id/dataonline-home](http://dataonline.bmkg.go.id/dataonline-home). 
    Data harian selama lima tahun terakhir, mulai Januari 2018 hingga Desember 2024, digunakan untuk penelitian ini. 
    Total jumlah data hingga saat ini berjumlah 2.557 data. 
    Parameter yang digunakan dalam penelitian ini adalah:
    - **Curah Hujan** dengan satuan millimeter (mm)
    - **Titik Kelembapan Rata-rata** dengan satuan persen (%)
    - **Temperatur Rata-rata** dengan satuan derajat Celcius (°C)
    - **Kecepatan Angin Rata-rata** dengan satuan meter per sekon (m/s)
    """)
    # Menampilkan data frame
    data = load_data()
    st.write("### Dataframe")
    st.write("""
    Data yang ditampilkan di bawah ini adalah potongan pertama (5 baris) dari dataset yang digunakan. 
    Data ini mencakup parameter curah hujan, kelembapan, temperatur, dan kecepatan angin pada stasiun meteorologi Perak I. 
    Anda dapat melihat nilai-nilai untuk setiap parameter per tanggal dalam dataset ini.
    """)
    st.dataframe(data.head())
    # Menampilkan statistik deskriptif
    st.write("### Statistik Deskriptif")
    st.write("""
    Statistik deskriptif ini memberikan gambaran umum tentang distribusi data untuk setiap parameter. 
    Rata-rata, deviasi standar, nilai minimum dan maksimum, serta kuartil pertama dan ketiga akan memberikan wawasan tentang sebaran data yang ada. 
    Informasi ini penting untuk memahami karakteristik dan variasi dalam dataset sebelum melakukan analisis lebih lanjut.
    """)
    st.write(data.describe())
    # Plot Grafik Curah Hujan
    st.write("### Grafik Curah Hujan 2018-2024")
    st.write("""
    Grafik ini menunjukkan perubahan curah hujan selama periode Januari 2018 hingga Desember 2024. 
    Visualisasi ini memberikan gambaran umum tentang tren curah hujan selama lima tahun terakhir, 
    yang dapat digunakan untuk analisis lebih lanjut, seperti identifikasi pola musiman atau ekstrem.
    """)
    plt.figure(figsize=(14, 6))
    plt.plot(data['curah_hujan'], label='Curah Hujan', color='green')
    plt.title('Grafik Curah Hujan 2018-2024')
    plt.xlabel('Tahun')
    plt.ylabel('Curah Hujan (mm)')
    plt.gca().xaxis.set_major_locator(mdates.YearLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid()
    st.pyplot(plt)


elif choice == "Preprocessing":
    st.title("Preprocessing Data")
    data = load_data()
    tab1, tab2, tab3 = st.tabs(["Imputasi Missing Value", "Identifikasi Outlier", "Normalisasi Min-Max"])
    
    with tab1:
        st.subheader("Imputasi Missing Value")
        st.write("""
        **Imputasi Missing Value** adalah proses pengisian nilai yang hilang (missing values) dalam dataset dengan nilai tertentu 
        agar tidak mengganggu analisis lebih lanjut. Di sini, imputasi dilakukan menggunakan metode **Interpolasi Linier**. 
        Interpolasi linier digunakan untuk mengisi nilai yang hilang dengan mengestimasi nilai berdasarkan nilai-nilai yang ada di sekitarnya.
        
        **Rumus Interpolasi Linier**:
        ![Rumus Interpolasi Linier](https://goshohib.wordpress.com/wp-content/uploads/2012/06/rumus-interpolasi-linier.png)
        Dalam rumus ini, nilai yang hilang diestimasi dengan menggunakan dua titik data yang ada di sekitar nilai yang hilang.
        """)
        st.write("Jumlah data yang kosong:")
        st.dataframe(data[['temperatur', 'kelembapan', 'kecepatan_angin', 'curah_hujan']].isnull().sum())
        # Imputasi dengan interpolasi
        data['curah_hujan'] = data['curah_hujan'].interpolate(method='linear', limit_direction='both').fillna(0)
        st.write("Missing values setelah interpolasi:")
        st.dataframe(data.isnull().sum())

    with tab2:
        st.subheader("Identifikasi Outlier")
        st.write("""
        **Outlier** adalah data yang sangat berbeda atau tidak sesuai dengan pola umum dari data lainnya. 
        Di sini, identifikasi outlier dilakukan menggunakan **Z-Score**. Z-score mengukur seberapa jauh suatu data dari rata-rata dalam satuan deviasi standar.
        Jika Z-score suatu data lebih besar dari threshold tertentu (misalnya 3), maka data tersebut dianggap sebagai outlier.

        **Rumus Z-Score**:
        ![Rumus Z-Score](https://miro.medium.com/v2/resize:fit:540/format:webp/1*vMeaZyAG-z2wKYyvTHmF9w.png)
        """)
        data_zscore = data[['temperatur', 'kelembapan', 'kecepatan_angin', 'curah_hujan']].apply(zscore)
        threshold = 3
        outliers = (data_zscore > threshold) | (data_zscore < -threshold)
        st.write("### Jumlah Outlier per Kolom")
        st.write("""
        Bagian ini menampilkan jumlah data yang terdeteksi sebagai outlier untuk setiap kolom (parameter). 
        Data yang memiliki Z-score lebih besar dari threshold yang ditentukan (misalnya 3) dianggap sebagai outlier. 
        Jumlah ini membantu kita mengetahui seberapa banyak data yang perlu dianalisis lebih lanjut.
        """)
        st.dataframe(outliers.sum())
        st.write("### Data yang Teridentifikasi sebagai Outlier")
        st.write("""
        Bagian ini menampilkan data yang teridentifikasi sebagai outlier berdasarkan Z-score. 
        Data ini memiliki nilai yang jauh berbeda dari nilai rata-rata dan dapat mempengaruhi model yang akan dibangun. 
        Biasanya, data ini akan dihilangkan atau diganti dengan nilai yang lebih representatif.
        """)
        st.dataframe(data[outliers.any(axis=1)])
        data_outlier = data.copy()
        for col in ['temperatur', 'kelembapan', 'kecepatan_angin', 'curah_hujan']:
            outlier_index = data_zscore[col][(data_zscore[col] > threshold) | (data_zscore[col] < -threshold)].index
            data_outlier.loc[outlier_index, col] = np.nan
        data_outlier.interpolate(method='linear', inplace=True)
        st.write("### Data Setelah Outlier Diganti dengan Interpolasi")
        st.write("""
        Bagian ini menampilkan data setelah nilai-nilai outlier diganti dengan interpolasi linier. 
        Interpolasi digunakan untuk menggantikan nilai yang hilang dengan perkiraan berdasarkan nilai-nilai yang ada di sekitarnya.
        Hal ini memungkinkan kita untuk mengatasi masalah outlier tanpa menghapus data, sehingga dataset tetap lengkap dan siap untuk analisis lebih lanjut.
        """)
        st.dataframe(data_outlier[outliers.any(axis=1)])

    with tab3:
        st.subheader("Normalisasi Min-Max")
        st.write("""
        **Normalisasi Min-Max** adalah teknik untuk mengubah nilai-nilai data sehingga berada dalam rentang tertentu, biasanya antara 0 dan 1. 
        Teknik ini berguna untuk memastikan bahwa semua fitur memiliki skala yang sama, terutama untuk algoritma yang sensitif terhadap skala data, seperti algoritma berbasis jarak (misalnya KNN atau SVM).
    
        **Rumus Normalisasi Min-Max**:
        ![Rumus Min-Max](https://miro.medium.com/v2/resize:fit:506/format:webp/0*S68TJiF_lZWV1THX.png)
        """)
        st.write("### Dataset sebelum normalisasi:")
        st.write("""
        Bagian ini menampilkan dataset sebelum dilakukan normalisasi. Nilai-nilai pada setiap fitur (seperti temperatur, kelembapan, kecepatan angin, dan curah hujan) berada dalam rentang yang berbeda-beda, 
        yang bisa mempengaruhi kinerja model.
        """)
        st.dataframe(data_outlier.head())
        # Menyimpan nilai minimum dan maksimum dari data asli
        data_min = data_outlier['curah_hujan'].min()
        data_max = data_outlier['curah_hujan'].max()
        st.session_state['data_min'] = data_min
        st.session_state['data_max'] = data_max
        # Cek apakah 'scaler' sudah ada di session_state, jika belum buat dan simpan
        if 'scaler' not in st.session_state:
            scaler = MinMaxScaler()
            st.session_state['scaler'] = scaler

        # Normalisasi menggunakan MinMaxScaler
        data_scaled = data_outlier.copy()
        features_to_normalize = ['temperatur', 'kelembapan', 'kecepatan_angin', 'curah_hujan']
        data_scaled[features_to_normalize] = st.session_state['scaler'].fit_transform(data_outlier[features_to_normalize])
        
        st.write("### Dataset setelah normalisasi:")
        st.write("""
        Bagian ini menampilkan dataset setelah dilakukan normalisasi menggunakan Min-Max Scaling. 
        Setelah normalisasi, nilai-nilai setiap fitur berada dalam rentang 0 hingga 1, yang akan membantu model dalam belajar secara lebih efisien.
        """)
        st.dataframe(data_scaled.head())
        
        # Menyimpan data yang telah dinormalisasi ke session_state
        st.session_state['data_scaled'] = data_scaled
        if st.button("Simpan Hasil Normalisasi ke Pickle"):
            with open("data_scaled.pkl", "wb") as file:
                pickle.dump(data_scaled, file)
            st.success("Hasil normalisasi berhasil disimpan ke file data_scaled.pkl.")



elif choice == "Pembagian Data":
    st.title("Pembagian Data")
    if 'data_scaled' not in st.session_state:
        st.error("Data belum dinormalisasi. Silakan lakukan normalisasi terlebih dahulu.")
        st.stop()
    data_scaled = st.session_state['data_scaled']
    st.success("Data hasil normalisasi berhasil dimuat!")
    df = data_scaled.copy()
    df['temperatur_lag'] = df['temperatur']
    df['kelembapan_lag'] = df['kelembapan']
    df['kecepatan_angin_lag'] = df['kecepatan_angin']
    st.write("""
    **Lag Features**:
    Pada tahap ini, fitur lag (data t-1, t-2, dst.) ditambahkan untuk membuat dataset menjadi bentuk supervised. 
    Fitur lag ini digunakan untuk memodelkan hubungan antara curah hujan saat ini dengan data sebelumnya. 
    Namun, tidak hanya `curah_hujan_lag1` sampai `curah_hujan_lag4` yang ditambahkan berdasarkan hasil sliding window, 
    tetapi juga variabel-variabel lain seperti kelembapan, temperatur, dan kecepatan angin.

    Hal ini dilakukan karena kelembapan, temperatur, dan kecepatan angin dianggap sebagai **variabel eksogen** dalam model. 
    Variabel eksogen adalah variabel yang memengaruhi variabel target (dalam hal ini, curah hujan) tetapi tidak dipengaruhi oleh curah hujan itu sendiri.

    Sebagai contoh:
    - `curah_hujan_lag1` adalah nilai curah hujan pada t-1.
    - `curah_hujan_lag2` adalah nilai curah hujan pada t-2, dan seterusnya.
    - `kelembapan_lag`, `temperatur_lag`, dan `kecepatan_angin_lag` tetap menggunakan nilai dari waktu **saat ini (t)**, karena mereka bersifat eksogen dan memberikan informasi tambahan yang relevan untuk model.

    Dengan pendekatan ini:
    - Model dapat mempelajari hubungan temporal antara curah hujan saat ini dengan nilai-nilai curah hujan sebelumnya melalui lag features.
    - Model juga dapat memanfaatkan informasi lingkungan dari nilai-nilai **kelembapan**, **temperatur**, dan **kecepatan angin** pada waktu yang sama (t) 
    untuk memodelkan hubungan kausalitas yang lebih baik.

    Lag features dari curah hujan dihasilkan dengan metode sliding window, sedangkan variabel eksogen tetap diambil dari waktu t tanpa perubahan.
    """)
    for lag in [1, 2, 3, 4]:
        df[f'curah_hujan_lag{lag}'] = df['curah_hujan'].shift(lag)
    df_supervised = df.dropna()
    # Setelah dropna pada df_supervised
    st.session_state['df_supervised'] = df_supervised
    st.write("Setelah menambahkan lag, data yang memiliki nilai kosong dihilangkan menggunakan `.dropna()`.")
    X = df_supervised[[col for col in df_supervised.columns if 'lag' in col]]
    y = df_supervised['curah_hujan']
    st.write("""
    **Pembagian Data Training dan Testing**:
    Dataset dibagi menjadi **data training (80%)** dan **data testing (20%)**. 
    Data training digunakan untuk melatih model, sedangkan data testing digunakan untuk mengevaluasi performa model pada data baru.
    
    Karena masalah ini berkaitan dengan data deret waktu (time series), parameter `shuffle=False` digunakan 
    dalam pembagian data agar urutan waktu tetap terjaga. Pembagian ini dilakukan menggunakan fungsi `train_test_split` dari scikit-learn.
    """)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
    st.write("### Pembagian Data:")
    st.write(f"Ukuran X_train: {X_train.shape}")
    st.write(f"Ukuran X_test: {X_test.shape}")
    st.write(f"Ukuran y_train: {y_train.shape}")
    st.write(f"Ukuran y_test: {y_test.shape}")
    st.write("""
    **Penjelasan Ukuran Data**:
    - `X_train` dan `y_train` digunakan untuk melatih model dengan data yang berisi 80% dari total dataset.
    - `X_test` dan `y_test` digunakan untuk menguji performa model dengan data yang berisi 20% dari total dataset.
    """)
    st.write("## Data Training (X_train):")
    st.write("""
    **Data Training (X_train)**:
    Berikut adalah beberapa baris pertama dari data training (`X_train`) yang berisi fitur-fitur lag (t-1, t-2, dst.).
    """)
    st.dataframe(X_train.head())
    st.write("## Data Testing (X_test):")
    st.write("""
    **Data Testing (X_test)**:
    Berikut adalah beberapa baris pertama dari data testing (`X_test`) yang juga berisi fitur-fitur lag. Data ini tidak akan digunakan dalam pelatihan model, 
    melainkan untuk mengevaluasi model yang telah dilatih.
    """)
    st.dataframe(X_test.head())
    st.session_state['X_train'] = X_train
    st.session_state['X_test'] = X_test
    st.session_state['y_train'] = y_train
    st.session_state['y_test'] = y_test
    st.success("Hasil pembagian data telah disimpan ke Session State!")


elif choice == "Modelling":
    st.title("Modelling")
    # Pastikan data hasil normalisasi tersedia di session state
    if "data_scaled" in st.session_state:
        data_scaled = st.session_state["data_scaled"]
    else:
        st.error("Data belum dinormalisasi. Silakan normalisasi data terlebih dahulu.")
        st.stop()
    # Pastikan data training/testing tersedia di session state
    if all(key in st.session_state for key in ["X_train", "X_test", "y_train", "y_test"]):
        X_train = st.session_state["X_train"]
        X_test = st.session_state["X_test"]
        y_train = st.session_state["y_train"]
        y_test = st.session_state["y_test"]
    else:
        st.error("Data training/testing belum dibagi. Silakan lakukan pembagian data terlebih dahulu.")
        st.stop()

    # Min dan Max dari data asli
    if "data_min" in st.session_state and "data_max" in st.session_state:
        data_min = st.session_state["data_min"]
        data_max = st.session_state["data_max"]
    else:
        st.error("Nilai minimum dan maksimum dari data asli tidak ditemukan. Pastikan Anda telah menyimpannya selama proses normalisasi.")
        st.stop()
    # Fungsi denormalisasi untuk curah hujan
    def denormalize(y):
        return y * (data_max - data_min) + data_min

    # Fungsi evaluasi MAPE atau fallback ke MAE
    def calculate_mape_or_mae(y_test, y_pred):
        if np.any(y_test == 0):  # Jika ada nilai nol di y_test
            mae = np.mean(np.abs(y_test - y_pred))
            st.warning("Terdapat nilai nol di data aktual. Menggunakan MAE sebagai fallback.")
            return mae, "MAE"
        else:
            mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
            return mape, "MAPE"

    # Pilihan kernel dan jumlah estimator
    kernel = st.selectbox("Pilih Kernel untuk SVR:", ["linear", "rbf", "poly"])
    n_estimators = st.selectbox("Pilih Jumlah Estimator Bagging:", [5, 10, 20])

    # Hyperparameter otomatis berdasarkan kernel
    if kernel == "linear":
        param_grid = {
            "C": [0.0001, 0.001, 0.01],
            "epsilon": [0.1, 0.01, 0.001]
        }
    elif kernel == "rbf":
        param_grid = {
            "gamma": [0.0005, 0.001, 0.01],
            "C": [0.001, 0.01, 1],
            "epsilon": [0.1, 0.01, 0.001]
        }
    elif kernel == "poly":
        param_grid = {
            "degree": [1, 2, 3],
            "C": [0.001, 0.01, 1],
            "epsilon": [0.1, 0.01, 0.001]
        }

    if st.button("Latih Model dan Prediksi"):
        # Model dan pelatihan
        svr = SVR(kernel=kernel)
        # Pencarian hyperparameter
        grid_search = GridSearchCV(svr, param_grid, cv=5, n_jobs=-1)
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_

        # Bagging dengan estimator terbaik
        bagging_model = BaggingRegressor(base_estimator=best_model, n_estimators=n_estimators, random_state=42)
        bagging_model.fit(X_train, y_train)

        # Prediksi pada data testing
        y_pred = bagging_model.predict(X_test)
        # Denormalisasi y_test dan y_pred
        y_test_denorm = denormalize(y_test)
        y_pred_denorm = denormalize(y_pred)
        # Evaluasi
        rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
        mape_or_mae, metric_name = calculate_mape_or_mae(y_test, y_pred)
        # Output evaluasi
        st.write("### Hasil Evaluasi:")
        st.write(f"**RMSE:** {rmse:.2f}")
        st.write(f"**{metric_name}:** {mape_or_mae:.2f}")
        st.write("Hyperparameter optimal:", grid_search.best_params_)
        # Simpan model ke dalam file
        joblib.dump(best_model, 'bagging_svr_model.pkl')
        st.success("Model berhasil disimpan ke file 'bagging_svr_model.pkl'")
        # Visualisasi Prediksi vs Aktual
        plt.figure(figsize=(10, 6))
        plt.plot(y_test.index, y_test_denorm, label='Actual', color='green', linestyle='-')
        plt.plot(y_test.index, y_pred_denorm, label='Predicted', color='blue', linestyle='-')
        plt.title("Actual vs Predicted Curah Hujan")
        plt.xlabel("Bulan-Tahun")
        plt.ylabel("Curah Hujan (mm)")
        plt.legend()
        plt.grid()
        st.pyplot(plt)
        # Simpan model ke session state
        st.session_state['svr_model'] = bagging_model
        st.success("Model SVR telah disimpan ke dalam Session State.")

elif choice == "Prediksi":
    st.title("Prediksi Curah Hujan")
    # Pastikan model dan data tersedia
    if 'svr_model' not in st.session_state or 'df_supervised' not in st.session_state:
        st.error("Model atau data belum tersedia. Silakan latih model atau lakukan pembagian data terlebih dahulu.")
        st.stop()
    # Ambil data dan model dari session state
    df_supervised = st.session_state['df_supervised'] 
    model = st.session_state['svr_model']

    # Input jumlah hari prediksi
    st.write("Silakan masukkan jumlah hari yang ingin diprediksi (1-14):")
    pred_days = st.number_input("Jumlah Hari Prediksi", min_value=1, max_value=14, value=1, step=1)

    # Tombol untuk memulai prediksi
    if st.button("Lakukan Prediksi"):
        # Ambil data terakhir dari df_supervised
        last_data = df_supervised.iloc[-1].copy()
        # Inisialisasi daftar untuk menyimpan hasil prediksi
        predictions = []
        # Iterasi untuk melakukan prediksi N hari ke depan
        for _ in range(pred_days):
            # Ambil hanya kolom lag sebagai input untuk model
            X_input = pd.DataFrame([last_data[[col for col in last_data.index if 'lag' in col]]])
            # Prediksi curah hujan
            next_prediction = model.predict(X_input)[0]
            predictions.append(next_prediction)
            # Perbarui data lag untuk iterasi berikutnya
            last_data['curah_hujan_lag4'] = last_data['curah_hujan_lag3']
            last_data['curah_hujan_lag3'] = last_data['curah_hujan_lag2']
            last_data['curah_hujan_lag2'] = last_data['curah_hujan_lag1']
            last_data['curah_hujan_lag1'] = next_prediction
        # Denormalisasi hasil prediksi
        if 'data_min' in st.session_state and 'data_max' in st.session_state:
            curah_hujan_min = st.session_state['data_min']
            curah_hujan_max = st.session_state['data_max']
            y_pred_denorm = [
                pred * (curah_hujan_max - curah_hujan_min) + curah_hujan_min
                for pred in predictions
            ]
        else:
            st.warning("Data asli (sebelum normalisasi) tidak tersedia. Hasil prediksi tetap dalam skala normalisasi.")
            y_pred_denorm = predictions
        # Denormalisasi data aktual (y_test)
        y_test = df_supervised['curah_hujan'][-30:].values  # Data aktual 30 hari terakhir
        y_test_denorm = [
            val * (curah_hujan_max - curah_hujan_min) + curah_hujan_min
            for val in y_test
        ]
        # Buat DataFrame untuk hasil prediksi dengan tanggal
        start_date = df_supervised.index[-1] + pd.Timedelta(days=1)  # Mulai prediksi dari hari setelah data terakhir
        future_dates = pd.date_range(start=start_date, periods=pred_days)
        pred_df = pd.DataFrame({
            'Tanggal': future_dates,
            'Prediksi Curah Hujan': y_pred_denorm
        })
        # Tampilkan hasil prediksi
        st.write(f"### Prediksi Curah Hujan {pred_days} Hari ke Depan:")
        st.dataframe(pred_df)
        # Visualisasi hasil prediksi
        st.write("### Grafik Prediksi:")
        fig, ax = plt.subplots(figsize=(12, 6))
        # Plot data aktual dari 30 hari terakhir
        actual_dates = df_supervised.index[-30:]
        ax.plot(actual_dates, y_test_denorm, label='Data Aktual (30 Hari Terakhir)', color='red', marker='o')
        # Plot data prediksi
        ax.plot(pred_df['Tanggal'], pred_df['Prediksi Curah Hujan'], label=f'Prediksi {pred_days} Hari ke Depan', color='green', marker='o')
        ax.set_xlabel("Tanggal")
        ax.set_ylabel("Curah Hujan (mm)")
        ax.set_title(f"Prediksi Curah Hujan {pred_days} Hari ke Depan")
        ax.legend()
        ax.grid()
        plt.xticks(rotation=45)
        st.pyplot(fig)






