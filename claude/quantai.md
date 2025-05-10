**Audit Kode `quantai.py` dan Pencarian Kesalahan Logika:**

**Rekomendasi Peningkatan dan Otomatisasi (berdasarkan API MetaTrader 5 dan TensorFlow/Keras):**

Berikut adalah daftar API yang relevan dari dokumen yang Anda berikan, beserta cara mereka dapat diintegrasikan untuk optimasi dan peningkatan fungsionalitas:

**A. Peningkatan Pipeline Data dan Feature Engineering (Integrasi `quantai.py` dengan API Tambahan):**

1.  **TensorFlow/Keras API untuk Data:**
    * **`TimeSeriesGenerator` (Utilitas):** Meskipun Anda sudah menggunakan `tf.data.Dataset.from_tensor_slices` dan windowing manual, `TimeSeriesGenerator` bisa menjadi alternatif yang lebih *high-level* jika Anda memiliki data yang belum di-window dan ingin melakukannya dengan cara Keras yang lebih terstruktur, terutama jika tidak memerlukan fleksibilitas penuh dari windowing manual.
        * **Peningkatan:** Bisa menyederhanakan bagian pembuatan `X_windowed` dan `y_windowed`.
    * **`WindowGenerator` (Utilitas):** Jika Anda memerlukan fungsionalitas windowing yang lebih kompleks (misalnya, input dengan lebar berbeda, offset berbeda, label dengan lebar berbeda), ini bisa sangat berguna. Kode Anda saat ini melakukan windowing secara manual, yang sudah cukup untuk kasus ini.
    * **`FeatureEngineering` (Integrasi Data):**
        * **Peningkatan:** Ini adalah area krusial. Fungsi `add_technical_features` Anda **harus diintegrasikan**. Anda bisa melakukannya dengan:
            1.  **Sebagai `tf.keras.layers.Lambda` layer:** Setelah data di-batch, Anda bisa menambahkan Lambda layer yang memanggil fungsi ini. Ini akan membuat feature engineering bagian dari grafik komputasi model.
                ```python
                # Di dalam build_model:
                # inputs = Input(shape=input_shape)
                # technical_features_layer = tf.keras.layers.Lambda(add_technical_features)(inputs)
                # ... lanjutkan dengan layer LSTM/CNN menggunakan output dari technical_features_layer atau gabungan
                ```
            2.  **Preprocessing Step di `tf.data` pipeline:**
                ```python
                # def preprocess_batch(x_batch, y_batch):
                #     processed_x = add_technical_features(x_batch) # Perlu disesuaikan jika add_technical_features tidak vectorized untuk batch
                #     return processed_x, y_batch
                # train_dataset = train_dataset.map(preprocess_batch, num_parallel_calls=tf.data.AUTOTUNE)
                ```
                Perlu diperhatikan bahwa `add_technical_features` Anda saat ini dirancang untuk input berbentuk `[batch, window, features]`. Jika Anda menerapkannya *sebelum* batching, Anda perlu menyesuaikannya. Menerapkannya sebagai Lambda layer setelah batching mungkin lebih mudah.
    * **`OHLCVDataset` (Integrasi Data):** Jika Anda memiliki loader data khusus, Anda bisa mengabstraksikannya menjadi kelas seperti ini untuk reusabilitas. Saat ini, Anda memuat dari JSON secara langsung.
    * **`MinMaxScaler` (Integrasi Data):** Anda sudah menggunakan `sklearn.preprocessing.MinMaxScaler`. Ini sudah baik. TensorFlow juga memiliki `tf.keras.layers.Normalization` yang bisa diadaptasi sebagai bagian dari model (pelajari statistik normalisasi dari data training).
    * **`Augmentation` (Integrasi Data):** Untuk data time series finansial, augmentasi harus hati-hati. Teknik seperti menambahkan noise kecil atau time warping bisa dieksplorasi, tetapi harus divalidasi dampaknya.
        * **Peningkatan Fungsional:** Dapat membantu model menjadi lebih robust jika dilakukan dengan benar.
    * **`MICE` (Integrasi Data - Multiple Imputation by Chained Equations):** Jika data Anda sering memiliki nilai yang hilang (NaNs) yang signifikan, MICE adalah teknik yang lebih canggih untuk mengatasinya daripada sekadar menghapus baris atau mengisi dengan mean/median.
        * **Peningkatan:** Meningkatkan kualitas data input jika banyak missing values.
    * **`tf.data.Options()`:** Anda sudah menggunakannya, ini bagus untuk optimasi performa.

2.  **MetaTrader 5 API untuk Data Real-time dan Historis:**
    * **`mt5.copy_rates_from()` / `mt5.copy_rates_from_pos()` / `mt5.copy_rates_range()`:**
        * **Otomatisasi & Peningkatan:** Daripada memuat data dari file JSON statis (`XAU_1d_data_processed.json`), Anda bisa mengotomatiskan pengambilan data historis terbaru langsung dari MetaTrader 5. Ini memastikan model Anda dilatih dengan data paling mutakhir.
        * **Integrasi:** Buat fungsi yang menggunakan API ini untuk mengambil data OHLCV untuk XAUUSD (atau simbol lain), memprosesnya (mirip dengan Langkah 2 Anda), lalu menyimpannya ke format yang bisa dibaca oleh pipeline Anda atau langsung memasukkannya ke pipeline.
    * **`mt5.symbol_info_tick()`:**
        * **Peningkatan Fungsional:** Untuk prediksi real-time, `predict_next_day_prices` saat ini memerlukan `latest_data` sebagai input manual. Anda bisa memodifikasinya untuk mengambil `window_size` data terakhir langsung dari MT5 menggunakan `copy_rates_from_pos` dan kemudian tick terbaru jika diperlukan untuk fitur yang sangat up-to-date.
    * **`mt5.initialize()`, `mt5.login()`, `mt5.shutdown()`:** Ini adalah dasar untuk berinteraksi dengan MT5. Harus diimplementasikan di awal dan akhir skrip yang berinteraksi dengan MT5.

**B. Peningkatan Arsitektur Model dan Pelatihan:**

1.  **TensorFlow/Keras API untuk Model dan Layer:**
    * **Model Khusus (Sudah Digunakan Sebagian):**
        * `LSTM`, `Conv1D`, `MultiHeadAttention`, `GlobalAveragePooling1D`, `Dense`, `Dropout`, `Input`, `Add`, `LayerNormalization`: Anda sudah menggunakan banyak layer fundamental dengan baik.
    * **Model Khusus (Potensi untuk Ditambahkan/Eksplorasi):**
        * **`GRU` (Gated Recurrent Unit):** Alternatif untuk LSTM, seringkali dengan performa serupa tetapi lebih sedikit parameter. Bisa diuji sebagai salah satu model dalam ensemble.
        * **`Bidirectional` (Layer Wrapper):** Membungkus layer LSTM atau GRU dengan `Bidirectional` dapat memungkinkan model untuk belajar dari sekuens baik dari arah maju maupun mundur, yang terkadang meningkatkan performa pada data time series.
            ```python
            # from tensorflow.keras.layers import Bidirectional
            # x = Bidirectional(LSTM(64, return_sequences=True))(inputs)
            ```
        * **`Transformer` / `Attention LSTM` / `Informer` / `Autoformer`:** Anda sudah menggunakan `MultiHeadAttention`. Model Transformer lengkap atau varian yang lebih canggih seperti Informer (untuk prediksi jangka panjang) bisa dieksplorasi jika kompleksitasnya sesuai dengan dataset Anda. Ini biasanya memerlukan lebih banyak data dan tuning.
        * **`TCN` (Temporal Convolutional Network):** Alternatif atau pelengkap untuk RNN. TCN menggunakan konvolusi kausal dan dilasi, bisa efektif untuk menangkap dependensi jangka panjang.
        * **`WaveNet`:** Mirip dengan TCN, menggunakan dilated causal convolutions.
        * **`Bayesian LSTM` (atau layer probabilistik lainnya dari `TensorFlow Probability`):**
            * **Peningkatan Fungsional:** Alih-alih hanya memprediksi satu nilai (misalnya, harga penutupan), model Bayesian dapat memberikan distribusi probabilitas prediksi. Ini sangat berguna untuk manajemen risiko karena memberikan estimasi ketidakpastian.
            * Integrasi: Ganti layer Dense terakhir dengan layer dari TFP (misalnya, `tfp.layers.DenseVariational`) dan gunakan loss function yang sesuai (misalnya, negative log-likelihood).
    * **Representasi Model:**
        * `Functional API`: Anda sudah menggunakannya, yang sangat baik untuk model kompleks.
    * **Optimalisasi:**
        * **`KerasTuner`:**
            * **Otomatisasi & Peningkatan:** Sangat direkomendasikan untuk mengotomatiskan pencarian hyperparameter (jumlah unit di layer, learning rate, dropout rate, jumlah head attention, dll.). Ini dapat secara signifikan meningkatkan performa model dibandingkan tuning manual.
        * **`Pruning API`:** Jika ukuran model menjadi perhatian (misalnya, untuk deployment di perangkat terbatas, meskipun kurang relevan untuk server-side trading), pruning dapat membantu mengurangi ukuran model dengan dampak minimal pada akurasi.
        * **`MixedPrecisionPolicy`:** Jika Anda berlatih di GPU yang mendukungnya (misalnya, NVIDIA Volta, Turing, Ampere), menggunakan mixed precision dapat mempercepat pelatihan dan mengurangi penggunaan memori GPU.
            ```python
            # from tensorflow.keras import mixed_precision
            # policy = mixed_precision.Policy('mixed_float16')
            # mixed_precision.set_global_policy(policy)
            ```
        * **`XLACompiler` (Accelerated Linear Algebra):** TensorFlow dapat menggunakan XLA untuk mengoptimalkan grafik komputasi. Seringkali diaktifkan secara otomatis atau bisa di-enable dengan `tf.config.optimizer.set_jit(True)`.
    * **Kasus Penggunaan Khusus (untuk Pelatihan):**
        * **`Cross-Validation` (dan `TimeSeriesSplit`):** Anda sudah menggunakannya.
        * **`EarlyStopping`, `ModelCheckpoint`, `TensorBoard`:** Sudah digunakan.
        * **`CustomLoss`:** Jika MSE atau MAE standar tidak sepenuhnya mencerminkan tujuan trading Anda (misalnya, Anda ingin memberi penalti lebih pada kesalahan prediksi arah atau kesalahan besar pada volatilitas tertentu), Anda bisa mendefinisikan fungsi loss kustom.
            * **Peningkatan Fungsional:** Menyelaraskan optimasi model lebih dekat dengan tujuan bisnis/trading.
        * **`ModelEnsemble`:** Anda sudah mengimplementasikan ensembling secara manual. Ada library yang bisa membantu mengelola ini (misalnya, bagian dari `scikit-learn` atau library ensembling khusus).
        * **`Feature Importance`:** Setelah model dilatih, gunakan teknik seperti permutation importance atau SHAP (SHapley Additive exPlanations) untuk memahami fitur mana yang paling berpengaruh pada prediksi. Ini dapat memberikan wawasan dan membantu dalam pemilihan fitur di masa depan.
            * **Peningkatan:** Membantu memahami "black box" dan memvalidasi apakah model belajar pola yang masuk akal.

Dengan mengintegrasikan API ini dan memperbaiki logika yang disebutkan, Anda dapat meningkatkan akurasi model, mengotomatiskan pipeline data, dan membangun sistem trading algoritmik yang lebih canggih dan robust. Mulailah dengan perbaikan logika inti dalam `quantai.py` sebelum beralih ke otomatisasi trading penuh.
