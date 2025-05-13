import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import LSTM, Dense, Dropout, MultiHeadAttention, LayerNormalization, GlobalAveragePooling1D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
# Tambah metrik MAPE
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from flask import Flask, request, jsonify
from dotenv import load_dotenv
import os
import logging
import joblib # Untuk menyimpan/memuat scaler
import matplotlib.pyplot as plt # Untuk visualisasi
import json # Untuk menyimpan metadata
import sys # Untuk keluar jika ada error fatal

# ============ Load Env & Logging ============
# Memuat variabel lingkungan dari file .env.
# Pastikan ada file .env di direktori yang sama
# dengan isi seperti ini:
# MT5_LOGIN=your_login_id
# MT5_SERVER=your_server_name
# MT5_PASSWORD=your_password
# JANGAN simpan kredensial di kode sumber!
load_dotenv()
# Format log lebih informatif dengan waktu
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ============ MetaTrader5 Setup ============
def connect_mt5():
    """Menghubungkan ke terminal MetaTrader5 menggunakan kredensial dari .env."""
    # Mengambil kredensial dari variabel lingkungan
    login_str = os.getenv("MT5_LOGIN")
    server = os.getenv("MT5_SERVER")
    password = os.getenv("MT5_PASSWORD")

    # Validasi keberadaan variabel lingkungan
    if not login_str or not server or not password:
        logging.error("❌ Kredensial MT5 (MT5_LOGIN, MT5_SERVER, MT5_PASSWORD) tidak ditemukan di file .env atau environment variables.")
        logging.error("Mohon buat file .env atau atur variabel lingkungan dengan kredensial yang benar.")
        # Menggunakan sys.exit(1) untuk keluar dari program jika konfigurasi dasar gagal
        sys.exit(1)

    try:
        login = int(login_str)
    except ValueError:
        logging.error(f"❌ Nilai MT5_LOGIN di .env bukan angka: '{login_str}'")
        sys.exit(1)

    # Melakukan inisialisasi koneksi MT5
    # Untuk skrip yang berjalan singkat seperti ini (ambil data sekali), inisialisasi di sini OK.
    # Untuk aplikasi jangka panjang/periodik, inisialisasi di awal aplikasi dan shutdown di akhir.
    # Tambahkan retry sederhana untuk error umum
    max_retries = 3
    for i in range(max_retries):
        if mt5.initialize(login=login, server=server, password=password):
            logging.info("✅ Berhasil login ke MetaTrader5")
            return
        last_error = mt5.last_error()
        logging.warning(f"⚠️ Gagal koneksi MT5 (Percobaan {i+1}/{max_retries}). Kode error: {last_error}")
        if i < max_retries - 1:
             # Tunggu sebelum mencoba lagi (contoh: 5 detik)
             import time
             time.sleep(5)
    
    # Jika semua percobaan gagal
    logging.error(f"❌ Gagal koneksi MT5 setelah {max_retries} percobaan.")
    raise ConnectionError(f"Gagal koneksi MT5: {last_error}") # Melempar error setelah semua retry gagal


def acquire_data(symbol, timeframe, start, end):
    """Mengambil data historis dari MetaTrader5 dan melakukan shutdown."""
    connect_mt5() # Panggil fungsi koneksi
    logging.info(f"⏳ Mengambil data {symbol} dari {start} hingga {end} dengan timeframe {timeframe}...")
    
    try:
        rates = mt5.copy_rates_range(symbol, timeframe, start, end)
    except Exception as e:
        # Tangani error saat mengambil data, pastikan `rates` menjadi None
        logging.error(f"❌ Error saat memanggil copy_rates_range: {e}")
        rates = None
    finally:
        # Selalu shutdown koneksi setelah selesai mengambil data, terlepas dari sukses atau gagal
        mt5.shutdown()
        logging.info("✅ Koneksi MT5 di-shutdown.")

    if rates is None or len(rates) == 0:
        # Gunakan pesan error yang lebih spesifik
        error_msg = f"❌ Gagal mengambil data dari MT5 untuk {symbol} ({start} to {end}). Data kosong atau terjadi kesalahan."
        logging.error(error_msg)
        raise ValueError(error_msg)

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    # Urutkan berdasarkan waktu untuk memastikan kronologi yang benar (penting untuk timeseries)
    df = df.sort_values('time').reset_index(drop=True)
    logging.info(f"✅ Berhasil mengambil {len(df)} baris data.")

    return df[['time', 'open', 'high', 'low', 'close', 'tick_volume']]

# ============ Preprocessing ============
def preprocess(df):
    """
    Melakukan pra-pemrosesan data:
    - Menghapus baris dengan nilai NaN.
    - Menambahkan fitur turunan (teknikal indikator sederhana).
    - Melakukan scaling pada fitur input (X).
    """
    df = df.dropna().reset_index(drop=True)
    if df.empty:
         raise ValueError("DataFrame kosong setelah menghapus NaN awal.")

    # --- Menambahkan Fitur Teknikal ---
    # SMA (Simple Moving Average)
    df['sma_5'] = df['close'].rolling(window=5, min_periods=1).mean()
    # EMA (Exponential Moving Average)
    df['ema_5'] = df['close'].ewm(span=5, adjust=False).mean()
    # Tambahkan fitur teknikal lainnya di sini jika perlu (misal: RSI, MACD)
    # Anda mungkin perlu pustaka seperti `pandas_ta` untuk indikator yang lebih kompleks.
    # Contoh:
    # import pandas_ta as ta
    # df['rsi_14'] = df['close'].ta.rsi(length=14)
    # macd_data = df['close'].ta.macd(fast=12, slow=26, signal=9)
    # df[['macd', 'macd_hist', 'macd_signal']] = macd_data

    # Identifikasi fitur yang akan digunakan (sesuaikan jika menambah fitur)
    features = ['open','high','low','close','tick_volume','sma_5', 'ema_5']

    # Hapus baris yang mungkin memiliki NaN setelah penambahan fitur teknikal
    df = df.dropna().reset_index(drop=True)
    if df.empty:
         raise ValueError("DataFrame kosong setelah menambahkan fitur teknikal dan menghapus NaN lagi.")
         
    # Pastikan semua kolom fitur yang diharapkan ada
    if not all(f in df.columns for f in features):
         missing_features = [f for f in features if f not in df.columns]
         raise ValueError(f"Fitur yang diharapkan tidak ada di DataFrame setelah pra-pemrosesan: {missing_features}")

    logging.info(f"✅ Data diproses. Menggunakan {len(features)} fitur: {features}")

    # Scaling fitur X (data input) menggunakan MinMaxScaler
    scaler_X = MinMaxScaler()
    df[features] = scaler_X.fit_transform(df[features])
    logging.info("✅ Fitur input (X) berhasil di-scale.")

    return df, scaler_X, features

def create_dataset(df, window_size, feature_cols):
    """
    Membuat dataset jendela (windowed dataset) untuk training time series.
    Setiap sampel X adalah jendela `window_size` langkah waktu,
    Setiap sampel y adalah target (harga high, low, close) pada langkah waktu berikutnya setelah jendela.
    """
    X, y = [], []
    # Loop untuk membuat jendela
    # Jendela berakhir di index i + window_size - 1
    # Target adalah di index i + window_size
    # Loop harus berhenti `window_size` langkah sebelum akhir data
    for i in range(len(df) - window_size):
        # Ambil data jendela untuk fitur X
        window = df.iloc[i : i + window_size][feature_cols].values
        # Ambil data target (harga H/L/C) pada langkah waktu SETELAH jendela
        target = df.iloc[i + window_size][['high','low','close']].values
        X.append(window)
        y.append(target)

    X = np.array(X)
    y = np.array(y)

    # Pastikan dataset berhasil dibuat
    if len(X) == 0:
         raise ValueError(f"Tidak cukup data untuk membuat dataset dengan window_size={window_size}. Butuh minimal {window_size+1} baris data valid setelah pra-pemrosesan.")

    logging.info(f"✅ Dataset dibuat. Shape X: {X.shape}, Shape y: {y.shape}")
    return X, y

# ============ Model (LSTM + Attention) ============
def build_lstm_attention_model(input_shape, output_dim):
    """Membangun model LSTM dengan layer Attention dan residual connection."""
    inputs = Input(shape=input_shape) # Input shape: (window_size, num_features)

    # Layer LSTM, return_sequences=True untuk output sequence ke layer berikutnya
    x = LSTM(64, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(0.001))(inputs) # Contoh regularisasi L2
    # x = Dropout(0.2)(x) # Opsional: Dropout setelah LSTM

    # Layer MultiHeadAttention
    # key_dim harus membagi dimensi terakhir input attention_input (yaitu 64 dari output LSTM)
    # num_heads * key_dim = last_dimension_of_attention_input
    attention_output = MultiHeadAttention(num_heads=2, key_dim=32, dropout=0.1)(x, x) # Dropout di Attention

    # Residual connection: Tambahkan output attention ke inputnya
    # Pastikan shape attention_output sama dengan shape x (return_sequences=True)
    # Jika shape tidak cocok, residual connection tidak bisa dilakukan langsung.
    # Dalam kasus ini, shape x (batch, timesteps, 64) dan attention_output (batch, timesteps, 64) cocok.
    x = tf.keras.layers.Add()([x, attention_output])

    # Layer Normalization setelah residual connection
    x = LayerNormalization()(x)

    # Layer GlobalAveragePooling1D untuk mereduksi dimensi waktu
    x = GlobalAveragePooling1D()(x) # Output shape: (batch, last_dimension)

    # Dense layers sebelum output
    x = Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x) # Contoh regularisasi L2
    x = Dropout(0.3)(x) # Dropout setelah Dense
    # x = Dense(16, activation='relu')(x) # Opsional: Dense layer lain

    # Output layer: Linear activation untuk prediksi harga (regresi)
    outputs = Dense(output_dim)(x) # Output shape: (batch, 3) untuk High, Low, Close

    model = Model(inputs=inputs, outputs=outputs)
    logging.info("✅ Model LSTM + Attention berhasil dibangun.")
    model.summary(print_fn=lambda msg: logging.info(msg)) # Print summary ke log

    # Komentar tentang Model:
    # - Arsitektur ini adalah kombinasi yang valid untuk timeseries.
    # - Jumlah unit LSTM, heads Attention, dan Dense units adalah hyperparameter.
    # - Regularisasi L2 dan Dropout ditambahkan sebagai contoh untuk membantu mencegah overfitting.
    # - Dapat dieksplorasi arsitektur lain (CNN, Transformer) atau model ensemble.

    return model

# ============ Training ============
def train_model(model, train_ds, val_ds, name):
    """
    Melatih model menggunakan dataset TensorFlow.
    Menggunakan EarlyStopping dan ModelCheckpoint.
    """
    logging.info(f"⏳ Memulai training model fold: {name}...")
    # Compile model: optimizer Adam, loss MSE (untuk regresi), metrik MAE dan MSE
    model.compile(optimizer='adam', loss='mse', metrics=['mae', 'mse'])
    
    # Path untuk menyimpan checkpoint bobot model terbaik selama training fold ini
    # Hanya simpan bobot (save_weights_only=True) agar lebih cepat, atau seluruh model (.h5)
    # Kita simpan seluruh model karena akan dimuat kembali nanti.
    checkpoint_path = f"model_artifacts/{name}_checkpoint.h5" 

    # Callbacks:
    # - EarlyStopping: Menghentikan training jika metrik monitor (val_loss) tidak membaik
    #   dalam jumlah `patience` epoch. `restore_best_weights=True` akan memuat bobot terbaik.
    # - ModelCheckpoint: Menyimpan model atau bobot terbaik berdasarkan metrik monitor.
    callbacks = [
        EarlyStopping(patience=10, restore_best_weights=True, verbose=1, monitor='val_loss'), # Kesabaran 10 epoch
        ModelCheckpoint(checkpoint_path, save_best_only=True, verbose=1, monitor='val_loss')
    ]

    # Melatih model
    # Gunakan jumlah epoch yang cukup tinggi, EarlyStopping akan menghentikan secara otomatis
    history = model.fit(train_ds, validation_data=val_ds, epochs=200, callbacks=callbacks, verbose=1) # Contoh epoch lebih tinggi

    # Setelah model.fit dengan restore_best_weights=True, model di memori sudah memiliki bobot terbaik
    # yang dicatat oleh ModelCheckpoint. Tidak perlu load_weights() lagi dari checkpoint_path
    # kecuali untuk keperluan debugging atau verifikasi file.
    # logging.info(f"✅ Training fold '{name}' selesai. Bobot terbaik sudah di-restore.")

    # Model `model` sekarang berisi bobot terbaik dari fold ini (berdasarkan val_loss)
    return model, history # Mengembalikan model dengan bobot terbaik dan history training

# ============ Evaluation ============
def evaluate_model(model, X_set, y_set_scaled, scaler_y, set_name="Test"):
    """
    Mengevaluasi model pada dataset yang diberikan.
    Mengembalikan prediksi dan nilai aktual dalam skala asli, serta metrik evaluasi.
    """
    logging.info(f"⏳ Mengevaluasi model pada set {set_name} ({len(X_set)} sampel)...")
    
    if len(X_set) == 0:
         logging.warning(f"⚠️ Tidak ada data di set {set_name} untuk dievaluasi.")
         return np.array([]), np.array([]), {'mae': np.nan, 'rmse': np.nan, 'mape': np.nan}

    # Melakukan prediksi dalam skala scaled
    preds_scaled = model.predict(X_set, verbose=0)

    # Mengembalikan nilai aktual dan prediksi ke skala asli
    try:
        y_set_original = scaler_y.inverse_transform(y_set_scaled)
        preds_original = scaler_y.inverse_transform(preds_scaled)
    except Exception as e:
        logging.error(f"❌ Gagal inverse transform data untuk set {set_name}: {e}")
        # Kembalikan data scaled jika inverse transform gagal
        return preds_scaled, y_set_scaled, {'mae': np.nan, 'rmse': np.nan, 'mape': np.nan}


    # Menghitung metrik evaluasi dalam skala asli
    mae = mean_absolute_error(y_set_original, preds_original)
    rmse = np.sqrt(mean_squared_error(y_set_original, preds_original))
    
    # Menghitung MAPE, tambahkan epsilon kecil untuk menghindari pembagian dengan nol
    # Periksa juga apakah nilai aktual mendekati nol
    if np.all(y_set_original < 1e-6): # Jika semua nilai aktual sangat kecil (dekat nol)
         mape = np.nan # MAPE tidak relevan atau akan menjadi Inf
         logging.warning(f"⚠️ Semua nilai aktual di set {set_name} mendekati nol. MAPE tidak dihitung.")
    else:
         mape = np.mean(np.abs((y_set_original - preds_original) / (y_set_original + 1e-8))) * 100 # Tambah epsilon

    logging.info(f"📊 Hasil Evaluasi {set_name}:")
    logging.info(f"  MAE  (skala asli): {mae:.4f}")
    logging.info(f"  RMSE (skala asli): {rmse:.4f}")
    if not np.isnan(mape):
         logging.info(f"  MAPE (skala asli): {mape:.2f}%")
    else:
         logging.info("  MAPE (skala asli): N/A")


    return preds_original, y_set_original, {'mae': mae, 'rmse': rmse, 'mape': mape}

# ============ Main Training ============
def main_training():
    """
    Fungsi utama untuk akuisisi data, pra-pemrosesan,
    training model menggunakan TSCV, evaluasi pada set test akhir,
    dan penyimpanan artefak.
    """
    logging.info("--- Memulai Proses Training Model ---")
    try:
        # --- Akuisisi Data ---
        # Ambil data untuk periode tertentu
        # Pertimbangkan menggunakan data yang lebih baru untuk relevansi pasar saat ini
        df = acquire_data('XAUUSD', mt5.TIMEFRAME_D1, pd.Timestamp(2008,1,1), pd.Timestamp(2023,12,31)) # Contoh: Diperpanjang hingga 2023

        # --- Pra-pemrosesan Data ---
        df, scaler_X, feature_cols = preprocess(df)
        
        # --- Pembuatan Dataset Jendela ---
        WINDOW_SIZE = 10 # Tentukan ukuran jendela di sini atau sebagai konfigurasi
        X_all, y_raw_all = create_dataset(df, window_size=WINDOW_SIZE, feature_cols=feature_cols)

        # Scaling target y (harga H/L/C berikutnya)
        scaler_y = MinMaxScaler()
        y_all = scaler_y.fit_transform(y_raw_all)
        logging.info("✅ Target output (y) berhasil di-scale.")

        # --- Setup Direktori Artefak ---
        # Buat direktori untuk menyimpan model, scaler, plot, dan metadata
        ARTIFACTS_DIR = "model_artifacts"
        os.makedirs(ARTIFACTS_DIR, exist_ok=True)
        logging.info(f"✅ Direktori artefak '{ARTIFACTS_DIR}' siap.")

        # --- Simpan Scaler ---
        # Simpan objek scaler ke file untuk digunakan kembali saat inferensi (API)
        joblib.dump(scaler_X, os.path.join(ARTIFACTS_DIR, "scaler_X.save"))
        joblib.dump(scaler_y, os.path.join(ARTIFACTS_DIR, "scaler_y.save"))
        logging.info(f"✅ Scaler X dan Y disimpan di '{ARTIFACTS_DIR}'.")

        # --- Split Data Training/Validasi vs Test ---
        # Bagi data X_all dan y_all menjadi set training/validation (trainval) dan set test akhir.
        # Set test adalah data TERAKHIR secara kronologis.
        TEST_SET_SIZE_PERCENT = 0.2 # Ukuran test set: 20% data terakhir
        split_idx = int(len(X_all) * (1 - TEST_SET_SIZE_PERCENT))
        
        if split_idx < WINDOW_SIZE + 1: # Pastikan ada cukup data di trainval
             raise ValueError(f"Ukuran data tidak cukup untuk split ({len(X_all)} sampel). Butuh minimal {WINDOW_SIZE + 1} untuk trainval.")

        X_trainval, y_trainval = X_all[:split_idx], y_all[:split_idx]
        X_test, y_test = X_all[split_idx:], y_all[split_idx:]
        logging.info(f"✅ Data dibagi: Training+Validation ({len(X_trainval)} sampel), Test ({len(X_test)} sampel)")
        
        # Beri peringatan jika set test terlalu kecil
        if len(X_test) < WINDOW_SIZE + 1:
             logging.warning(f"⚠️ Jumlah data test ({len(X_test)}) mungkin terlalu sedikit. Pertimbangkan rentang data atau persentase split yang berbeda.")


        # --- Time Series Cross-Validation (TSCV) ---
        # TSCV membagi data trainval menjadi fold training dan validasi secara kronologis
        # Model terbaik dipilih berdasarkan performa pada fold validasi TSCV
        N_SPLITS_TSCV = 5 # Jumlah fold untuk TSCV
        tscv = TimeSeriesSplit(n_splits=N_SPLITS_TSCV)
        
        best_val_loss_in_tscv = float('inf') # Monitor val_loss untuk EarlyStopping & Checkpoint
        # Path sementara untuk menyimpan model terbaik dari validasi TSCV
        temp_model_path = os.path.join(ARTIFACTS_DIR, "temp_best_model_fold_tscv.h5") 

        logging.info(f"⏳ Memulai Time Series Cross-Validation dengan {tscv.n_splits} fold pada data Training+Validation...")

        # Loop melalui setiap fold TSCV
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_trainval)):
            logging.info(f"--- 🔁 Fold {fold+1}/{tscv.n_splits} ---")
            
            # Ambil data untuk training dan validasi fold saat ini
            X_train, y_train = X_trainval[train_idx], y_trainval[train_idx]
            X_val, y_val = X_trainval[val_idx], y_trainval[val_idx]

            # Buat dataset TensorFlow dari data train/val fold
            # Shuffle hanya pada training set, bukan validasi atau test set
            # AUTOTUNE memungkinkan TensorFlow menyesuaikan prefetching dan batching
            train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(len(X_train), seed=42).batch(32).prefetch(tf.data.AUTOTUNE) # Tambah seed untuk reproducibility
            val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(32).prefetch(tf.data.AUTOTUNE)

            # Bangun instance model baru untuk setiap fold
            # Ini memastikan setiap fold mulai dari bobot acak yang baru
            fold_model = build_lstm_attention_model(X_trainval.shape[1:], y_trainval.shape[1])
            
            # Latih model pada data training fold, validasi pada data validasi fold
            # Checkpoint akan menyimpan model terbaik fold ini berdasarkan val_loss
            trained_fold_model, history = train_model(fold_model, train_ds, val_ds, name=f"fold_{fold+1}")

            # Evaluasi model fold ini pada set validasinya (dalam skala scaled)
            # Gunakan bobot terbaik yang sudah di-restore oleh EarlyStopping
            val_loss, val_mae, val_mse = trained_fold_model.evaluate(val_ds, verbose=0)

            logging.info(f"✅ Fold {fold+1} selesai. Val Loss (scaled): {val_loss:.4f}, Val MAE (scaled): {val_mae:.4f}")

            # Logika untuk memilih model terbaik di antara semua fold TSCV
            # Pilih berdasarkan Val Loss (karena EarlyStopping & Checkpoint juga pakai Val Loss)
            if val_loss < best_val_loss_in_tscv:
                best_val_loss_in_tscv = val_loss
                # Simpan model terbaik dari VALIDASI fold ini ke path sementara
                # Model yang disimpan sudah memiliki bobot terbaik karena restore_best_weights=True
                trained_fold_model.save(temp_model_path) 
                logging.info(f"⭐ Model Fold {fold+1} adalah yang terbaik sejauh ini (Val Loss scaled: {best_val_loss_in_tscv:.4f}), disimpan sementara di '{temp_model_path}'.")

        # --- Evaluasi Akhir pada Test Set ---
        # Muat model terbaik yang ditemukan selama TSCV (berdasarkan Val Loss) dari file sementara
        logging.info(f"--- ⏳ Memuat model terbaik dari '{temp_model_path}' untuk evaluasi akhir pada Test Set ---")
        
        # Tambahkan custom_objects jika model menggunakan layer kustom
        custom_objects = {'MultiHeadAttention': MultiHeadAttention}
        
        if not os.path.exists(temp_model_path):
             logging.error(f"❌ File model terbaik sementara '{temp_model_path}' tidak ditemukan! Proses training mungkin gagal.")
             sys.exit(1) # Keluar jika file tidak ada
             
        final_best_model_from_tscv = tf.keras.models.load_model(temp_model_path, custom_objects=custom_objects)

        # Lakukan evaluasi pada set data test yang terpisah (dalam skala asli)
        test_preds_original, y_test_original, test_metrics = evaluate_model(
            final_best_model_from_tscv, X_test, y_test, scaler_y, set_name="Test Set Akhir"
        )

        # --- Simpan Model Terbaik Final ---
        # Simpan model terbaik (yang terbaik di validasi TSCV dan dievaluasi di test)
        # Ini adalah model yang akan digunakan oleh API
        FINAL_MODEL_PATH = os.path.join(ARTIFACTS_DIR, "best_model.h5")
        final_best_model_from_tscv.save(FINAL_MODEL_PATH)
        logging.info(f"✅ Model terbaik (berdasarkan validasi TSCV) disimpan di '{FINAL_MODEL_PATH}'.")

        # --- Hapus File Model Sementara ---
        # Hapus file model checkpoint per fold (jika tidak ingin menyimpannya)
        # Hapus file model sementara setelah model final disimpan
        try:
            for fold in range(N_SPLITS_TSCV):
                 fold_checkpoint_path = os.path.join(ARTIFACTS_DIR, f"fold_{fold+1}_checkpoint.h5")
                 if os.path.exists(fold_checkpoint_path):
                      os.remove(fold_checkpoint_path)
                      logging.info(f"✅ Checkpoint fold {fold+1} dihapus: '{fold_checkpoint_path}'")

            if os.path.exists(temp_model_path):
                 os.remove(temp_model_path)
                 logging.info(f"✅ File model sementara dihapus: '{temp_model_path}'")
                 
        except Exception as e:
             logging.warning(f"⚠️ Gagal menghapus file model sementara: {e}")


        # --- Visualisasi Hasil Prediksi pada Test Set ---
        logging.info("⏳ Membuat visualisasi prediksi pada Test Set...")
        plt.figure(figsize=(15, 8)) # Ukuran plot lebih besar

        # Visualisasi beberapa titik data terakhir dari test set agar lebih terlihat
        # Pilih jumlah titik yang masuk akal
        num_points_to_plot = min(len(y_test_original), 150) # Plot 150 titik terakhir atau kurang jika data test < 150
        start_plot_idx = len(y_test_original) - num_points_to_plot

        # Gunakan indeks waktu aktual jika tersedia di DataFrame asli sebelum create_dataset
        # Saat ini menggunakan indeks array, yang mungkin tidak ideal.
        # Untuk visualisasi yang lebih baik, simpan tanggal/waktu saat membuat dataset.
        
        plot_indices = np.arange(start_plot_idx, len(y_test_original))

        plt.subplot(3, 1, 1) # 3 baris, 1 kolom, plot ke-1 (High)
        plt.plot(plot_indices, y_test_original[start_plot_idx:, 0], label='Aktual High', color='blue', marker='.')
        plt.plot(plot_indices, test_preds_original[start_plot_idx:, 0], label='Prediksi High', color='red', linestyle='--', marker='.')
        plt.title(f'Prediksi vs Aktual High - Test Set ({num_points_to_plot} Titik Terakhir)')
        plt.ylabel("Harga")
        plt.legend()
        plt.grid(True)

        plt.subplot(3, 1, 2) # Plot ke-2 (Low)
        plt.plot(plot_indices, y_test_original[start_plot_idx:, 1], label='Aktual Low', color='blue', marker='.')
        plt.plot(plot_indices, test_preds_original[start_plot_idx:, 1], label='Prediksi Low', color='red', linestyle='--', marker='.')
        plt.title(f'Prediksi vs Aktual Low - Test Set ({num_points_to_plot} Titik Terakhir)')
        plt.ylabel("Harga")
        plt.legend()
        plt.grid(True)

        plt.subplot(3, 1, 3) # Plot ke-3 (Close)
        plt.plot(plot_indices, y_test_original[start_plot_idx:, 2], label='Aktual Close', color='blue', marker='.')
        plt.plot(plot_indices, test_preds_original[start_plot_idx:, 2], label='Prediksi Close', color='red', linestyle='--', marker='.')
        plt.title(f'Prediksi vs Aktual Close - Test Set ({num_points_to_plot} Titik Terakhir)')
        plt.xlabel(f"Indeks Sampel Test Set (Dimulai dari {start_plot_idx})")
        plt.ylabel("Harga")
        plt.legend()
        plt.grid(True)

        plt.tight_layout() # Menyesuaikan layout
        VISUALIZATION_PATH = os.path.join(ARTIFACTS_DIR, "test_prediction.png")
        plt.savefig(VISUALIZATION_PATH)
        logging.info(f"✅ Visualisasi prediksi Test Set disimpan di '{VISUALIZATION_PATH}'")


        # --- Simpan Metadata dan Hasil Akhir ---
        logging.info("⏳ Menyimpan metadata model dan hasil evaluasi...")
        METADATA_PATH = os.path.join(ARTIFACTS_DIR, "model_info.json")
        metadata = {
            "model_architecture": "LSTM + Attention",
            "features_used": feature_cols,
            "outputs_predicted": ["high", "low", "close"],
            "window_size": WINDOW_SIZE,
            "training_data_range": f"{df['time'].iloc[0].strftime('%Y-%m-%d')} to {df['time'].iloc[-1].strftime('%Y-%m-%d')}" if not df.empty else "N/A",
            "train_validation_size": len(X_trainval),
            "test_set_size": len(X_test),
            "test_set_metrics": test_metrics,
            "notes": "Model terbaik dipilih berdasarkan Val Loss selama TSCV dan dievaluasi pada Test Set terpisah. Kredensial MT5 dimuat dari .env."
        }
        with open(METADATA_PATH, "w") as f:
            # Gunakan indent untuk format JSON yang mudah dibaca
            json.dump(metadata, f, indent=4)
        logging.info(f"✅ Metadata dan hasil evaluasi Test Set disimpan di '{METADATA_PATH}'")

        logging.info("--- Proses Training Model Selesai ---")

    except ValueError as ve:
        logging.error(f"❌ Terjadi ValueError: {ve}")
        sys.exit(1) # Keluar jika ada error data/preprocessing/dataset
    except ConnectionError as ce:
         logging.error(f"❌ Terjadi ConnectionError: {ce}")
         sys.exit(1) # Keluar jika gagal koneksi MT5
    except Exception as e:
        # Tangani error tak terduga dan log traceback lengkap
        logging.error(f"❌ Terjadi error tak terduga selama training: {e}", exc_info=True)
        sys.exit(1) # Keluar jika ada error lain

# ============ Flask API ============
app = Flask(__name__)
# Definisikan variabel global untuk model dan scaler
loaded_model = None
loaded_scaler_X = None
loaded_scaler_y = None
# Tentukan path artefak yang akan dimuat
ARTIFACTS_DIR = "model_artifacts"
FINAL_MODEL_PATH = os.path.join(ARTIFACTS_DIR, "best_model.h5")
SCALER_X_PATH = os.path.join(ARTIFACTS_DIR, "scaler_X.save")
SCALER_Y_PATH = os.path.join(ARTIFACTS_DIR, "scaler_y.save")
# Definisikan ukuran jendela dan jumlah fitur yang diharapkan berdasarkan training
# Nilai ini harus konsisten dengan saat training
WINDOW_SIZE = 10 # Harus sama dengan window_size saat training
# Jumlah fitur X harus sama dengan len(feature_cols) saat training
# Karena kita menambah SMA dan EMA, jumlah fitur menjadi 5 + 2 = 7
EXPECTED_FEATURES_COUNT = 7

@app.before_first_request
def load_model():
    """
    Memuat model dan scaler dari file saat aplikasi Flask pertama kali dijalankan.
    Melakukan validasi dasar keberadaan file.
    """
    global loaded_model, loaded_scaler_X, loaded_scaler_y
    logging.info("⏳ Memuat model dan scaler untuk API...")
    
    # Cek apakah semua file artefak yang diperlukan ada
    if not os.path.exists(FINAL_MODEL_PATH):
         logging.error(f"❌ File model tidak ditemukan: {FINAL_MODEL_PATH}. Jalankan mode 'train' terlebih dahulu.")
         return # Tidak memuat jika file tidak ada
    if not os.path.exists(SCALER_X_PATH):
         logging.error(f"❌ File scaler_X tidak ditemukan: {SCALER_X_PATH}. Jalankan mode 'train' terlebih dahulu.")
         return
    if not os.path.exists(SCALER_Y_PATH):
         logging.error(f"❌ File scaler_y tidak ditemukan: {SCALER_Y_PATH}. Jalankan mode 'train' terlebih dahulu.")
         return

    try:
        # Muat model. Tambahkan custom objects jika ada layer non-standar (seperti MultiHeadAttention)
        custom_objects = {'MultiHeadAttention': MultiHeadAttention}
        loaded_model = tf.keras.models.load_model(FINAL_MODEL_PATH, custom_objects=custom_objects)
        
        # Muat scaler menggunakan joblib
        loaded_scaler_X = joblib.load(SCALER_X_PATH)
        loaded_scaler_y = joblib.load(SCALER_Y_PATH)
        
        logging.info("✅ Model dan scaler berhasil dimuat.")
        logging.info(f"  Model dimuat dari: {FINAL_MODEL_PATH}")
        logging.info(f"  Scaler X dimuat dari: {SCALER_X_PATH}")
        logging.info(f"  Scaler Y dimuat dari: {SCALER_Y_PATH}")

    except Exception as e:
        # Tangani error saat memuat file dan log traceback
        logging.error(f"❌ Gagal memuat model atau scaler: {e}", exc_info=True)


@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint API /predict.
    Menerima input jendela data (list of lists) melalui POST request body (JSON).
    Melakukan pra-pemrosesan, prediksi, dan mengembalikan hasil dalam skala asli.
    """
    # Cek apakah model dan scaler berhasil dimuat saat startup
    if loaded_model is None or loaded_scaler_X is None or loaded_scaler_y is None:
        logging.error("❌ Permintaan /predict ditolak karena model atau scaler belum siap.")
        # Kembalikan status 503 Service Unavailable jika API belum siap
        return jsonify({'error': 'Model atau scaler belum siap. Cek log startup API.'}), 503

    try:
        # Mengurai input JSON dari request body
        content = request.get_json()
        if content is None:
             logging.warning("Permintaan /predict tanpa body JSON.")
             return jsonify({'error': 'Request body must be JSON'}), 415 # Unsupported Media Type atau 400 Bad Request
             
        window = content.get("window") # Diharapkan berupa list of lists

        # Validasi keberadaan key 'window'
        if window is None:
            logging.warning("Permintaan /predict tanpa key 'window' di body JSON.")
            return jsonify({'error': 'Missing "window" key in JSON request body'}), 400

        # Konversi input list of lists menjadi numpy array
        try:
            arr = np.array(window)
        except Exception as e:
             logging.warning(f"Gagal mengkonversi 'window' menjadi numpy array: {e}")
             return jsonify({'error': f'Invalid format for "window". Expected list of lists: {e}'}), 400


        # Validasi shape input: harus (WINDOW_SIZE, EXPECTED_FEATURES_COUNT)
        # Ini sangat penting agar sesuai dengan input shape model
        if arr.shape != (WINDOW_SIZE, EXPECTED_FEATURES_COUNT):
            logging.warning(f"Shape input tidak sesuai. Diharapkan ({WINDOW_SIZE}, {EXPECTED_FEATURES_COUNT}), tapi mendapat {arr.shape}")
            return jsonify({'error': f'Input shape mismatch. Expected ({WINDOW_SIZE}, {EXPECTED_FEATURES_COUNT}), got {arr.shape}. Ensure you provide {WINDOW_SIZE} time steps with {EXPECTED_FEATURES_COUNT} features each, in the correct order.'}), 400

        # Validasi nilai input: cek apakah ada NaN atau Inf
        if not np.isfinite(arr).all():
            logging.warning("Input mengandung nilai NaN atau Inf.")
            return jsonify({'error': 'Input contains NaN or Inf values'}), 400
            
        # Tambahkan validasi sederhana untuk urutan kolom (opsional, tapi baik)
        # Idealnya, API menerima data mentah dan melakukan preprocessing penuh seperti fungsi preprocess
        # Tapi karena model dilatih pada data scaled dengan fitur spesifik, API harus menerima data dalam format yang sama
        # Jika API menerima data mentah (H/L/C/V saja), API perlu mengimplementasikan preprocessing & feature engineering yang SAMA PERSIS.
        # Untuk saat ini, asumsikan input 'window' sudah memiliki fitur turunan (SMA, EMA) dan dalam urutan yang benar, TAPI BELUM di-scale.
        # Ini adalah KETERBATASAN desain API saat ini. API seharusnya menerima data 'mentah' atau hanya OHLCV.

        # Scaling input menggunakan scaler X yang sudah dilatih
        # Transformasi harus dilakukan pada seluruh jendela (shape 10, 7)
        try:
            arr_scaled = loaded_scaler_X.transform(arr)
        except Exception as e:
             logging.error(f"Gagal melakukan scaling input X: {e}", exc_info=True)
             # Mungkin input tidak sesuai dengan format scaler (misal: kolom salah/kurang)
             return jsonify({'error': f'Failed to scale input data. Ensure correct features and order: {e}'}), 400


        # Menambah dimensi batch: model mengharapkan input shape (batch_size, window_size, num_features)
        arr_scaled = np.expand_dims(arr_scaled, axis=0) # Shape menjadi (1, 10, 7)

        # Melakukan prediksi menggunakan model yang sudah dimuat
        # verbose=0 agar model.predict tidak mencetak output saat API dipanggil
        pred_scaled = loaded_model.predict(arr_scaled, verbose=0)

        # Inverse transform prediksi (yang masih dalam skala scaled) ke skala asli
        # Hasil prediksi adalah (1, 3), inverse_transform akan menghasilkan (1, 3)
        try:
            pred_original = loaded_scaler_y.inverse_transform(pred_scaled)
        except Exception as e:
             logging.error(f"Gagal melakukan inverse scaling output Y: {e}", exc_info=True)
             return jsonify({'error': f'Failed to inverse scale prediction output: {e}'}), 500


        # Mengembalikan hasil prediksi
        # Ambil hasil dari batch pertama (index 0) dan konversi numpy array menjadi list
        return jsonify({'prediction': pred_original[0].tolist()})

    # Tangani error spesifik dan umum
    except ValueError as ve:
        # Menangani error terkait konversi, shape, atau validasi nilai
        logging.error(f"ValueError saat prediksi: {str(ve)}", exc_info=True)
        return jsonify({'error': f'Input data processing failed: {str(ve)}'}), 400
    except KeyError as ke:
         # Menangani jika struktur JSON tidak sesuai (misal: tidak ada key 'window')
         logging.error(f"KeyError saat prediksi (JSON structure issue): {str(ke)}", exc_info=True)
         return jsonify({'error': f'Invalid JSON structure: Missing key {str(ke)}'}), 400
    except Exception as e:
        # Menangani error umum lainnya yang tidak terduga
        logging.error(f"Error tak terduga saat prediksi: {str(e)}", exc_info=True)
        return jsonify({'error': 'Internal server error during prediction. Check API logs for details.'}), 500

# ============ Entry Point ============
if __name__ == "__main__":
    # load_dotenv() sudah dipanggil di awal file, tidak perlu di sini lagi
    
    mode = input("Ketik 'train' untuk melatih model atau 'run' untuk menjalankan API: ").strip().lower()

    if mode == "train":
        # Jalankan proses training utama
        main_training()
    elif mode == "run":
        # Jalankan aplikasi Flask API
        # load_model() akan dipanggil secara otomatis sebelum permintaan pertama
        
        # Informasi tambahan saat menjalankan API
        logging.info("💡 Menjalankan Flask API.")
        logging.info(f"💡 Memuat model dari '{FINAL_MODEL_PATH}' dan scaler dari '{SCALER_X_PATH}', '{SCALER_Y_PATH}'.")
        logging.info("💡 API akan mendengarkan di http://0.0.0.0:5000")
        logging.info("💡 Endpoint prediksi: POST ke http://0.0.0.0:5000/predict")
        logging.info(f"💡 Endpoint /predict mengharapkan JSON dengan key 'window' berisi list of lists shape ({WINDOW_SIZE}, {EXPECTED_FEATURES_COUNT}).")
        
        # Gunakan debug=True hanya untuk pengembangan. Matikan di produksi!
        app.run(host='0.0.0.0', port=5000, debug=False)
    else:
        # Pilihan mode tidak dikenali
        print("❌ Pilihan tidak dikenali. Gunakan 'train' atau 'run'.")
