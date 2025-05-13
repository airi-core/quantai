# SanClass Trading Labs

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import LSTM, Dense, Dropout, MultiHeadAttention, LayerNormalization, GlobalAveragePooling1D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from flask import Flask, request, jsonify
from dotenv import load_dotenv
import os
import logging
import joblib
import matplotlib.pyplot as plt
import json
import sys
import random # Untuk reproducibility
import time # Untuk sleep di retry

# ============ Reproducibility ============
# Set random seeds untuk hasil yang lebih konsisten
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)
os.environ['TF_DETERMINISTIC_OPS'] = '1' # Opsional, membantu reproducibility tapi bisa memperlambat

# ============ Load Env & Logging ============
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ============ MetaTrader5 Setup ============
def connect_mt5():
    login_str = os.getenv("MT5_LOGIN")
    server = os.getenv("MT5_SERVER")
    password = os.getenv("MT5_PASSWORD")

    if not login_str or not server or not password:
        logging.error("❌ Kredensial MT5 (MT5_LOGIN, MT5_SERVER, MT5_PASSWORD) tidak ditemukan.")
        sys.exit(1)

    try:
        login = int(login_str)
    except ValueError:
        logging.error(f"❌ Nilai MT5_LOGIN di .env bukan angka: '{login_str}'")
        sys.exit(1)

    max_retries = 5 # Tambah percobaan retry
    retry_delay_sec = 5 # Jeda antar percobaan (detik)

    for i in range(max_retries):
        logging.info(f"⏳ Mencoba koneksi MT5 (Percobaan {i+1}/{max_retries})...")
        if mt5.initialize(login=login, server=server, password=password):
            logging.info("✅ Berhasil login ke MetaTrader5")
            return
        last_error = mt5.last_error()
        logging.warning(f"⚠️ Gagal koneksi MT5 (Percobaan {i+1}/{max_retries}). Kode error: {last_error}")
        if i < max_retries - 1:
             # Jeda di sini setelah inisialisasi gagal, terlepas dari apakah ada exception
             time.sleep(retry_delay_sec)

    logging.error(f"❌ Gagal koneksi MT5 setelah {max_retries} percobaan.")
    raise ConnectionError(f"Gagal koneksi MT5: {last_error}")

def acquire_data(symbol, timeframe, start, end):
    connect_mt5()
    logging.info(f"⏳ Mengambil data {symbol} dari {start} hingga {end} dengan timeframe {timeframe}...")

    try:
        rates = mt5.copy_rates_range(symbol, timeframe, start, end)
        if rates is None: # copy_rates_range bisa mengembalikan None saat error
             raise Exception(f"mt5.copy_rates_range mengembalikan None. Error: {mt5.last_error()}")

    except Exception as e:
        logging.error(f"❌ Error saat memanggil copy_rates_range: {e}")
        rates = None # Pastikan rates None jika ada error
    finally:
        mt5.shutdown()
        logging.info("✅ Koneksi MT5 di-shutdown.")

    if rates is None or len(rates) == 0:
        error_msg = f"❌ Gagal mengambil data dari MT5 untuk {symbol} ({start} to {end}). Data kosong atau terjadi kesalahan."
        logging.error(error_msg)
        # Jangan raise ValueError jika dipanggil dari API yang mungkin expect data kosong
        # Ganti dengan mengembalikan DataFrame kosong dan biarkan pemanggil menangani
        # raise ValueError(error_msg) # Original behavior
        return pd.DataFrame() # Kembalikan DataFrame kosong


    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df = df.sort_values('time').reset_index(drop=True)
    logging.info(f"✅ Berhasil mengambil {len(df)} baris data.")

    return df[['time', 'open', 'high', 'low', 'close', 'tick_volume']]

# ============ Preprocessing ============
def add_technical_features(df):
    # Menambahkan fitur teknikal seperti SMA dan EMA.
    # Perhitungan ini sensitif terhadap data historis yang tersedia sebelum jendela.
    # Saat training, ini dihitung pada seluruh data historis.
    # Saat API inferensi, ini akan dihitung pada blok data yang baru saja diambil.
    # Konsistensi dalam konteks historis adalah penting untuk performa.
    df['sma_5'] = df['close'].rolling(window=5, min_periods=1).mean()
    df['ema_5'] = df['close'].ewm(span=5, adjust=False).mean()
    # Anda bisa tambahkan fitur teknikal lainnya di sini
    # Contoh: RSI membutuhkan setidaknya 14 bar data sebelumnya.
    # df['rsi_14'] = df['close'].ta.rsi(length=14) # Membutuhkan pandas_ta
    return df

def preprocess_for_training(df):
    # Membersihkan data awal dari NaN (misalnya dari data akuisisi)
    df = df.dropna().reset_index(drop=True)
    if df.empty:
         raise ValueError("DataFrame kosong setelah menghapus NaN awal.")

    # Menambahkan fitur teknikal ke SELURUH DataFrame historis
    df = add_technical_features(df)

    # Menghapus baris yang memiliki NaN setelah penambahan fitur teknikal
    # (NaN akan muncul di awal data untuk fitur seperti SMA/EMA yang butuh riwayat)
    df = df.dropna().reset_index(drop=True)
    if df.empty:
         raise ValueError("DataFrame kosong setelah menambahkan fitur teknikal dan menghapus NaN lagi.")

    # Mendefinisikan fitur yang akan digunakan (sesuaikan jika menambah fitur di add_technical_features)
    features = ['open','high','low','close','tick_volume','sma_5', 'ema_5']

    # Validasi apakah semua kolom fitur yang diharapkan ada
    if not all(f in df.columns for f in features):
         missing_features = [f for f in features if f not in df.columns]
         raise ValueError(f"Fitur yang diharapkan tidak ada di DataFrame setelah pra-pemrosesan: {missing_features}")

    logging.info(f"✅ Data diproses untuk training. Menggunakan {len(features)} fitur: {features}")

    # Melakukan scaling pada fitur input (X) menggunakan MinMaxScaler
    scaler_X = MinMaxScaler()
    df_scaled_features = scaler_X.fit_transform(df[features])
    df[features] = df_scaled_features # Update DataFrame dengan nilai scaled
    logging.info("✅ Fitur input (X) untuk training berhasil di-scale.")

    return df, scaler_X, features

def create_dataset(df, window_size, feature_cols):
    X, y = [], []
    y_time = []

    # Pastikan ada cukup data untuk membuat setidaknya satu sampel (window_size + 1 target)
    if len(df) <= window_size:
        logging.warning(f"Tidak cukup data ({len(df)} bar) untuk membuat dataset dengan window_size={window_size}. Butuh setidaknya {window_size + 1} bar.")
        return np.array([]), np.array([]), pd.DatetimeIndex([]) # Kembalikan tipe yang konsisten


    # Loop untuk membuat jendela dan target
    # Jendela berakhir di index i + window_size - 1
    # Target adalah di index i + window_size
    # Loop harus berjalan hingga index target terakhir adalah index data terakhir
    for i in range(len(df) - window_size):
        # Ambil jendela data untuk fitur X (dari i hingga i + window_size-1)
        window = df.iloc[i : i + window_size][feature_cols].values
        # Ambil data target (harga H/L/C) pada langkah waktu SETELAH jendela (index i + window_size)
        target = df.iloc[i + window_size][['high','low','close']].values
        # Ambil waktu target
        target_time = df.iloc[i + window_size]['time']

        # Cek apakah ada NaN/Inf di jendela atau target (seharusnya sudah bersih setelah preprocess, tapi sebagai pengaman)
        if not np.isfinite(window).all() or not np.isfinite(target).all():
            logging.warning(f"⚠️ NaN/Inf terdeteksi di jendela atau target pada index awal {i}. Melompati sampel ini.")
            continue

        X.append(window)
        y.append(target)
        y_time.append(target_time)

    X = np.array(X)
    y = np.array(y)
    y_time = pd.to_datetime(y_time) # Konversi list of Timestamps ke DatetimeIndex


    if len(X) == 0 and len(df) > window_size:
         logging.warning(f"Tidak ada sampel valid yang dibuat meskipun data awal cukup. Mungkin ada NaN/Inf di data setelah pra-pemrosesan yang terlewat.")

    logging.info(f"✅ Dataset dibuat. Shape X: {X.shape}, Shape y: {y.shape}")
    return X, y, y_time

# ============ Model (LSTM + Attention) ============
def build_lstm_attention_model(input_shape, output_dim):
    # Input shape: (window_size, num_features)
    inputs = Input(shape=input_shape)

    # Layer LSTM dengan regularisasi
    x = LSTM(128, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(0.0005))(inputs) # Contoh: tambah unit LSTM, turunkan L2
    x = Dropout(0.2)(x) # Tambah dropout setelah LSTM

    # Layer MultiHeadAttention
    # key_dim harus membagi dimensi terakhir input attention_input (yaitu 128 dari output LSTM)
    attention_output = MultiHeadAttention(num_heads=4, key_dim=32, dropout=0.1)(x, x) # Contoh: tambah heads, sesuaikan key_dim

    # Residual connection
    # Pastikan shape attention_output sama dengan shape x (batch, timesteps, units)
    x = tf.keras.layers.Add()([x, attention_output])

    # Layer Normalization
    x = LayerNormalization()(x)

    # GlobalAveragePooling1D untuk mereduksi dimensi waktu
    x = GlobalAveragePooling1D()(x)

    # Dense layers sebelum output
    x = Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0005))(x) # Contoh: tambah unit Dense, turunkan L2
    x = Dropout(0.3)(x) # Dropout setelah Dense
    # x = Dense(32, activation='relu')(x) # Opsional: Dense layer lain

    # Output layer: Linear activation untuk regresi harga
    outputs = Dense(output_dim)(x) # Output shape: (batch, 3) untuk High, Low, Close

    model = Model(inputs=inputs, outputs=outputs)
    logging.info("✅ Model LSTM + Attention berhasil dibangun.")
    model.summary(print_fn=lambda msg: logging.info(msg))

    # Komentar tentang model hyperparameters:
    # Angka-angka seperti 128, 4, 32, 64, 0.2, 0.3, 0.0005 adalah hyperparameters
    # yang optimalnya ditemukan melalui eksperimen (tuning). Nilai saat ini adalah contoh.

    return model

# ============ Training ============
def train_model(model, train_ds, val_ds, name):
    logging.info(f"⏳ Memulai training model fold: {name}...")
    # Compile model
    # Ganti optimizer dengan AdamW atau gunakan Adam dengan learning rate finder/scheduler
    # AdamW seringkali memberikan performa lebih baik dengan regularisasi.
    # learning_rate = tf.keras.optimizers.schedules.CosineDecay(initial_learning_rate=0.001, decay_steps=1000) # Contoh scheduler
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001) # Contoh: atur learning rate eksplisit
    # optimizer = tf.keras.optimizers.AdamW(learning_rate=0.001, weight_decay=0.004) # Contoh AdamW (butuh TF 2.11+)

    model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mse']) # Jaga metrik yang sama untuk konsistensi

    checkpoint_path = f"model_artifacts/{name}_checkpoint.h5"

    # Callbacks:
    callbacks = [
        EarlyStopping(patience=15, restore_best_weights=True, verbose=1, monitor='val_loss'), # Tambah patience
        ModelCheckpoint(checkpoint_path, save_best_only=True, verbose=1, monitor='val_loss')
        # Tambahkan ReduceLROnPlateau jika tidak pakai scheduler
        # tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)
    ]

    # Melatih model
    # Gunakan jumlah epoch yang cukup tinggi, EarlyStopping akan menghentikan secara otomatis
    history = model.fit(train_ds, validation_data=val_ds, epochs=300, callbacks=callbacks, verbose=1) # Contoh epoch lebih tinggi

    return model, history # Mengembalikan model dengan bobot terbaik dan history training

# ============ Evaluation ============
def evaluate_model(model, X_set, y_set_scaled, scaler_y, y_set_time, set_name="Test"):
    logging.info(f"⏳ Mengevaluasi model pada set {set_name} ({len(X_set)} sampel)...")

    if len(X_set) == 0:
         logging.warning(f"⚠️ Tidak ada data di set {set_name} para dievaluasi.")
         return np.array([]), np.array([]), pd.DatetimeIndex([]), {'mae': np.nan, 'rmse': np.nan, 'mape': np.nan}

    preds_scaled = model.predict(X_set, verbose=0)

    try:
        y_set_original = scaler_y.inverse_transform(y_set_scaled)
        preds_original = scaler_y.inverse_transform(preds_scaled)
    except Exception as e:
        logging.error(f"❌ Gagal inverse transform data para set {set_name}: {e}")
        return preds_scaled, y_set_scaled, y_set_time, {'mae': np.nan, 'rmse': np.nan, 'mape': np.nan}

    # Hitung metrik dalam skala asli
    mae = mean_absolute_error(y_set_original, preds_original)
    rmse = np.sqrt(mean_squared_error(y_set_original, preds_original))

    if np.all(y_set_original < 1e-6): # Cek jika nilai aktual sangat kecil
         mape = np.nan
         logging.warning(f"⚠️ Semua nilai aktual di set {set_name} mendekati nol. MAPE tidak dihitung.")
    else:
         # Hitung MAPE per output (High, Low, Close) lalu rata-ratakan
         # Menambah epsilon untuk menghindari pembagian nol
         mape_per_output = np.abs((y_set_original - preds_original) / (y_set_original + 1e-8)) * 100
         # Hilangkan Inf atau nilai sangat besar yang mungkin muncul dari pembagian nilai kecil
         mape_per_output = mape_per_output[np.isfinite(mape_per_output)]
         mape = np.mean(mape_per_output) if mape_per_output.size > 0 else np.nan # Hitung rata-rata jika ada nilai valid


    logging.info(f"📊 Hasil Evaluasi {set_name}:")
    logging.info(f"  MAE  (skala asli): {mae:.4f}")
    logging.info(f"  RMSE (skala asli): {rmse:.4f}")
    if not np.isnan(mape):
         logging.info(f"  MAPE (skala asli): {mape:.2f}%")
    else:
         logging.info("  MAPE (skala asli): N/A")

    # Potensi Peningkatan: Tambahkan metrik evaluasi spesifik trading di sini
    # Contoh: directional_accuracy = np.mean(np.sign(preds_original[:, 2] - y_set_original[:, 2]) == np.sign(y_set_original[:, 2] - y_set_original[:, 2].shift(1))) # Perlu data sebelumnya untuk hitung arah aktual


    return preds_original, y_set_original, y_set_time, {'mae': mae, 'rmse': rmse, 'mape': mape}

# ============ Main Training ============
def main_training():
    logging.info("--- Memulai Proses Training Model ---")
    try:
        # Range data training yang lebih luas atau disesuaikan relevansi pasar saat ini
        df = acquire_data('XAUUSD', mt5.TIMEFRAME_D1, pd.Timestamp(2008,1,1), pd.Timestamp(2024,12,31)) # Contoh: Diperpanjang hingga akhir 2024

        if df.empty:
             logging.error("❌ Akuisisi data gagal atau mengembalikan data kosong.")
             sys.exit(1)

        # Pra-pemrosesan data untuk training
        # Ini termasuk menambahkan fitur teknikal dan menghapus baris yang mengandung NaN
        df_processed, scaler_X, feature_cols = preprocess_for_training(df.copy())

        # Ukuran jendela dan lookback fitur teknikal (harus konsisten)
        WINDOW_SIZE = 10
        # FEATURE_LOOKBACK = 5 # Untuk SMA/EMA 5. Jika pakai indikator lain (misal RSI 14), sesuaikan.
        # Sebaiknya simpan FEATURE_LOOKBACK di metadata juga.
        # Untuk saat ini, kita hardcode lookback max di metadata berdasarkan fitur saat ini.
        max_feature_lookback = 5 # SMA 5 dan EMA 5 butuh 5 bar

        # Membuat dataset jendela (X, y) dan waktu target (y_time)
        X_all, y_raw_all, y_time_all = create_dataset(df_processed, window_size=WINDOW_SIZE, feature_cols=feature_cols)

        # Jika create_dataset mengembalikan data kosong, hentikan proses.
        if len(X_all) == 0:
            logging.error(f"❌ Gagal membuat dataset training dari data yang diproses dengan window_size={WINDOW_SIZE}. Tidak cukup sampel valid.")
            sys.exit(1)

        # Scaling target y
        scaler_y = MinMaxScaler()
        y_all = scaler_y.fit_transform(y_raw_all)
        logging.info("✅ Target output (y) berhasil di-scale.")

        # Setup direktori artefak
        ARTIFACTS_DIR = "model_artifacts"
        os.makedirs(ARTIFACTS_DIR, exist_ok=True)
        logging.info(f"✅ Direktori artefak '{ARTIFACTS_DIR}' siap.")

        # Simpan scaler dan metadata
        joblib.dump(scaler_X, os.path.join(ARTIFACTS_DIR, "scaler_X.save"))
        joblib.dump(scaler_y, os.path.join(ARTIFACTS_DIR, "scaler_y.save"))
        logging.info(f"✅ Scaler X dan Y disimpan di '{ARTIFACTS_DIR}'.")

        metadata = {
            "model_architecture": "LSTM + Attention",
            "features_used": feature_cols,
            "outputs_predicted": ["high", "low", "close"],
            "window_size": WINDOW_SIZE,
            "feature_lookback": max_feature_lookback, # Simpan lookback fitur
            "training_data_range": f"{df['time'].iloc[0].strftime('%Y-%m-%d')} to {df['time'].iloc[-1].strftime('%Y-%m-%d')}" if not df.empty else "N/A",
            "notes": "Model terbaik dipilih berdasarkan Val Loss selama TSCV dan dievaluasi pada Test Set terpisah. Kredensial MT5 dimuat dari .env.",
             # Placeholder untuk hyperparameters (bisa diisi jika tuning dilakukan)
            "hyperparameters": {
                "lstm_units": 128,
                "attention_heads": 4,
                "attention_key_dim": 32,
                "dropout_rate_lstm": 0.2,
                "dropout_rate_dense": 0.3,
                "l2_regularization": 0.0005,
                "optimizer": "Adam", # atau AdamW
                "learning_rate": 0.001,
                "batch_size": 32,
                "early_stopping_patience": 15,
                "tscv_splits": 5
            }
        }
        METADATA_PATH = os.path.join(ARTIFACTS_DIR, "model_info.json")
        with open(METADATA_PATH, "w") as f:
            json.dump(metadata, f, indent=4)
        logging.info(f"✅ Metadata model disimpan di '{METADATA_PATH}'")


        # Split data trainval vs Test
        TEST_SET_SIZE_PERCENT = 0.2
        split_idx = int(len(X_all) * (1 - TEST_SET_SIZE_PERCENT))

        # Validasi ukuran split
        if split_idx <= WINDOW_SIZE:
             logging.error(f"❌ Ukuran data trainval ({split_idx} sampel) tidak cukup untuk window_size={WINDOW_SIZE}. Butuh minimal {WINDOW_SIZE + 1} sampel.")
             sys.exit(1)

        if len(X_all) - split_idx < 1:
             logging.error(f"❌ Ukuran data test ({len(X_all) - split_idx} sampel) tidak cukup. Butuh minimal 1 sampel.")
             sys.exit(1)


        X_trainval, y_trainval = X_all[:split_idx], y_all[:split_idx]
        X_test, y_test = X_all[split_idx:], y_all[split_idx:]
        y_time_test = y_time_all[split_idx:]

        logging.info(f"✅ Data dibagi: Training+Validation ({len(X_trainval)} sampel), Test ({len(X_test)} sampel)")

        # Time Series Cross-Validation (TSCV)
        N_SPLITS_TSCV = 5
        tscv = TimeSeriesSplit(n_splits=N_SPLITS_TSCV)

        best_val_loss_in_tscv = float('inf')
        temp_model_path = os.path.join(ARTIFACTS_DIR, "temp_best_model_fold_tscv.h5")

        logging.info(f"⏳ Memulai Time Series Cross-Validation dengan {tscv.n_splits} fold pada data Training+Validation...")

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_trainval)):
            logging.info(f"--- 🔁 Fold {fold+1}/{tscv.n_splits} ---")

            X_train, y_train = X_trainval[train_idx], y_trainval[train_idx]
            X_val, y_val = X_trainval[val_idx], y_trainval[val_idx]

            if len(X_train) == 0 or len(X_val) == 0:
                 logging.warning(f"⚠️ Fold {fold+1}: Ukuran data train ({len(X_train)}) atau validasi ({len(X_val)}) kosong. Melewati fold ini.")
                 continue

            # Set seed lagi sebelum membuat model dan training per fold untuk konsistensi tambahan
            np.random.seed(42 + fold)
            tf.random.set_seed(42 + fold)
            random.seed(42 + fold)

            train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(len(X_train), seed=42).batch(32).prefetch(tf.data.AUTOTUNE)
            val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(32).prefetch(tf.data.AUTOTUNE)

            # Bangun instance model baru per fold
            fold_model = build_lstm_attention_model(X_trainval.shape[1:], y_trainval.shape[1])

            trained_fold_model, history = train_model(fold_model, train_ds, val_ds, name=f"fold_{fold+1}")

            val_loss, val_mae, val_mse = trained_fold_model.evaluate(val_ds, verbose=0)

            logging.info(f"✅ Fold {fold+1} selesai. Val Loss (scaled): {val_loss:.4f}, Val MAE (scaled): {val_mae:.4f}")

            if val_loss < best_val_loss_in_tscv:
                best_val_loss_in_tscv = val_loss
                trained_fold_model.save(temp_model_path)
                logging.info(f"⭐ Model Fold {fold+1} adalah yang terbaik sejauh ini (Val Loss scaled: {best_val_loss_in_tscv:.4f}), disimpan sementara di '{temp_model_path}'.")

        if not os.path.exists(temp_model_path):
            logging.error(f"❌ Tidak ada model terbaik yang ditemukan atau disimpan selama TSCV di '{temp_model_path}'. Proses training gagal atau semua fold dilewati.")
            sys.exit(1)

        logging.info(f"--- ⏳ Memuat model terbaik dari '{temp_model_path}' untuk evaluasi akhir pada Test Set ---")

        custom_objects = {'MultiHeadAttention': MultiHeadAttention}
        final_best_model_from_tscv = tf.keras.models.load_model(temp_model_path, custom_objects=custom_objects)

        # Lakukan evaluasi pada test set akhir (dalam skala asli)
        test_preds_original, y_test_original, y_time_test_eval, test_metrics = evaluate_model(
            final_best_model_from_tscv, X_test, y_test, scaler_y, y_time_test, set_name="Test Set Akhir"
        )

        # Simpan model terbaik final
        FINAL_MODEL_PATH = os.path.join(ARTIFACTS_DIR, "best_model.h5")
        final_best_model_from_tscv.save(FINAL_MODEL_PATH)
        logging.info(f"✅ Model terbaik (berdasarkan validasi TSCV) disimpan di '{FINAL_MODEL_PATH}'.")

        # Hapus file sementara dan checkpoint fold jika diinginkan
        try:
            for fold in range(N_SPLITS_TSCV):
                 fold_checkpoint_path = os.path.join(ARTIFACTS_DIR, f"fold_{fold+1}_checkpoint.h5")
                 if os.path.exists(fold_checkpoint_path):
                      os.remove(fold_checkpoint_path)
                      # logging.info(f"✅ Checkpoint fold {fold+1} dihapus: '{fold_checkpoint_path}'") # Kurangi log verbosity
            if os.path.exists(temp_model_path):
                 os.remove(temp_model_path)
                 logging.info(f"✅ File model sementara dihapus: '{temp_model_path}'")

        except Exception as e:
             logging.warning(f"⚠️ Gagal menghapus file model sementara: {e}")

        logging.info("⏳ Membuat visualisasi prediksi pada Test Set...")
        plt.figure(figsize=(15, 10)) # Ukuran plot lebih besar


        num_points_to_plot = min(len(y_test_original), 200) # Plot hingga 200 titik
        start_plot_idx = len(y_test_original) - num_points_to_plot

        plot_times = y_time_test_eval[start_plot_idx:]
        plot_actual = y_test_original[start_plot_idx:]
        plot_preds = test_preds_original[start_plot_idx:]


        plt.subplot(3, 1, 1)
        plt.plot(plot_times, plot_actual[:, 0], label='Aktual High', color='blue', marker='.', markersize=4, linestyle='-')
        plt.plot(plot_times, plot_preds[:, 0], label='Prediksi High', color='red', linestyle='--', marker='.', markersize=4)
        plt.title(f'Prediksi vs Aktual High - Test Set ({num_points_to_plot} Titik Terakhir)')
        plt.ylabel("Harga")
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45, ha='right') # Atur rotasi dan alignment label
        plt.tight_layout() # Menyesuaikan layout agar tidak tumpang tindih


        plt.subplot(3, 1, 2)
        plt.plot(plot_times, plot_actual[:, 1], label='Aktual Low', color='blue', marker='.', markersize=4, linestyle='-')
        plt.plot(plot_times, plot_preds[:, 1], label='Prediksi Low', color='red', linestyle='--', marker='.', markersize=4)
        plt.title(f'Prediksi vs Aktual Low - Test Set ({num_points_to_plot} Titik Terakhir)')
        plt.ylabel("Harga")
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()


        plt.subplot(3, 1, 3)
        plt.plot(plot_times, plot_actual[:, 2], label='Aktual Close', color='blue', marker='.', markersize=4, linestyle='-')
        plt.plot(plot_times, plot_preds[:, 2], label='Prediksi Close', color='red', linestyle='--', marker='.', markersize=4)
        plt.title(f'Prediksi vs Aktual Close - Test Set ({num_points_to_plot} Titik Terakhir)')
        plt.xlabel("Waktu")
        plt.ylabel("Harga")
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        VISUALIZATION_PATH = os.path.join(ARTIFACTS_DIR, "test_prediction.png")
        plt.savefig(VISUALIZATION_PATH)
        logging.info(f"✅ Visualisasi prediksi Test Set disimpan di '{VISUALIZATION_PATH}'")

        # Update metadata dengan hasil evaluasi test set akhir
        metadata["test_set_size"] = len(X_test)
        metadata["test_set_metrics"] = test_metrics
        METADATA_PATH = os.path.join(ARTIFACTS_DIR, "model_info.json")
        with open(METADATA_PATH, "w") as f:
            json.dump(metadata, f, indent=4)
        logging.info(f"✅ Metadata model (termasuk hasil evaluasi test set) disimpan di '{METADATA_PATH}'")


        logging.info("--- Proses Training Model Selesai ---")

    except ValueError as ve:
        logging.error(f"❌ Terjadi ValueError selama training: {ve}")
        sys.exit(1)
    except ConnectionError as ce:
         logging.error(f"❌ Terjadi ConnectionError selama training: {ce}")
         sys.exit(1)
    except Exception as e:
        logging.error(f"❌ Terjadi error tak terduga selama training: {e}", exc_info=True)
        sys.exit(1)

# ============ Flask API ============
app = Flask(__name__)

loaded_model = None
loaded_scaler_X = None
loaded_scaler_y = None
loaded_metadata = None

ARTIFACTS_DIR = "model_artifacts"
FINAL_MODEL_PATH = os.path.join(ARTIFACTS_DIR, "best_model.h5")
SCALER_X_PATH = os.path.join(ARTIFACTS_DIR, "scaler_X.save")
SCALER_Y_PATH = os.path.join(ARTIFACTS_DIR, "scaler_y.save")
METADATA_PATH = os.path.join(ARTIFACTS_DIR, "model_info.json")

# Konfigurasi API dimuat dari metadata
API_WINDOW_SIZE = None
API_FEATURES_USED = None
API_OUTPUTS_PREDICTED = None
API_FEATURE_LOOKBACK = None # Tambah lookback fitur untuk API

@app.before_first_request
def load_artifacts():
    global loaded_model, loaded_scaler_X, loaded_scaler_y, loaded_metadata
    global API_WINDOW_SIZE, API_FEATURES_USED, API_OUTPUTS_PREDICTED, API_FEATURE_LOOKBACK

    logging.info("⏳ Memuat artefak (model, scaler, metadata) untuk API...")

    artefact_files = {
        "model": FINAL_MODEL_PATH,
        "scaler_x": SCALER_X_PATH,
        "scaler_y": SCALER_Y_PATH,
        "metadata": METADATA_PATH
    }

    for name, path in artefact_files.items():
        if not os.path.exists(path):
            logging.error(f"❌ File artefak '{name}' tidak ditemukan: {path}. Jalankan mode 'train' terlebih dahulu.")
            sys.exit(1)

    try:
        with open(METADATA_PATH, 'r') as f:
            loaded_metadata = json.load(f)

        API_WINDOW_SIZE = loaded_metadata.get("window_size")
        API_FEATURES_USED = loaded_metadata.get("features_used")
        API_OUTPUTS_PREDICTED = loaded_metadata.get("outputs_predicted")
        API_FEATURE_LOOKBACK = loaded_metadata.get("feature_lookback", 0) # Ambil lookback, default 0 jika tidak ada

        if API_WINDOW_SIZE is None or API_FEATURES_USED is None or API_OUTPUTS_PREDICTED is None:
            logging.error(f"❌ Metadata ({METADATA_PATH}) tidak lengkap. Pastikan berisi 'window_size', 'features_used', 'outputs_predicted'.")
            sys.exit(1)

        custom_objects = {'MultiHeadAttention': MultiHeadAttention}
        loaded_model = tf.keras.models.load_model(FINAL_MODEL_PATH, custom_objects=custom_objects)

        loaded_scaler_X = joblib.load(SCALER_X_PATH)
        loaded_scaler_y = joblib.load(SCALER_Y_PATH)

        logging.info("✅ Artefak (model, scaler, metadata) berhasil dimuat.")
        logging.info(f"  Model dimuat dari: {FINAL_MODEL_PATH}")
        logging.info(f"  Scaler X dimuat dari: {SCALER_X_PATH}")
        logging.info(f"  Scaler Y dimuat dari: {SCALER_Y_PATH}")
        logging.info(f"  Metadata dimuat dari: {METADATA_PATH}")
        logging.info(f"  Konfigurasi dari metadata: Window Size = {API_WINDOW_SIZE}, Fitur Digunakan = {API_FEATURES_USED}, Feature Lookback = {API_FEATURE_LOOKBACK}")


    except Exception as e:
        logging.error(f"❌ Gagal memuat artefak (model, scaler, atau metadata): {e}", exc_info=True)
        sys.exit(1)

@app.route('/predict', methods=['POST'])
def predict():
    if loaded_model is None or loaded_scaler_X is None or loaded_scaler_y is None or loaded_metadata is None:
        logging.error("❌ Permintaan /predict ditolak karena artefak (model/scaler/metadata) belum siap.")
        return jsonify({'error': 'Artefak model belum siap. Cek log startup API. Jalankan mode "train" jika belum.'}), 503

    try:
        content = request.get_json()
        if content is None:
             logging.warning("Permintaan /predict tanpa body JSON.")
             return jsonify({'error': 'Request body must be JSON'}), 415

        # API sekarang mengharapkan symbol, timeframe, dan end_time
        symbol = content.get("symbol")
        timeframe_str = content.get("timeframe")
        end_time_str = content.get("end_time") # Format ISO 8601 atau timestamp string

        if not all([symbol, timeframe_str, end_time_str]):
            missing = [k for k, v in {'symbol': symbol, 'timeframe': timeframe_str, 'end_time': end_time_str}.items() if not v]
            logging.warning(f"Permintaan /predict tanpa parameter wajib: {missing}")
            return jsonify({'error': f'Missing required parameters: {missing}. Expected symbol, timeframe, end_time.'}), 400

        # Konversi timeframe string ke nilai MT5
        # Anda perlu mapping string timeframe ke nilai mt5.TIMEFRAME_...
        timeframe_map = {
            'M1': mt5.TIMEFRAME_M1, 'M5': mt5.TIMEFRAME_M5, 'M15': mt5.TIMEFRAME_M15,
            'M30': mt5.TIMEFRAME_M30, 'H1': mt5.TIMEFRAME_H1, 'H4': mt5.TIMEFRAME_H4,
            'D1': mt5.TIMEFRAME_D1, 'W1': mt5.TIMEFRAME_W1, 'MN1': mt5.TIMEFRAME_MN1
            # Tambahkan timeframe lain jika diperlukan
        }
        timeframe = timeframe_map.get(timeframe_str.upper())

        if timeframe is None:
            logging.warning(f"Timeframe tidak valid: {timeframe_str}. Timeframe yang didukung: {list(timeframe_map.keys())}")
            return jsonify({'error': f'Invalid timeframe: {timeframe_str}. Supported timeframes: {list(timeframe_map.keys())}.'}), 400

        # Konversi end_time string ke objek datetime
        try:
            # Asumsikan format ISO 8601, atau coba parsing umum
            end_time = pd.to_datetime(end_time_str)
        except Exception as e:
            logging.warning(f"Format end_time tidak valid: {end_time_str}. Error: {e}")
            return jsonify({'error': f'Invalid end_time format: {end_time_str}. Expected parseable date/time string (e.g., ISO 8601).'}), 400

        # --- Akuisisi dan Pemrosesan Data untuk Prediksi ---
        # Tentukan jumlah bar yang perlu diambil: window_size + lookback fitur maksimal
        # Ini untuk memastikan ada cukup riwayat untuk menghitung fitur di awal jendela
        num_bars_needed = API_WINDOW_SIZE + API_FEATURE_LOOKBACK
        logging.info(f"⏳ API: Mengambil {num_bars_needed} bar data historis hingga {end_time} untuk {symbol} ({timeframe_str})...")

        # Hitung waktu mulai berdasarkan end_time dan jumlah bar yang dibutuhkan
        # Cara sederhana: ambil lebih banyak bar dari yang dibutuhkan, nanti dipotong
        # atau estimasi waktu start (kurang akurat karena gap/libur)
        # Cara yang lebih andal: MT5 copy_rates_from_pos
        # Kita gunakan copy_rates_range ke belakang dari end_time untuk N bar
        # MT5 API copy_rates_range(symbol, timeframe, date_from, date_to)
        # copy_rates_from_pos(symbol, timeframe, pos, count) - Ini lebih cocok!
        # Mari kita modifikasi acquire_data atau buat fungsi baru yang pakai copy_rates_from_pos

        # Karena acquire_data pakai copy_rates_range, kita estimasi start_time
        # Ini kurang presisi karena jarak antar bar tidak selalu sama (gap weekend dll)
        # Sebuah solusi yang lebih baik adalah memodifikasi acquire_data untuk mengambil N bar terakhir HINGGA end_time
        # dengan copy_rates_from_pos
        # Sebagai workaround cepat, kita bisa ambil lebih banyak data dari yang dibutuhkan
        # atau modifikasi acquire_data untuk API:

        # --- Modifikasi acquire_data untuk API ---
        def acquire_data_for_api(symbol, timeframe, end_time, count):
             connect_mt5()
             logging.info(f"⏳ API: Mengambil {count} bar data historis untuk {symbol} ({timeframe}) hingga {end_time}...")
             try:
                 # Cari posisi bar yang paling dekat dengan end_time
                 # Ini memerlukan iterasi atau cara lain untuk menemukan 'pos' yang tepat
                 # copy_rates_from_pos membutuhkan 'pos' (index dari awal sejarah)
                 # Ini rumit. Mari kita gunakan copy_rates_range saja untuk sementara, tapi sadari limitasinya.
                 # Ambil data dari tanggal yang cukup jauh ke belakang
                 # Estimasi kasar start time: end_time - timedelta(days = count * avg_days_per_bar)
                 # Untuk D1, avg_days_per_bar = 1 (kurang lebih, abaikan weekend)
                 # start_time_estimate = end_time - pd.Timedelta(days=count * 1.5) # Ambil sedikit lebih banyak
                 # rates = mt5.copy_rates_range(symbol, timeframe, start_time_estimate, end_time)

                 # Alternatif: Gunakan copy_rates_from(symbol, timeframe, start_pos, count)
                 # Tapi menentukan start_pos untuk mendapatkan N bar *hingga* end_time itu rumit.
                 # Solusi paling sederhana dengan fungsi acquire_data yang ada: ambil range yang luas.
                 # Atau, kita perlu fungsi baru di acquire_data_for_api yang secara akurat mengambil N bar terakhir HINGGA end_time.
                 # Implementasi akurat butuh menemukan 'pos' untuk end_time. Ini bisa lambat.
                 # Pilihan termudah saat ini: terima list of bars dari client (versi sebelumnya)
                 # Pilihan kedua: ambil N bar dari "akhir" sejarah, tapi itu bukan "hingga end_time spesifik"
                 # Pilihan ketiga (yang coba diimplementasikan): ambil range yang cukup luas, lalu filter/potong

                 # Implementasi baru: Ambil lebih banyak bar dari yang dibutuhkan, lalu filter
                 # Kita butuh setidaknya num_bars_needed. Ambil 2 * num_bars_needed bar terakhir dari end_time.
                 # Ini menggunakan copy_rates_from, yang lebih akurat untuk mengambil N bar terakhir.
                 rates = mt5.copy_rates_from(timeframe, end_time, num_bars_needed * 2) # Ambil 2x lipat bar

                 if rates is None:
                      raise Exception(f"mt5.copy_rates_from mengembalikan None. Error: {mt5.last_error()}")

             except Exception as e:
                 logging.error(f"❌ API: Error saat memanggil MT5 API: {e}")
                 rates = None
             finally:
                 mt5.shutdown()
                 logging.info("✅ API: Koneksi MT5 di-shutdown.")

             if rates is None or len(rates) == 0:
                 error_msg = f"❌ API: Gagal mengambil data dari MT5 untuk {symbol} ({count} bar hingga {end_time}). Data kosong atau terjadi kesalahan."
                 logging.error(error_msg)
                 return pd.DataFrame() # Kembalikan DataFrame kosong

             df_api = pd.DataFrame(rates)
             df_api['time'] = pd.to_datetime(df_api['time'], unit='s')
             df_api = df_api.sort_values('time').reset_index(drop=True)

             # Filter data yang diambil agar hanya yang <= end_time
             df_api = df_api[df_api['time'] <= end_time].reset_index(drop=True)

             # Ambil jumlah bar yang dibutuhkan (num_bars_needed) dari data terbaru
             if len(df_api) < num_bars_needed:
                  logging.warning(f"⚠️ API: Hanya berhasil mengambil {len(df_api)} bar data hingga {end_time}, kurang dari yang dibutuhkan ({num_bars_needed}).")
                  # Lanjutkan dengan data yang ada, tapi mungkin tidak cukup untuk fitur/window
                  df_needed = df_api # Gunakan semua data yang ada
             else:
                  df_needed = df_api.tail(num_bars_needed).reset_index(drop=True) # Ambil N bar terbaru

             logging.info(f"✅ API: Berhasil memproses {len(df_needed)} bar data untuk pemrosesan lebih lanjut.")
             return df_needed[['time', 'open', 'high', 'low', 'close', 'tick_volume']]
        # --- End of acquire_data_for_api ---


        # Panggil fungsi akuisisi data spesifik API
        df_history = acquire_data_for_api(symbol, timeframe, end_time, num_bars_needed)

        if df_history.empty or len(df_history) < API_WINDOW_SIZE + API_FEATURE_LOOKBACK:
             # Perbaiki cek: butuh cukup data untuk menghitung fitur DAN membentuk window
             logging.error(f"❌ API: Data historis yang diambil tidak cukup ({len(df_history)} bar) untuk menghitung fitur ({API_FEATURE_LOOKBACK} bar lookback) dan membentuk jendela ({API_WINDOW_SIZE} bar).")
             return jsonify({'error': 'Not enough historical data available from MT5 for the requested time to calculate features and form the prediction window.'}), 400


        # Lakukan preprocessing dan feature engineering pada data historis yang diambil
        df_processed_api = add_technical_features(df_history)

        # Tangani NaN yang dihasilkan dari fitur teknikal (di awal df_processed_api)
        # Mirip dengan training, kita DROP baris yang mengandung NaN setelah penambahan fitur
        # Ini lebih konsisten dengan training daripada fillna(0).
        df_processed_api = df_processed_api.dropna().reset_index(drop=True)

        if df_processed_api.empty:
             logging.error("❌ API: DataFrame kosong setelah menambahkan fitur teknikal dan menghapus NaN.")
             return jsonify({'error': 'Error processing historical data: DataFrame became empty after feature engineering.'}), 500


        # Ambil jendela input untuk model (N bar TERAKHIR dari data yang sudah diproses dan bersih)
        if len(df_processed_api) < API_WINDOW_SIZE:
             logging.error(f"❌ API: Data setelah feature engineering dan dropna ({len(df_processed_api)} bar) kurang dari window size ({API_WINDOW_SIZE}). Tidak bisa membentuk jendela input.")
             return jsonify({'error': 'Not enough valid data after feature engineering to form the prediction window. Adjust data range or lookback.'}), 500

        df_window_final = df_processed_api.tail(API_WINDOW_SIZE).reset_index(drop=True)


        # Seleksi fitur yang digunakan saat training, pastikan urutannya benar
        # Gunakan API_FEATURES_USED dari metadata
        try:
            # Pastikan semua API_FEATURES_USED ada di df_window_final
            if not all(f in df_window_final.columns for f in API_FEATURES_USED):
                 missing_api_features = [f for f in API_FEATURES_USED if f not in df_window_final.columns]
                 logging.error(f"❌ API: Fitur yang diharapkan ({API_FEATURES_USED}) tidak lengkap di jendela input final. Missing: {missing_api_features}")
                 return jsonify({'error': 'Internal error: Final input window is missing required features.'}), 500


            arr_to_scale = df_window_final[API_FEATURES_USED].values

        except Exception as e:
             logging.error(f"❌ API: Gagal mengambil atau memvalidasi kolom fitur dari jendela input final: {e}", exc_info=True)
             return jsonify({'error': f'Error preparing data for scaling: {e}'}), 500

        # Validasi shape array sebelum scaling dan prediksi
        if arr_to_scale.shape != (API_WINDOW_SIZE, len(API_FEATURES_USED)):
             logging.error(f"❌ API: Shape array fitur input final tidak sesuai. Diharapkan ({API_WINDOW_SIZE}, {len(API_FEATURES_USED)}), mendapat {arr_to_scale.shape}")
             return jsonify({'error': 'Internal error: Final processed input shape mismatch.'}), 500

        # 6. Scaling input
        try:
            arr_scaled = loaded_scaler_X.transform(arr_to_scale)
        except Exception as e:
             logging.error(f"❌ API: Gagal melakukan scaling input X: {e}", exc_info=True)
             return jsonify({'error': f'Failed to scale input data using loaded scaler: {e}'}), 500

        # 7. Menambah dimensi batch
        arr_scaled = np.expand_dims(arr_scaled, axis=0)

        # 8. Melakukan prediksi
        pred_scaled = loaded_model.predict(arr_scaled, verbose=0)

        # 9. Inverse transform prediksi
        try:
            pred_original = loaded_scaler_y.inverse_transform(pred_scaled)
        except Exception as e:
             logging.error(f"❌ API: Gagal melakukan inverse scaling output Y: {e}", exc_info=True)
             return jsonify({'error': f'Failed to inverse scale prediction output: {e}'}), 500

        # 10. Mengembalikan hasil
        predicted_values = pred_original[0].tolist()
        output_dict = dict(zip(API_OUTPUTS_PREDICTED, predicted_values))

        logging.info(f"✅ API: Prediksi berhasil untuk {symbol} {timeframe_str} hingga {end_time}.")
        return jsonify({'prediction': output_dict})

    except ValueError as ve:
        logging.error(f"❌ ValueError saat prediksi API: {str(ve)}", exc_info=True)
        return jsonify({'error': f'Input data processing failed: {str(ve)}'}), 400
    except KeyError as ke:
         logging.error(f"❌ KeyError saat prediksi API (JSON structure issue): {str(ke)}", exc_info=True)
         return jsonify({'error': f'Invalid JSON structure: Missing key {str(ke)}'}), 400
    except ConnectionError as ce:
         logging.error(f"❌ ConnectionError saat prediksi API: {str(ce)}", exc_info=True)
         return jsonify({'error': f'Failed to connect to MetaTrader5: {str(ce)}'}), 500 # Gunakan 500 untuk error server eksternal
    except Exception as e:
        logging.error(f"❌ Error tak terduga saat prediksi API: {str(e)}", exc_info=True)
        return jsonify({'error': 'Internal server error during prediction. Check API logs for details.'}), 500

# ============ Entry Point ============
if __name__ == "__main__":
    # load_dotenv() sudah dipanggil di awal
    # Set seed di sini juga untuk konsistensi main execution path
    np.random.seed(42)
    tf.random.set_seed(42)
    random.seed(42)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'

    mode = input("Ketik 'train' untuk melatih model atau 'run' untuk menjalankan API: ").strip().lower()

    if mode == "train":
        main_training()
    elif mode == "run":
        logging.info("💡 Menjalankan Flask API.")
        logging.info(f"💡 Memuat artefak dari '{ARTIFACTS_DIR}'.")
        logging.info("💡 API akan mendengarkan di http://0.0.0.0:5000")
        logging.info("💡 Endpoint prediksi: POST ke http://0.0.0.0:5000/predict")
        # Pesan input API diperbarui
        logging.info("💡 Endpoint /predict mengharapkan JSON dengan key 'symbol', 'timeframe', dan 'end_time' (format parseable date/time).")
        logging.info("💡 API akan mengambil data historis yang dibutuhkan dari MT5, melakukan feature engineering, dan memprediksi.")

        app.run(host='0.0.0.0', port=5000, debug=False)
    else:
        print("❌ Pilihan tidak dikenali. Gunakan 'train' atau 'run'.")
