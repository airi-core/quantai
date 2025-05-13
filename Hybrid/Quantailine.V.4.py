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

    max_retries = 3
    for i in range(max_retries):
        if mt5.initialize(login=login, server=server, password=password):
            logging.info("✅ Berhasil login ke MetaTrader5")
            return
        last_error = mt5.last_error()
        logging.warning(f"⚠️ Gagal koneksi MT5 (Percobaan {i+1}/{max_retries}). Kode error: {last_error}")
        if i < max_retries - 1:
             import time
             time.sleep(5)

    logging.error(f"❌ Gagal koneksi MT5 setelah {max_retries} percobaan.")
    raise ConnectionError(f"Gagal koneksi MT5: {last_error}")

def acquire_data(symbol, timeframe, start, end):
    connect_mt5()
    logging.info(f"⏳ Mengambil data {symbol} dari {start} hingga {end} dengan timeframe {timeframe}...")

    try:
        rates = mt5.copy_rates_range(symbol, timeframe, start, end)
    except Exception as e:
        logging.error(f"❌ Error saat memanggil copy_rates_range: {e}")
        rates = None
    finally:
        mt5.shutdown()
        logging.info("✅ Koneksi MT5 di-shutdown.")

    if rates is None or len(rates) == 0:
        error_msg = f"❌ Gagal mengambil data dari MT5 untuk {symbol} ({start} to {end}). Data kosong atau terjadi kesalahan."
        logging.error(error_msg)
        raise ValueError(error_msg)

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df = df.sort_values('time').reset_index(drop=True)
    logging.info(f"✅ Berhasil mengambil {len(df)} baris data.")

    return df[['time', 'open', 'high', 'low', 'close', 'tick_volume']]

# ============ Preprocessing ============
def add_technical_features(df):
    # Fungsi ini asumsikan menerima DataFrame dengan kolom OHLCV dan tick_volume, serta kolom 'time'
    # Perhitungan SMA dan EMA akan dilakukan pada DataFrame yang diterima.
    # Jika DataFrame ini adalah jendela data, NaN akan muncul di awal.
    df['sma_5'] = df['close'].rolling(window=5, min_periods=1).mean()
    df['ema_5'] = df['close'].ewm(span=5, adjust=False).mean()
    return df # Jangan dropna di sini, penanganan NaN akan dilakukan setelahnya jika perlu

def preprocess_for_training(df):
    df = df.dropna().reset_index(drop=True)
    if df.empty:
         raise ValueError("DataFrame kosong setelah menghapus NaN awal.")

    # Tambahkan fitur teknikal ke seluruh DataFrame historis
    df = add_technical_features(df)

    # Di sini, setelah menambahkan fitur ke data historis, NaN di awal akan dihapus
    # oleh create_dataset karena loop dimulai setelah window_size.
    # Tapi untuk memastikan tidak ada NaN dari fitur teknikal, kita dropna lagi di sini.
    # *CATATAN:* Ini menyebabkan perbedaan logika dengan API jika API hanya memproses jendela pendek.
    df = df.dropna().reset_index(drop=True)
    if df.empty:
         raise ValueError("DataFrame kosong setelah menambahkan fitur teknikal dan menghapus NaN lagi.")

    features = ['open','high','low','close','tick_volume','sma_5', 'ema_5']

    if not all(f in df.columns for f in features):
         missing_features = [f for f in features if f not in df.columns]
         raise ValueError(f"Fitur yang diharapkan tidak ada di DataFrame setelah pra-pemrosesan: {missing_features}")

    logging.info(f"✅ Data diproses. Menggunakan {len(features)} fitur: {features}")

    scaler_X = MinMaxScaler()
    # Lakukan scaling pada kolom fitur
    df_scaled_features = scaler_X.fit_transform(df[features])
    df[features] = df_scaled_features # Update DataFrame dengan nilai scaled
    logging.info("✅ Fitur input (X) berhasil di-scale.")

    return df, scaler_X, features

def create_dataset(df, window_size, feature_cols):
    X, y = [], []
    y_time = []

    # Pastikan ada cukup data untuk membuat setidaknya satu sampel
    if len(df) <= window_size:
        logging.warning(f"Tidak cukup data ({len(df)} bar) untuk membuat dataset dengan window_size={window_size}. Butuh setidaknya {window_size + 1} bar.")
        return np.array([]), np.array([]), np.array([])


    for i in range(len(df) - window_size):
        # Ambil jendela data untuk fitur X
        window = df.iloc[i : i + window_size][feature_cols].values
        # Ambil data target (harga H/L/C) pada langkah waktu SETELAH jendela
        target = df.iloc[i + window_size][['high','low','close']].values
        # Ambil waktu target
        target_time = df.iloc[i + window_size]['time']

        # Pastikan jendela tidak mengandung NaN (seharusnya sudah ditangani oleh dropna di preprocess_for_training)
        if not np.isfinite(window).all():
            logging.warning(f"⚠️ NaN/Inf terdeteksi di jendela data pada index {i}. Melompati jendela ini.")
            continue # Lompat jika ada NaN/Inf

        X.append(window)
        y.append(target)
        y_time.append(target_time)

    X = np.array(X)
    y = np.array(y)
    y_time = pd.to_datetime(y_time) # Konversi list of Timestamps ke DatetimeIndex atau Array


    if len(X) == 0 and len(df) > window_size:
         logging.warning(f"Tidak ada sampel valid yang dibuat setelah memeriksa NaN/Inf dalam jendela. Mungkin ada NaN/Inf di data setelah pra-pemrosesan.")
         # Jika data awal cukup tapi tidak ada sampel dibuat, mungkin ada error dalam pra-pemrosesan.
         # Pertimbangkan raise ValueError di sini tergantung seberapa ketat validasinya.


    logging.info(f"✅ Dataset dibuat. Shape X: {X.shape}, Shape y: {y.shape}")
    return X, y, y_time

# ============ Model (LSTM + Attention) ============
def build_lstm_attention_model(input_shape, output_dim):
    inputs = Input(shape=input_shape)

    x = LSTM(64, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(0.001))(inputs)

    attention_output = MultiHeadAttention(num_heads=2, key_dim=32, dropout=0.1)(x, x)

    x = tf.keras.layers.Add()([x, attention_output])

    x = LayerNormalization()(x)

    x = GlobalAveragePooling1D()(x)

    x = Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = Dropout(0.3)(x)

    outputs = Dense(output_dim)(x)

    model = Model(inputs=inputs, outputs=outputs)
    logging.info("✅ Model LSTM + Attention berhasil dibangun.")
    model.summary(print_fn=lambda msg: logging.info(msg))

    return model

# ============ Training ============
def train_model(model, train_ds, val_ds, name):
    logging.info(f"⏳ Memulai training model fold: {name}...")
    model.compile(optimizer='adam', loss='mse', metrics=['mae', 'mse'])

    checkpoint_path = f"model_artifacts/{name}_checkpoint.h5"

    callbacks = [
        EarlyStopping(patience=10, restore_best_weights=True, verbose=1, monitor='val_loss'),
        ModelCheckpoint(checkpoint_path, save_best_only=True, verbose=1, monitor='val_loss')
    ]

    history = model.fit(train_ds, validation_data=val_ds, epochs=200, callbacks=callbacks, verbose=1)

    return model, history

# ============ Evaluation ============
def evaluate_model(model, X_set, y_set_scaled, scaler_y, y_set_time, set_name="Test"):
    logging.info(f"⏳ Mengevaluasi model pada set {set_name} ({len(X_set)} sampel)...")

    if len(X_set) == 0:
         logging.warning(f"⚠️ Tidak ada data di set {set_name} para dievaluasi.")
         return np.array([]), np.array([]), np.array([]), {'mae': np.nan, 'rmse': np.nan, 'mape': np.nan}

    preds_scaled = model.predict(X_set, verbose=0)

    try:
        y_set_original = scaler_y.inverse_transform(y_set_scaled)
        preds_original = scaler_y.inverse_transform(preds_scaled)
    except Exception as e:
        logging.error(f"❌ Gagal inverse transform data para set {set_name}: {e}")
        # Kembalikan scaled dan waktu meskipun inverse transform gagal
        return preds_scaled, y_set_scaled, y_set_time, {'mae': np.nan, 'rmse': np.nan, 'mape': np.nan}

    mae = mean_absolute_error(y_set_original, preds_original)
    rmse = np.sqrt(mean_squared_error(y_set_original, preds_original))

    if np.all(y_set_original < 1e-6):
         mape = np.nan
         logging.warning(f"⚠️ Semua nilai aktual di set {set_name} mendekati nol. MAPE tidak dihitung.")
    else:
         # Hitung MAPE hanya untuk kolom harga (High, Low, Close)
         mape_per_output = np.abs((y_set_original - preds_original) / (y_set_original + 1e-8)) * 100
         mape = np.mean(mape_per_output) # MAPE rata-rata dari 3 output

    logging.info(f"📊 Hasil Evaluasi {set_name}:")
    logging.info(f"  MAE  (skala asli): {mae:.4f}")
    logging.info(f"  RMSE (skala asli): {rmse:.4f}")
    if not np.isnan(mape):
         logging.info(f"  MAPE (skala asli): {mape:.2f}%")
    else:
         logging.info("  MAPE (skala asli): N/A")


    return preds_original, y_set_original, y_set_time, {'mae': mae, 'rmse': rmse, 'mape': mape}

# ============ Main Training ============
def main_training():
    logging.info("--- Memulai Proses Training Model ---")
    try:
        df = acquire_data('XAUUSD', mt5.TIMEFRAME_D1, pd.Timestamp(2008,1,1), pd.Timestamp(2023,12,31))

        # Gunakan copy() agar df asli tidak terpengaruh preprocessing (jika nanti df asli digunakan lagi)
        df_processed, scaler_X, feature_cols = preprocess_for_training(df.copy())

        WINDOW_SIZE = 10
        X_all, y_raw_all, y_time_all = create_dataset(df_processed, window_size=WINDOW_SIZE, feature_cols=feature_cols)

        # Jika create_dataset mengembalikan data kosong, hentikan proses.
        if len(X_all) == 0:
            logging.error("❌ Gagal membuat dataset dari data yang diproses. Tidak cukup sampel valid.")
            sys.exit(1)

        scaler_y = MinMaxScaler()
        y_all = scaler_y.fit_transform(y_raw_all)
        logging.info("✅ Target output (y) berhasil di-scale.")

        ARTIFACTS_DIR = "model_artifacts"
        os.makedirs(ARTIFACTS_DIR, exist_ok=True)
        logging.info(f"✅ Direktori artefak '{ARTIFACTS_DIR}' siap.")

        joblib.dump(scaler_X, os.path.join(ARTIFACTS_DIR, "scaler_X.save"))
        joblib.dump(scaler_y, os.path.join(ARTIFACTS_DIR, "scaler_y.save"))
        logging.info(f"✅ Scaler X dan Y disimpan di '{ARTIFACTS_DIR}'.")

        # Simpan metadata model, termasuk nama fitur dan window size
        metadata = {
            "model_architecture": "LSTM + Attention",
            "features_used": feature_cols, # Simpan nama fitur
            "outputs_predicted": ["high", "low", "close"], # Simpan nama output
            "window_size": WINDOW_SIZE, # Simpan window size
            "training_data_range": f"{df['time'].iloc[0].strftime('%Y-%m-%d')} to {df['time'].iloc[-1].strftime('%Y-%m-%d')}" if not df.empty else "N/A",
            "notes": "Model terbaik dipilih berdasarkan Val Loss selama TSCV dan dievaluasi pada Test Set terpisah. Kredensial MT5 dimuat dari .env."
        }
        METADATA_PATH = os.path.join(ARTIFACTS_DIR, "model_info.json")
        with open(METADATA_PATH, "w") as f:
            json.dump(metadata, f, indent=4)
        logging.info(f"✅ Metadata model disimpan di '{METADATA_PATH}'")


        TEST_SET_SIZE_PERCENT = 0.2
        split_idx = int(len(X_all) * (1 - TEST_SET_SIZE_PERCENT))

        # Perbaiki logika split: Pastikan trainval punya setidaknya window_size + 1 bar untuk training
        # dan test set punya setidaknya 1 bar.
        # split_idx adalah titik akhir data trainval (eksklusif)
        # Data test dimulai dari split_idx
        if split_idx <= WINDOW_SIZE: # Harus ada setidaknya WINDOW_SIZE+1 elemen di X_trainval
             logging.error(f"❌ Ukuran data trainval ({split_idx} sampel) tidak cukup untuk window_size={WINDOW_SIZE}. Butuh minimal {WINDOW_SIZE + 1} sampel.")
             sys.exit(1)

        if len(X_all) - split_idx < 1: # Pastikan ada setidaknya 1 elemen di X_test
             logging.error(f"❌ Ukuran data test ({len(X_all) - split_idx} sampel) tidak cukup. Butuh minimal 1 sampel.")
             sys.exit(1)


        X_trainval, y_trainval = X_all[:split_idx], y_all[:split_idx]
        X_test, y_test = X_all[split_idx:], y_all[split_idx:]
        y_time_test = y_time_all[split_idx:]

        logging.info(f"✅ Data dibagi: Training+Validation ({len(X_trainval)} sampel), Test ({len(X_test)} sampel)")

        N_SPLITS_TSCV = 5
        tscv = TimeSeriesSplit(n_splits=N_SPLITS_TSCV)

        best_val_loss_in_tscv = float('inf')
        temp_model_path = os.path.join(ARTIFACTS_DIR, "temp_best_model_fold_tscv.h5")

        logging.info(f"⏳ Memulai Time Series Cross-Validation dengan {tscv.n_splits} fold pada data Training+Validation...")

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_trainval)):
            logging.info(f"--- 🔁 Fold {fold+1}/{tscv.n_splits} ---")

            X_train, y_train = X_trainval[train_idx], y_trainval[train_idx]
            X_val, y_val = X_trainval[val_idx], y_trainval[val_idx]
            
            # Tambahkan validasi ukuran fold jika terlalu kecil untuk batching/training
            if len(X_train) == 0 or len(X_val) == 0:
                 logging.warning(f"⚠️ Fold {fold+1}: Ukuran data train ({len(X_train)}) atau validasi ({len(X_val)}) kosong. Melewati fold ini.")
                 continue

            train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(len(X_train), seed=42).batch(32).prefetch(tf.data.AUTOTUNE)
            val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(32).prefetch(tf.data.AUTOTUNE)

            fold_model = build_lstm_attention_model(X_trainval.shape[1:], y_trainval.shape[1])

            trained_fold_model, history = train_model(fold_model, train_ds, val_ds, name=f"fold_{fold+1}")

            val_loss, val_mae, val_mse = trained_fold_model.evaluate(val_ds, verbose=0)

            logging.info(f"✅ Fold {fold+1} selesai. Val Loss (scaled): {val_loss:.4f}, Val MAE (scaled): {val_mae:.4f}")

            if val_loss < best_val_loss_in_tscv:
                best_val_loss_in_tscv = val_loss
                trained_fold_model.save(temp_model_path)
                logging.info(f"⭐ Model Fold {fold+1} adalah yang terbaik sejauh ini (Val Loss scaled: {best_val_loss_in_tscv:.4f}), disimpan sementara di '{temp_model_path}'.")

        # Cek apakah ada model terbaik yang disimpan
        if not os.path.exists(temp_model_path):
            logging.error(f"❌ Tidak ada model terbaik yang ditemukan atau disimpan selama TSCV di '{temp_model_path}'. Proses training gagal atau semua fold dilewati.")
            sys.exit(1)

        logging.info(f"--- ⏳ Memuat model terbaik dari '{temp_model_path}' untuk evaluasi akhir pada Test Set ---")

        custom_objects = {'MultiHeadAttention': MultiHeadAttention}
        final_best_model_from_tscv = tf.keras.models.load_model(temp_model_path, custom_objects=custom_objects)

        test_preds_original, y_test_original, y_time_test_eval, test_metrics = evaluate_model(
            final_best_model_from_tscv, X_test, y_test, scaler_y, y_time_test, set_name="Test Set Akhir"
        )

        FINAL_MODEL_PATH = os.path.join(ARTIFACTS_DIR, "best_model.h5")
        final_best_model_from_tscv.save(FINAL_MODEL_PATH)
        logging.info(f"✅ Model terbaik (berdasarkan validasi TSCV) disimpan di '{FINAL_MODEL_PATH}'.")

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

        logging.info("⏳ Membuat visualisasi prediksi pada Test Set...")
        plt.figure(figsize=(15, 8))

        num_points_to_plot = min(len(y_test_original), 150)
        start_plot_idx = len(y_test_original) - num_points_to_plot

        # Gunakan waktu aktual dari y_time_test_eval untuk sumbu X
        plot_times = y_time_test_eval[start_plot_idx:]


        plt.subplot(3, 1, 1)
        plt.plot(plot_times, y_test_original[start_plot_idx:, 0], label='Aktual High', color='blue', marker='.')
        plt.plot(plot_times, test_preds_original[start_plot_idx:, 0], label='Prediksi High', color='red', linestyle='--', marker='.')
        plt.title(f'Prediksi vs Aktual High - Test Set ({num_points_to_plot} Titik Terakhir)')
        plt.ylabel("Harga")
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout() # Pastikan label tidak tumpang tindih setelah rotasi


        plt.subplot(3, 1, 2)
        plt.plot(plot_times, y_test_original[start_plot_idx:, 1], label='Aktual Low', color='blue', marker='.')
        plt.plot(plot_times, test_preds_original[start_plot_idx:, 1], label='Prediksi Low', color='red', linestyle='--', marker='.')
        plt.title(f'Prediksi vs Aktual Low - Test Set ({num_points_to_plot} Titik Terakhir)')
        plt.ylabel("Harga")
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout() # Pastikan label tidak tumpang tindih setelah rotasi


        plt.subplot(3, 1, 3)
        plt.plot(plot_times, y_test_original[start_plot_idx:, 2], label='Aktual Close', color='blue', marker='.')
        plt.plot(plot_times, test_preds_original[start_plot_idx:, 2], label='Prediksi Close', color='red', linestyle='--', marker='.')
        plt.title(f'Prediksi vs Aktual Close - Test Set ({num_points_to_plot} Titik Terakhir)')
        plt.xlabel("Waktu")
        plt.ylabel("Harga")
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout() # Pastikan label tidak tumpang tindih setelah rotasi


        VISUALIZATION_PATH = os.path.join(ARTIFACTS_DIR, "test_prediction.png")
        plt.savefig(VISUALIZATION_PATH)
        logging.info(f"✅ Visualisasi prediksi Test Set disimpan di '{VISUALIZATION_PATH}'")

        logging.info("⏳ Menyimpan hasil evaluasi akhir...")
        metadata["test_set_size"] = len(X_test)
        metadata["test_set_metrics"] = test_metrics
        METADATA_PATH = os.path.join(ARTIFACTS_DIR, "model_info.json")
        with open(METADATA_PATH, "w") as f:
            json.dump(metadata, f, indent=4)
        logging.info(f"✅ Metadata model (termasuk hasil evaluasi test set) disimpan di '{METADATA_PATH}'")


        logging.info("--- Proses Training Model Selesai ---")

    except ValueError as ve:
        logging.error(f"❌ Terjadi ValueError: {ve}")
        sys.exit(1)
    except ConnectionError as ce:
         logging.error(f"❌ Terjadi ConnectionError: {ce}")
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

API_WINDOW_SIZE = None
API_FEATURES_USED = None
API_OUTPUTS_PREDICTED = None


@app.before_first_request
def load_artifacts():
    global loaded_model, loaded_scaler_X, loaded_scaler_y, loaded_metadata
    global API_WINDOW_SIZE, API_FEATURES_USED, API_OUTPUTS_PREDICTED

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
        logging.info(f"  Konfigurasi dari metadata: Window Size = {API_WINDOW_SIZE}, Fitur Digunakan = {API_FEATURES_USED}")
        logging.warning("⚠️ PERHATIAN: Perhitungan fitur teknikal (SMA/EMA) di API hanya berdasarkan jendela input yang diterima. Ini MUNGKIN berbeda dengan cara fitur dihitung saat training (menggunakan seluruh riwayat data), yang dapat mempengaruhi akurasi prediksi.")


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

        window_data_raw = content.get("window")

        if window_data_raw is None:
            logging.warning("Permintaan /predict tanpa key 'window' di body JSON.")
            return jsonify({'error': 'Missing "window" key in JSON request body. Expected list of candlestick bar objects.'}), 400

        if not isinstance(window_data_raw, list):
             logging.warning(f"Input 'window' bukan list. Tipe: {type(window_data_raw)}")
             return jsonify({'error': 'Input "window" must be a list of candlestick bar objects.'}), 400

        if len(window_data_raw) != API_WINDOW_SIZE:
            logging.warning(f"Panjang input window tidak sesuai. Diharapkan {API_WINDOW_SIZE} bar, tapi mendapat {len(window_data_raw)}")
            return jsonify({'error': f'Input window must contain exactly {API_WINDOW_SIZE} candlestick bars.'}), 400

        # --- Pemrosesan Input di API ---
        # 1. Konversi ke DataFrame, seleksi kolom yang diharapkan
        required_base_cols = ['time', 'open', 'high', 'low', 'close', 'tick_volume']
        df_window = pd.DataFrame(window_data_raw)

        # Validasi keberadaan kolom dasar
        if not all(col in df_window.columns for col in required_base_cols):
            missing = [col for col in required_base_cols if col not in df_window.columns]
            logging.warning(f"Input window DataFrame tidak memiliki kolom dasar yang dibutuhkan: {missing}")
            return jsonify({'error': f'Each bar object in the window must contain keys: {required_base_cols}. Missing: {missing}'}), 400

        # Seleksi hanya kolom dasar yang dibutuhkan untuk kontrol urutan
        df_window = df_window[required_base_cols]

        # Validasi data numerik dasar
        if not np.isfinite(df_window[['open', 'high', 'low', 'close', 'tick_volume']]).all().all():
            logging.warning("Input window mengandung nilai NaN atau Inf di kolom OHLCV/volume dasar.")
            return jsonify({'error': 'Input window contains non-finite values (NaN or Inf) in base OHLCV or volume columns.'}), 400

        # 2. Tambahkan fitur teknikal (akan menghasilkan NaN di awal jendela)
        df_window = add_technical_features(df_window)

        # 3. Tangani NaN yang dihasilkan dari fitur teknikal di awal jendela
        # JANGAN GUNAKAN dropna() karena akan mengurangi panjang jendela
        # Mengisi NaN dengan 0 sebagai strategi sederhana. Ini adalah kompromi logika vs training.
        # Strategi lain bisa fillna(method='ffill').
        df_window = df_window.fillna(0) # Mengisi NaN dengan 0

        # 4. Seleksi fitur yang digunakan saat training, pastikan urutannya benar
        # Gunakan API_FEATURES_USED dari metadata
        try:
            # Pastikan semua API_FEATURES_USED ada setelah feature engineering dan fillna
            if not all(f in df_window.columns for f in API_FEATURES_USED):
                 missing_api_features = [f for f in API_FEATURES_USED if f not in df_window.columns]
                 logging.error(f"Fitur yang diharapkan ({API_FEATURES_USED}) tidak lengkap setelah pemrosesan input API. Missing: {missing_api_features}")
                 # Ini menandakan ada masalah dalam add_technical_features atau API_FEATURES_USED
                 return jsonify({'error': 'Internal error during feature processing. Check API logs.'}), 500


            arr_to_scale = df_window[API_FEATURES_USED].values

        except Exception as e:
             logging.error(f"Gagal mengambil atau memvalidasi kolom fitur dari DataFrame input API setelah pemrosesan: {e}", exc_info=True)
             return jsonify({'error': f'Error preparing data for scaling: {e}'}), 500

        # 5. Validasi shape array sebelum scaling dan prediksi
        if arr_to_scale.shape != (API_WINDOW_SIZE, len(API_FEATURES_USED)):
             logging.error(f"Shape array fitur input API tidak sesuai setelah pemrosesan. Diharapkan ({API_WINDOW_SIZE}, {len(API_FEATURES_USED)}), mendapat {arr_to_scale.shape}")
             # Ini bisa terjadi jika fillna(0) tidak berfungsi sesuai harapan atau ada masalah lain.
             return jsonify({'error': 'Processed input shape mismatch. Internal error, check API logs.'}), 500


        # 6. Scaling input
        try:
            arr_scaled = loaded_scaler_X.transform(arr_to_scale)
        except Exception as e:
             logging.error(f"Gagal melakukan scaling input X pada API: {e}", exc_info=True)
             return jsonify({'error': f'Failed to scale input data using loaded scaler: {e}'}), 500

        # 7. Menambah dimensi batch
        arr_scaled = np.expand_dims(arr_scaled, axis=0)

        # 8. Melakukan prediksi
        pred_scaled = loaded_model.predict(arr_scaled, verbose=0)

        # 9. Inverse transform prediksi
        try:
            pred_original = loaded_scaler_y.inverse_transform(pred_scaled)
        except Exception as e:
             logging.error(f"Gagal melakukan inverse scaling output Y pada API: {e}", exc_info=True)
             return jsonify({'error': f'Failed to inverse scale prediction output: {e}'}), 500

        # 10. Mengembalikan hasil
        predicted_values = pred_original[0].tolist()
        output_dict = dict(zip(API_OUTPUTS_PREDICTED, predicted_values))

        return jsonify({'prediction': output_dict})

    except ValueError as ve:
        logging.error(f"ValueError saat prediksi API: {str(ve)}", exc_info=True)
        return jsonify({'error': f'Input data processing failed: {str(ve)}'}), 400
    except KeyError as ke:
         logging.error(f"KeyError saat prediksi API (JSON structure issue): {str(ke)}", exc_info=True)
         return jsonify({'error': f'Invalid JSON structure: Missing key {str(ke)}'}), 400
    except Exception as e:
        logging.error(f"Error tak terduga saat prediksi API: {str(e)}", exc_info=True)
        return jsonify({'error': 'Internal server error during prediction. Check API logs for details.'}), 500

# ============ Entry Point ============
if __name__ == "__main__":
    mode = input("Ketik 'train' untuk melatih model atau 'run' untuk menjalankan API: ").strip().lower()

    if mode == "train":
        main_training()
    elif mode == "run":
        logging.info("💡 Menjalankan Flask API.")
        logging.info(f"💡 Memuat artefak dari '{ARTIFACTS_DIR}'.")
        logging.info("💡 API akan mendengarkan di http://0.0.0.0:5000")
        logging.info("💡 Endpoint prediksi: POST ke http://0.0.0.0:5000/predict")
        # Pesan input API diperbarui
        logging.info(f"💡 Endpoint /predict mengharapkan JSON dengan key 'window' berisi LIST dari {API_WINDOW_SIZE if API_WINDOW_SIZE else '??'} objek candlestick bar (dengan key 'time', 'open', 'high', 'low', 'close', 'tick_volume'). API akan melakukan feature engineering (SMA/EMA) dan scaling secara internal.")
        logging.info("💡 CATATAN: Perhitungan SMA/EMA di API hanya pada jendela input, mungkin berbeda dengan training.")


        app.run(host='0.0.0.0', port=5000, debug=False)
    else:
        print("❌ Pilihan tidak dikenali. Gunakan 'train' atau 'run'.")
