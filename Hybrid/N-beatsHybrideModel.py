# SanClass Trading Labs Core 
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import LSTM, Dense, Dropout, MultiHeadAttention, LayerNormalization, GlobalAveragePooling1D
from tensorflow.keras.layers import Concatenate, Add
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
import random
import time

# ============ Reproducibility ============
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)
os.environ['TF_DETERMINISTIC_OPS'] = '1'

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

    max_retries = 5
    retry_delay_sec = 5

    for i in range(max_retries):
        logging.info(f"⏳ Mencoba koneksi MT5 (Percobaan {i+1}/{max_retries})...")
        if mt5.initialize(login=login, server=server, password=password):
            logging.info("✅ Berhasil login ke MetaTrader5")
            return
        last_error = mt5.last_error()
        logging.warning(f"⚠️ Gagal koneksi MT5 (Percobaan {i+1}/{max_retries}). Kode error: {last_error}")
        if i < max_retries - 1:
             time.sleep(retry_delay_sec)

    logging.error(f"❌ Gagal koneksi MT5 setelah {max_retries} percobaan.")
    raise ConnectionError(f"Gagal koneksi MT5: {last_error}")

def acquire_data(symbol, timeframe, start, end):
    connect_mt5()
    logging.info(f"⏳ Mengambil data {symbol} dari {start} hingga {end} dengan timeframe {timeframe}...")
    print(f"📊 VISUALISASI: Memulai pengambilan data historis {symbol}")

    try:
        rates = mt5.copy_rates_range(symbol, timeframe, start, end)
        if rates is None:
             raise Exception(f"mt5.copy_rates_range mengembalikan None. Error: {mt5.last_error()}")

    except Exception as e:
        logging.error(f"❌ Error saat memanggil copy_rates_range: {e}")
        rates = None
    finally:
        mt5.shutdown()
        logging.info("✅ Koneksi MT5 di-shutdown.")

    if rates is None or len(rates) == 0:
        error_msg = f"❌ Gagal mengambil data dari MT5 untuk {symbol} ({start} to {end}). Data kosong atau terjadi kesalahan."
        logging.error(error_msg)
        return pd.DataFrame()

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df = df.sort_values('time').reset_index(drop=True)
    logging.info(f"✅ Berhasil mengambil {len(df)} baris data.")
    print(f"📊 VISUALISASI: Data historis berhasil diambil: {len(df)} candle")

    return df[['time', 'open', 'high', 'low', 'close', 'tick_volume']]

# ============ Preprocessing ============
def calculate_atr(df, period):
    high_low = df['high'] - df['low']
    high_close_prev = abs(df['high'] - df['close'].shift(1))
    low_close_prev = abs(df['low'] - df['close'].shift(1))

    tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()

    return atr

def add_technical_features(df):
    df['ema_21'] = df['close'].ewm(span=21, adjust=False).mean()
    df['ema_89'] = df['close'].ewm(span=89, adjust=False).mean()

    df['atr_9'] = calculate_atr(df, 9)
    df['atr_45'] = calculate_atr(df, 45)

    df['sma_5'] = df['close'].rolling(window=5, min_periods=1).mean()
    df['ema_5'] = df['close'].ewm(span=5, adjust=False).mean()

    df['volatility_ratio'] = df['atr_9'] / df['atr_45'].rolling(window=3).mean()

    df['trend_direction'] = np.where(df['ema_21'] > df['ema_89'], 1, -1)

    df['momentum'] = df['close'] - df['close'].shift(5)

    print(f"📊 VISUALISASI: Indikator teknikal berhasil ditambahkan")
    return df

def preprocess_for_training(df):
    df = df.dropna().reset_index(drop=True)
    if df.empty:
         raise ValueError("DataFrame kosong setelah menghapus NaN awal.")

    df = add_technical_features(df)

    df = df.dropna().reset_index(drop=True)
    if df.empty:
         raise ValueError("DataFrame kosong setelah menambahkan fitur teknikal dan menghapus NaN lagi.")

    features = ['open', 'high', 'low', 'close', 'tick_volume',
                'ema_21', 'ema_89', 'atr_9', 'atr_45',
                'sma_5', 'ema_5', 'volatility_ratio', 'trend_direction', 'momentum']

    if not all(f in df.columns for f in features):
         missing_features = [f for f in features if f not in df.columns]
         raise ValueError(f"Fitur yang diharapkan tidak ada di DataFrame setelah pra-pemrosesan: {missing_features}")

    logging.info(f"✅ Data diproses untuk training. Menggunakan {len(features)} fitur: {features}")
    print(f"📊 VISUALISASI: Fitur untuk training siap: {len(features)} fitur")

    scaler_X = MinMaxScaler()
    df_scaled_features = scaler_X.fit_transform(df[features])
    df[features] = df_scaled_features
    logging.info("✅ Fitur input (X) untuk training berhasil di-scale.")

    return df, scaler_X, features

def create_dataset(df, window_size, feature_cols):
    X, y = [], []
    y_time = []

    if len(df) <= window_size:
        logging.warning(f"Tidak cukup data ({len(df)} bar) untuk membuat dataset dengan window_size={window_size}. Butuh setidaknya {window_size + 1} bar.")
        return np.array([]), np.array([]), pd.DatetimeIndex([])

    for i in range(len(df) - window_size):
        window = df.iloc[i : i + window_size][feature_cols].values
        target = df.iloc[i + window_size][['high', 'low', 'close']].values
        target_time = df.iloc[i + window_size]['time']

        if not np.isfinite(window).all() or not np.isfinite(target).all():
            logging.warning(f"⚠️ NaN/Inf terdeteksi di jendela atau target pada index awal {i}. Melompati sampel ini.")
            continue

        X.append(window)
        y.append(target)
        y_time.append(target_time)

    X = np.array(X)
    y = np.array(y)
    y_time = pd.to_datetime(y_time)

    if len(X) == 0 and len(df) > window_size:
         logging.warning(f"Tidak ada sampel valid yang dibuat meskipun data awal cukup. Mungkin ada NaN/Inf di data setelah pra-pemrosesan yang terlewat.")

    logging.info(f"✅ Dataset dibuat. Shape X: {X.shape}, Shape y: {y.shape}")
    print(f"📊 VISUALISASI: Dataset siap dengan {len(X)} sampel training")
    return X, y, y_time

# ============ N-BEATS Block ============
def create_nbeats_block(x, units, theta_dim, share_theta=False, layer_norm=True):
    for i in range(4):
        x = Dense(units, activation='relu')(x)
        if layer_norm:
            x = LayerNormalization()(x)

    theta = Dense(theta_dim)(x)

    backcast_size = x.shape[1]
    forecast_size = 3

    if share_theta:
        backcast = Dense(backcast_size, name='backcast')(theta)
        forecast = Dense(forecast_size, name='forecast')(theta)
    else:
        backcast_theta, forecast_theta = tf.split(theta, 2, axis=-1)
        backcast = Dense(backcast_size, name='backcast')(backcast_theta)
        forecast = Dense(forecast_size, name='forecast')(forecast_theta)

    return backcast, forecast

# ============ Model (LSTM + Attention + N-BEATS) ============
def build_hybrid_model(input_shape, output_dim, n_blocks=3):
    print(f"📊 VISUALISASI: Membangun model hibrida dengan {n_blocks} blok N-BEATS")

    inputs = Input(shape=input_shape)

    lstm_out = LSTM(128, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(0.0005))(inputs)
    lstm_out = Dropout(0.2)(lstm_out)

    attention_output = MultiHeadAttention(
        num_heads=4, key_dim=32, dropout=0.1)(lstm_out, lstm_out)

    lstm_out = Add()([lstm_out, attention_output])
    lstm_out = LayerNormalization()(lstm_out)

    lstm_out = GlobalAveragePooling1D()(lstm_out)

    lstm_out = Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0005))(lstm_out)
    lstm_out = Dropout(0.3)(lstm_out)

    backcast = inputs
    forecast = None
    units = 128
    theta_dim = 32

    for i in range(n_blocks):
        block_input = backcast
        b, f = create_nbeats_block(
            block_input,
            units=units,
            theta_dim=theta_dim,
            share_theta=(i % 2 == 0),
            layer_norm=True
        )

        backcast = Subtract()([backcast, b]) if i > 0 else b

        if forecast is None:
            forecast = f
        else:
            forecast = Add()([forecast, f])

    combined = Concatenate()([lstm_out, forecast])

    outputs = Dense(output_dim)(combined)

    model = Model(inputs=inputs, outputs=outputs)
    logging.info("✅ Model Hibrida LSTM + Attention + N-BEATS berhasil dibangun.")
    model.summary(print_fn=lambda msg: logging.info(msg))

    return model

# ============ Training ============
def train_model(model, train_ds, val_ds, name):
    logging.info(f"⏳ Memulai training model fold: {name}...")
    print(f"📊 VISUALISASI: Training model fold {name} dimulai")

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=['mae', 'mse']
    )

    checkpoint_path = f"model_artifacts/{name}_checkpoint.h5"

    callbacks = [
        EarlyStopping(
            patience=15,
            restore_best_weights=True,
            verbose=1,
            monitor='val_loss'
        ),
        ModelCheckpoint(
            checkpoint_path,
            save_best_only=True,
            verbose=1,
            monitor='val_loss'
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            verbose=1,
            min_lr=0.00001
        )
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=300,
        callbacks=callbacks,
        verbose=1
    )

    print(f"📊 VISUALISASI: Training model fold {name} selesai")

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'Training & Validation Loss - {name}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Training MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title(f'Training & Validation MAE - {name}')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"model_artifacts/{name}_training_history.png")
    print(f"📊 VISUALISASI: Grafik history training model fold {name} disimpan")

    return model, history

# ============ Evaluation ============
def evaluate_model(model, X_set, y_set_scaled, scaler_y, y_set_time, set_name="Test"):
    logging.info(f"⏳ Mengevaluasi model pada set {set_name} ({len(X_set)} sampel)...")
    print(f"📊 VISUALISASI: Evaluasi model pada {set_name} dimulai")

    if len(X_set) == 0:
         logging.warning(f"⚠️ Tidak ada data di set {set_name} untuk dievaluasi.")
         return np.array([]), np.array([]), pd.DatetimeIndex([]), {'mae': np.nan, 'rmse': np.nan, 'mape': np.nan, 'directional_accuracy': np.nan} # Tambah dir_acc saat data kosong

    preds_scaled = model.predict(X_set, verbose=0)

    try:
        y_set_original = scaler_y.inverse_transform(y_set_scaled)
        preds_original = scaler_y.inverse_transform(preds_scaled)
    except Exception as e:
        logging.error(f"❌ Gagal inverse transform data untuk set {set_name}: {e}")
        return preds_scaled, y_set_scaled, y_set_time, {'mae': np.nan, 'rmse': np.nan, 'mape': np.nan, 'directional_accuracy': np.nan}

    mae = mean_absolute_error(y_set_original, preds_original)
    rmse = np.sqrt(mean_squared_error(y_set_original, preds_original))

    if np.all(y_set_original < 1e-6):
         mape = np.nan
         logging.warning(f"⚠️ Semua nilai aktual di set {set_name} mendekati nol. MAPE tidak dihitung.")
    else:
         mape_per_output = np.abs((y_set_original - preds_original) / (y_set_original + 1e-8)) * 100
         mape_per_output = mape_per_output[np.isfinite(mape_per_output)]
         mape = np.mean(mape_per_output) if mape_per_output.size > 0 else np.nan

    direction_actual = np.sign(y_set_original[1:, 2] - y_set_original[:-1, 2])
    direction_pred = np.sign(preds_original[1:, 2] - preds_original[:-1, 2])
    directional_accuracy = np.mean(direction_actual == direction_pred) * 100 if len(direction_actual) > 0 else np.nan

    logging.info(f"📊 Hasil Evaluasi {set_name}:")
    logging.info(f"  MAE  (skala asli): {mae:.4f}")
    logging.info(f"  RMSE (skala asli): {rmse:.4f}")
    logging.info(f"  Directional Accuracy: {directional_accuracy:.2f}%")

    if not np.isnan(mape):
         logging.info(f"  MAPE (skala asli): {mape:.2f}%")
    else:
         logging.info("  MAPE (skala asli): N/A")

    print(f"📊 VISUALISASI: Evaluasi model pada {set_name} selesai")
    print(f"📊 VISUALISASI: MAE={mae:.4f}, RMSE={rmse:.4f}, DIR_ACC={directional_accuracy:.2f}%")

    return preds_original, y_set_original, y_set_time, {
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'directional_accuracy': directional_accuracy
    }

# ============ Kelas Subtract (untuk N-BEATS) ============
class Subtract(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Subtract, self).__init__(**kwargs)

    def call(self, inputs):
        return inputs[0] - inputs[1]

# ============ Main Training ============
def main_training():
    logging.info("--- Memulai Proses Training Model ---")
    print("📊 VISUALISASI: Proses training model dimulai")

    try:
        df = acquire_data('XAUUSD', mt5.TIMEFRAME_D1, pd.Timestamp(2008,1,1), pd.Timestamp(2024,12,31))

        if df.empty:
             logging.error("❌ Akuisisi data gagal atau mengembalikan data kosong.")
             sys.exit(1)

        df_processed, scaler_X, feature_cols = preprocess_for_training(df.copy())

        WINDOW_SIZE = 10
        max_feature_lookback = 89

        X_all, y_raw_all, y_time_all = create_dataset(df_processed, window_size=WINDOW_SIZE, feature_cols=feature_cols)

        if len(X_all) == 0:
            logging.error(f"❌ Gagal membuat dataset training dari data yang diproses dengan window_size={WINDOW_SIZE}. Tidak cukup sampel valid.")
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

        metadata = {
            "model_architecture": "Hybrid: LSTM + Attention + N-BEATS",
            "features_used": feature_cols,
            "outputs_predicted": ["high", "low", "close"],
            "window_size": WINDOW_SIZE,
            "feature_lookback": max_feature_lookback,
            "training_data_range": f"{df['time'].iloc[0].strftime('%Y-%m-%d')} to {df['time'].iloc[-1].strftime('%Y-%m-%d')}" if not df.empty else "N/A",
            "notes": "Model hibrida dengan N-BEATS, LSTM, dan Attention mechanism. Menggunakan indikator ATR 9, ATR 45, EMA 21, dan EMA 89.",
            "hyperparameters": {
                "lstm_units": 128,
                "attention_heads": 4,
                "attention_key_dim": 32,
                "dropout_rate_lstm": 0.2,
                "dropout_rate_dense": 0.3,
                "l2_regularization": 0.0005,
                "optimizer": "Adam",
                "learning_rate": 0.001,
                "batch_size": 32,
                "early_stopping_patience": 15,
                "tscv_splits": 5,
                "n_beats_blocks": 3,
                "n_beats_units": 128,
                "n_beats_theta_dim": 32
            }
        }
        METADATA_PATH = os.path.join(ARTIFACTS_DIR, "model_info.json")
        with open(METADATA_PATH, "w") as f:
            json.dump(metadata, f, indent=4)
        logging.info(f"✅ Metadata model disimpan di '{METADATA_PATH}'.")

        TEST_SET_SIZE_PERCENT = 0.2
        split_idx = int(len(X_all) * (1 - TEST_SET_SIZE_PERCENT))

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

        N_SPLITS_TSCV = 5
        tscv = TimeSeriesSplit(n_splits=N_SPLITS_TSCV)

        best_val_loss_in_tscv = float('inf')
        temp_model_path = os.path.join(ARTIFACTS_DIR, "temp_best_model_fold_tscv.h5")

        logging.info(f"⏳ Memulai Time Series Cross-Validation dengan {tscv.n_splits} fold pada data Training+Validation...")
        print(f"📊 VISUALISASI: Time Series Cross-Validation dengan {tscv.n_splits} fold dimulai")

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_trainval)):
            logging.info(f"--- 🔁 Fold {fold+1}/{tscv.n_splits} ---")

            X_train, y_train = X_trainval[train_idx], y_trainval[train_idx]
            X_val, y_val = X_trainval[val_idx], y_trainval[val_idx]

            if len(X_train) == 0 or len(X_val) == 0:
                 logging.warning(f"⚠️ Fold {fold+1}: Ukuran data train ({len(X_train)}) atau validasi ({len(X_val)}) kosong. Melewati fold ini.")
                 continue

            np.random.seed(42 + fold)
            tf.random.set_seed(42 + fold)
            random.seed(42 + fold)

            train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(len(X_train), seed=42).batch(32).prefetch(tf.data.AUTOTUNE)
            val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(32).prefetch(tf.data.AUTOTUNE)

            fold_model = build_hybrid_model(
                X_trainval.shape[1:],
                y_trainval.shape[1],
                n_blocks=3
            )

            trained_fold_model, history = train_model(fold_model, train_ds, val_ds, name=f"fold_{fold+1}")

            val_loss, val_mae, val_mse = trained_fold_model.evaluate(val_ds, verbose=0)

            logging.info(f"✅ Fold {fold+1} selesai. Val Loss (scaled): {val_loss:.4f}, Val MAE (scaled): {val_mae:.4f}")

            if val_loss < best_val_loss_in_tscv:
                best_val_loss_in_tscv = val_loss
                trained_fold_model.save(temp_model_path)
                logging.info(f"💾 Fold {fold+1} menjadi model terbaik sejauh ini. Model disimpan.")

        logging.info(f"✅ Time Series Cross-Validation selesai.")

        logging.info(f"⏳ Memuat model terbaik dari cross-validation...")
        print(f"📊 VISUALISASI: Memuat model terbaik dari {N_SPLITS_TSCV} fold TSCV")
        model = tf.keras.models.load_model(
            temp_model_path,
            custom_objects={'Subtract': Subtract}
        )
        logging.info(f"✅ Model terbaik dari cross-validation berhasil dimuat.")

        preds_test, y_test_original, y_time_test, test_metrics = evaluate_model(
            model, X_test, y_test, scaler_y, y_time_test, "Test"
        )

        if len(preds_test) > 0:
            plt.figure(figsize=(14, 7))
            plt.subplot(3, 1, 1)
            plt.plot(y_time_test, y_test_original[:, 0], 'b-', label='Aktual High')
            plt.plot(y_time_test, preds_test[:, 0], 'r--', label='Prediksi High')
            plt.title('Prediksi vs Aktual (High)')
            plt.legend()
            plt.grid(True)

            plt.subplot(3, 1, 2)
            plt.plot(y_time_test, y_test_original[:, 1], 'g-', label='Aktual Low')
            plt.plot(y_time_test, preds_test[:, 1], 'r--', label='Prediksi Low')
            plt.title('Prediksi vs Aktual (Low)')
            plt.legend()
            plt.grid(True)

            plt.subplot(3, 1, 3)
            plt.plot(y_time_test, y_test_original[:, 2], 'k-', label='Aktual Close')
            plt.plot(y_time_test, preds_test[:, 2], 'r--', label='Prediksi Close')
            plt.title('Prediksi vs Aktual (Close)')
            plt.legend()
            plt.grid(True)

            plt.tight_layout()
            plt.savefig(os.path.join(ARTIFACTS_DIR, "test_set_predictions.png"))
            print("📊 VISUALISASI: Grafik prediksi vs aktual pada test set disimpan")

        logging.info("⏳ Menyimpan model akhir dan artefak terkait...")
        model.save(os.path.join(ARTIFACTS_DIR, "final_model.h5"))

        metrics_dict = {
            "test_metrics": test_metrics
        }

        with open(os.path.join(ARTIFACTS_DIR, "evaluation_metrics.json"), "w") as f:
            metrics_dict_serializable = {}
            for key, val_dict in metrics_dict.items():
                metrics_dict_serializable[key] = {k: float(v) if not np.isnan(v) else None for k, v in val_dict.items()}

            json.dump(metrics_dict_serializable, f, indent=4)

        logging.info("✅ Model akhir dan artefak terkait berhasil disimpan.")
        print("📊 VISUALISASI: Model akhir berhasil disimpan")

        return model, metadata, test_metrics

    except Exception as e:
        logging.error(f"❌ Terjadi kesalahan dalam proses training: {e}")
        print(f"📊 VISUALISASI: ERROR - Proses training terhenti karena kesalahan")
        import traceback
        logging.error(traceback.format_exc())
        sys.exit(1)

# ============ Inference ============
def process_new_data_for_prediction(new_data_df, scaler_X, feature_cols, window_size):
    logging.info(f"⏳ Memproses data baru untuk prediksi...")
    print(f"📊 VISUALISASI: Memproses {len(new_data_df)} bar data baru untuk prediksi")

    # Pastikan new_data_df adalah DataFrame
    if not isinstance(new_data_df, pd.DataFrame):
        raise TypeError("Input data untuk process_new_data_for_prediction harus berupa Pandas DataFrame.")

    # Tambahkan fitur teknikal
    processed_data = add_technical_features(new_data_df.copy())

    # Hapus NaN setelah tambah fitur (konsisten dengan training)
    processed_data = processed_data.dropna().reset_index(drop=True)

    # Pastikan data cukup setelah preprocessing
    if len(processed_data) < window_size:
        raise ValueError(f"⚠️ Data tidak cukup setelah penambahan indikator teknikal dan pembersihan NaN. Tersisa {len(processed_data)} dari {window_size} yang dibutuhkan untuk jendela.")

    # Ambil sampel data terakhir sesuai window_size
    recent_data = processed_data.iloc[-window_size:].copy()

    # Validasi keberadaan semua kolom fitur yang digunakan saat training
    if not all(col in recent_data.columns for col in feature_cols):
         missing_cols = [col for col in feature_cols if col not in recent_data.columns]
         raise ValueError(f"⚠️ Kolom fitur yang diharapkan tidak ditemukan di data yang diproses untuk prediksi: {missing_cols}")


    # Scaling fitur X menggunakan scaler yang sudah di-fit di training
    # Pastikan tidak ada NaN/Inf sebelum scaling
    if not np.isfinite(recent_data[feature_cols]).all().all():
         logging.error("❌ Data untuk scaling mengandung NaN atau Inf setelah preprocessing.")
         raise ValueError("Data input untuk scaling mengandung NaN atau Inf.")

    recent_features = scaler_X.transform(recent_data[feature_cols])

    # Reshape untuk input model (menambah dimensi batch)
    X_pred = np.array([recent_features])

    logging.info(f"✅ Data baru berhasil diproses untuk prediksi. Shape: {X_pred.shape}")
    return X_pred

def predict_next_bar(model, new_data_df, scaler_X, scaler_y, feature_cols, window_size):
    """Memprediksi bar berikutnya dari data terbaru."""
    try:
        print(f"📊 VISUALISasi: Prediksi bar berikutnya dimulai")
        # Persiapkan data untuk prediksi menggunakan fungsi terpisah
        X_pred = process_new_data_for_prediction(new_data_df, scaler_X, feature_cols, window_size)

        # Prediksi dalam skala yang di-scale
        pred_scaled = model.predict(X_pred, verbose=0)[0]

        # Inverse transform untuk mendapatkan prediksi dalam skala asli
        pred_original = scaler_y.inverse_transform(pred_scaled.reshape(1, -1)).reshape(-1)

        # Bentuk output yang berarti (gunakan nama dari metadata jika perlu, tapi HLC tetap konsisten)
        prediction = {
            'high': float(pred_original[0]),
            'low': float(pred_original[1]),
            'close': float(pred_original[2])
        }

        logging.info(f"✅ Prediksi bar berikutnya: High={prediction['high']:.5f}, Low={prediction['low']:.5f}, Close={prediction['close']:.5f}")
        print(f"📊 VISUALISASI: Prediksi bar berikutnya berhasil - High={prediction['high']:.5f}, Low={prediction['low']:.5f}, Close={prediction['close']:.5f}")

        return prediction

    except Exception as e:
        logging.error(f"❌ Gagal memprediksi bar berikutnya: {e}")
        print(f"📊 VISUALISASI: ERROR - Prediksi bar berikutnya gagal")
        import traceback
        logging.error(traceback.format_exc())
        return None

# ============ Implementasi Rolling Window Backtest ============
def perform_rolling_window_backtest(df, model, scaler_X, scaler_y, feature_cols, window_size, backtest_window=30):
    logging.info(f"⏳ Memulai backtest dengan rolling window {backtest_window} candle...")
    print(f"📊 VISUALISASI: Backtest rolling window ({backtest_window} candle) dimulai")

    # Memastikan data cukup untuk backtest
    # Butuh window_size untuk input prediksi pertama + backtest_window langkah prediksi
    # dan juga lookback maksimum fitur teknikal sebelum jendela pertama.
    # Misal window_size=10, lookback_fitur_max=89, backtest_window=30.
    # Kita butuh data historis sebanyak lookback_fitur_max + window_size + backtest_window
    # Namun, fungsi process_new_data_for_prediction menerima data dan melakukan dropna.
    # Jadi kita butuh data yang cukup SEBELUM titik awal backtest.
    # Data minimum adalah lookback_fitur_max + window_size + backtest_window + 1
    # Asumsikan df input ke fungsi ini adalah data historis penuh.
    # Titik awal backtest adalah setelah data trainval + window_size (untuk jendela pertama)
    # atau sederhana: ambil backtest_window data terakhir dari dataset yang bisa dibuat jendelanya
    # Start index untuk backtest adalah len(data_jendela) - backtest_window
    # Data_jendela adalah hasil create_dataset.
    # Untuk backtest ini, kita akan memprediksi N bar terakhir dari df_processed
    # Data yang dibutuhkan untuk memprediksi bar di index `i` adalah bar `i - window_size + 1` hingga `i`.
    # Untuk menghitung fitur di bar-bar ini, kita butuh data sebelumnya sebanyak lookback_fitur_max.
    # Jadi, untuk memprediksi bar ke-(len(df_processed) - backtest_window), kita butuh data
    # hingga index len(df_processed) - backtest_window - 1, termasuk lookback fitur.
    # Data yang dibutuhkan adalah df_processed.iloc[: len(df_processed) - backtest_window]
    # dan kita loop dari sana.
    # Index pertama yang akan diprediksi adalah (len(df_processed) - backtest_window)
    # Data input untuk prediksi ini adalah bar dari index (len(df_processed) - backtest_window - window_size)
    # hingga index (len(df_processed) - backtest_window - 1).
    # Untuk menghitung fitur di bar (len(df_processed) - backtest_window - window_size), kita butuh
    # data sebelumnya sejauh lookback_fitur_max.
    # Jadi, data minimum yang dibutuhkan untuk backtest adalah:
    # lookback_fitur_max + window_size + backtest_window

    # Perbaiki cek data minimum
    max_feature_lookback_val = 89 # Hardcode lagi untuk fungsi ini, atau ambil dari metadata jika fungsi dipanggil dari main
    if len(df) < max_feature_lookback_val + window_size + backtest_window:
        logging.error(f"❌ Data tidak cukup untuk backtest. Butuh min {max_feature_lookback_val + window_size + backtest_window} candle, tersedia {len(df)}.")
        return None

    # Lakukan preprocessing (tambah fitur, dropna) pada seluruh data input backtest
    df_processed_bt = add_technical_features(df.copy())
    df_processed_bt = df_processed_bt.dropna().reset_index(drop=True)

    if len(df_processed_bt) < window_size + backtest_window:
         logging.error(f"❌ Data tidak cukup setelah preprocessing untuk backtest. Tersisa {len(df_processed_bt)} candle, butuh min {window_size + backtest_window}.")
         return None


    actuals = []
    predictions = []
    timestamps = []

    # Index pertama yang datanya akan digunakan sebagai akhir jendela input
    start_idx_input_window = len(df_processed_bt) - backtest_window - 1 # Ini adalah index bar terakhir di jendela input pertama
    start_idx_prediction = start_idx_input_window + 1 # Ini adalah index bar pertama yang diprediksi


    if start_idx_input_window < window_size - 1:
        logging.error(f"❌ Index awal jendela input ({start_idx_input_window}) kurang dari window_size ({window_size - 1}). Tidak bisa membentuk jendela pertama.")
        return None

    # Loop melalui setiap titik waktu yang akan diprediksi dalam periode backtest
    for i in range(backtest_window):
        # Index bar yang akan diprediksi saat ini
        current_predict_idx = start_idx_prediction + i

        # Data historis yang tersedia hingga *sebelum* bar yang diprediksi
        # Ini mencakup data yang cukup untuk membentuk jendela input DAN menghitung fitur teknikal
        historical_data_for_current_pred = df_processed_bt.iloc[:current_predict_idx].copy()

        if len(historical_data_for_current_pred) < window_size:
             logging.warning(f"⚠️ Backtest candle ke-{i}: Data historis ({len(historical_data_for_current_pred)} bar) kurang dari window size ({window_size}). Melewati prediksi ini.")
             continue # Lewati jika data historis kurang dari window size

        # Gunakan fungsi prediksi untuk memproses data dan memprediksi
        try:
            # process_new_data_for_prediction sudah mengambil N bar terakhir dan memprosesnya
            # historical_data_for_current_pred harus cukup panjang
            X_pred = process_new_data_for_prediction(
                historical_data_for_current_pred,
                scaler_X,
                feature_cols,
                window_size
            )

            # Prediksi
            pred_scaled = model.predict(X_pred, verbose=0)[0]
            pred_original = scaler_y.inverse_transform(pred_scaled.reshape(1, -1)).reshape(-1)

            # Ambil nilai aktual bar yang diprediksi
            actual_next = df_processed_bt.iloc[current_predict_idx][['high', 'low', 'close']].values
            timestamp = df_processed_bt.iloc[current_predict_idx]['time']

            predictions.append(pred_original)
            actuals.append(actual_next)
            timestamps.append(timestamp)

            if (i + 1) % 10 == 0: # Log setiap 10 candle
                logging.info(f"⏳ Backtest progress: {i + 1}/{backtest_window} candle diprediksi.")

        except Exception as e:
            logging.error(f"❌ Error pada backtest candle ke-{i}: {e}")
            # Lanjutkan loop meskipun ada error pada satu prediksi
            continue


    if not predictions:
        logging.warning("⚠️ Tidak ada prediksi yang dihasilkan dalam backtest.")
        return None

    predictions = np.array(predictions)
    actuals = np.array(actuals)
    timestamps = np.array(timestamps)

    mae = mean_absolute_error(actuals, predictions)
    rmse = np.sqrt(mean_squared_error(actuals, predictions))

    mape_values = np.abs((actuals - predictions) / np.maximum(np.abs(actuals), 1e-10)) * 100
    mape_values = mape_values[np.isfinite(mape_values)] # Hapus NaN/Inf dari MAPE
    mape = np.mean(mape_values) if mape_values.size > 0 else np.nan

    direction_actual = np.sign(actuals[1:, 2] - actuals[:-1, 2])
    direction_pred = np.sign(predictions[1:, 2] - predictions[:-1, 2])
    directional_accuracy = np.mean(direction_actual == direction_pred) * 100 if len(direction_actual) > 0 else np.nan

    logging.info(f"📊 Hasil Backtest ({backtest_window} candle):")
    logging.info(f"  MAE : {mae:.4f}")
    logging.info(f"  RMSE: {rmse:.4f}")
    if not np.isnan(mape):
        logging.info(f"  MAPE: {mape:.2f}%")
    else:
         logging.info("  MAPE: N/A")
    logging.info(f"  Directional Accuracy: {directional_accuracy:.2f}%")

    print(f"📊 VISUALISASI: Backtest selesai - MAE={mae:.4f}, RMSE={rmse:.4f}, MAPE={mape:.2f}%, DIR_ACC={directional_accuracy:.2f}%")

    plt.figure(figsize=(14, 10))

    plt.subplot(3, 1, 1)
    plt.plot(timestamps, actuals[:, 0], 'b-', label='Aktual High')
    plt.plot(timestamps, predictions[:, 0], 'r--', label='Prediksi High')
    plt.title('Backtest Prediksi vs Aktual (High)')
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.plot(timestamps, actuals[:, 1], 'g-', label='Aktual Low')
    plt.plot(timestamps, predictions[:, 1], 'r--', label='Prediksi Low')
    plt.title('Backtest Prediksi vs Aktual (Low)')
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.plot(timestamps, actuals[:, 2], 'k-', label='Aktual Close')
    plt.plot(timestamps, predictions[:, 2], 'r--', label='Prediksi Close')
    plt.title('Backtest Prediksi vs Aktual (Close)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('model_artifacts/backtest_results.png')
    print("📊 VISUALISASI: Grafik hasil backtest disimpan")

    return {
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'directional_accuracy': directional_accuracy,
        'timestamps': timestamps.tolist(), # Konversi numpy array ke list untuk JSON
        'actuals': actuals.tolist(),
        'predictions': predictions.tolist()
    }

# ============ Flask API ============
app = Flask(__name__)
model = None
scaler_X = None
scaler_y = None
model_info = None # Metadata loaded at startup

# Load model artifacts at startup
# Menggunakan @app.before_first_request tidak ideal jika server dijalankan dengan worker
# Lebih baik memuat di luar request context jika memungkinkan, atau di main thread.
# Pindah logika pemuatan ke load_model_artifacts dan panggil dari __main__

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "UP", "model_loaded": model is not None})

@app.route('/info', methods=['GET'])
def model_information():
    global model_info
    if model_info is None:
        return jsonify({"error": "Metadata model belum dimuat"}), 503

    # Hapus kredensial atau informasi sensitif lainnya jika ada di model_info
    display_info = model_info.copy()
    return jsonify(display_info)

@app.route('/predict', methods=['POST'])
def predict():
    global model, scaler_X, scaler_y, model_info

    if model is None or scaler_X is None or scaler_y is None or model_info is None:
        logging.error("❌ Permintaan /predict ditolak karena model atau artefak belum siap.")
        return jsonify({"error": "Model atau scaler belum dimuat. Silakan coba lagi nanti atau hubungi administrator."}), 503

    try:
        data = request.json
        if not data or 'data' not in data:
            logging.warning("Permintaan /predict tanpa body JSON atau key 'data'.")
            return jsonify({"error": "Format data tidak valid. Membutuhkan objek JSON dengan key 'data' berisi array data candlestick (dengan time, open, high, low, close, tick_volume)."}), 400

        # Ekstrak data dari request
        json_data = data['data']

        # Validasi apakah input 'data' adalah list
        if not isinstance(json_data, list):
             logging.warning("Input 'data' bukan list.")
             return jsonify({"error": "Input 'data' harus berupa array JSON (list of objects)."}), 400

        # Konversi ke DataFrame. API perlu data yang cukup untuk feature lookback + window size
        # API_WINDOW_SIZE dan API_FEATURE_LOOKBACK dimuat dari metadata
        window_size = model_info.get('window_size')
        feature_lookback = model_info.get('feature_lookback', 0) # Default 0 jika tidak ada
        feature_cols = model_info.get('features_used') # Ambil nama fitur dari metadata

        if window_size is None or feature_cols is None:
             logging.error("❌ Metadata tidak lengkap: window_size atau features_used tidak ada.")
             return jsonify({"error": "Konfigurasi model (window size, features) tidak lengkap. Hubungi administrator."}), 500

        # API mengharapkan input 'data' berisi riwayat yang cukup
        # untuk menghitung fitur DAN membentuk jendela prediksi.
        # Minimal data yang dibutuhkan = window_size + feature_lookback
        min_data_required = window_size + feature_lookback

        if len(json_data) < min_data_required:
             logging.warning(f"Data yang diterima ({len(json_data)}) kurang dari minimum yang dibutuhkan ({min_data_required}).")
             return jsonify({"error": f"Data historis yang diberikan kurang. Butuh minimal {min_data_required} candle data untuk prediksi."}), 400

        df_raw_input = pd.DataFrame(json_data)

        # Validasi kolom dasar di DataFrame input
        required_base_cols = ['time', 'open', 'high', 'low', 'close', 'tick_volume']
        if not all(col in df_raw_input.columns for col in required_base_cols):
            missing = [col for col in required_base_cols if col not in df_raw_input.columns]
            logging.warning(f"Input data DataFrame tidak memiliki kolom dasar yang dibutuhkan: {missing}")
            return jsonify({"error": f"Setiap objek candlestick dalam array 'data' harus memiliki kolom: {required_base_cols}. Missing: {missing}"}), 400

        # Konversi kolom waktu
        try:
             df_raw_input['time'] = pd.to_datetime(df_raw_input['time'])
        except Exception as e:
             logging.warning(f"Gagal mengkonversi kolom waktu: {e}")
             return jsonify({"error": f"Format kolom 'time' tidak valid: {e}"}), 400

        # Urutkan data berdasarkan waktu (penting untuk indikator)
        df_raw_input = df_raw_input.sort_values('time').reset_index(drop=True)

        # Lakukan preprocessing pada data input mentah (tambah fitur, dropna)
        # Fungsi process_new_data_for_prediction sudah melakukan ini dan mengambil jendela terakhir
        # Cukup panggil fungsi prediksi dengan DataFrame mentah yang sudah validasi dasar
        prediction = predict_next_bar(model, df_raw_input, scaler_X, scaler_y, feature_cols, window_size)

        if prediction is None:
            return jsonify({"error": "Gagal membuat prediksi setelah memproses data."}), 500

        return jsonify({
            "prediction": prediction,
            "timestamp": pd.Timestamp.now().isoformat() # Timestamp kapan prediksi dibuat oleh API
        })

    except Exception as e:
        logging.error(f"Error saat prediksi API: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route('/backtest', methods=['POST'])
def backtest():
    global model, scaler_X, scaler_y, model_info

    if model is None or scaler_X is None or scaler_y is None or model_info is None:
        logging.error("❌ Permintaan /backtest ditolak karena model atau artefak belum siap.")
        return jsonify({"error": "Model atau scaler belum dimuat. Silakan coba lagi nanti atau hubungi administrator."}), 503

    try:
        data = request.json
        if not data or 'data' not in data:
            logging.warning("Permintaan /backtest tanpa body JSON atau key 'data'.")
            return jsonify({"error": "Format data tidak valid. Membutuhkan objek JSON dengan key 'data' berisi array data candlestick."}), 400

        # Ambil parameter backtest window (jumlah candle yang akan diprediksi di akhir data)
        backtest_window = data.get('window', 100) # Default 100 candle untuk backtest

        # Ekstrak data historis penuh dari request
        json_data = data['data']

        # Validasi apakah input 'data' adalah list
        if not isinstance(json_data, list):
             logging.warning("Input 'data' untuk backtest bukan list.")
             return jsonify({"error": "Input 'data' untuk backtest harus berupa array JSON (list of objects)."}), 400

        df_full_history_input = pd.DataFrame(json_data)

        # Validasi kolom dasar di DataFrame input
        required_base_cols = ['time', 'open', 'high', 'low', 'close', 'tick_volume']
        if not all(col in df_full_history_input.columns for col in required_base_cols):
            missing = [col for col in required_base_cols if col not in df_full_history_input.columns]
            logging.warning(f"Input data backtest DataFrame tidak memiliki kolom dasar yang dibutuhkan: {missing}")
            return jsonify({"error": f"Setiap objek candlestick dalam array 'data' untuk backtest harus memiliki kolom: {required_base_cols}. Missing: {missing}"}), 400

        # Konversi kolom waktu
        try:
             df_full_history_input['time'] = pd.to_datetime(df_full_history_input['time'])
        except Exception as e:
             logging.warning(f"Gagal mengkonversi kolom waktu pada data backtest: {e}")
             return jsonify({"error": f"Format kolom 'time' pada data backtest tidak valid: {e}"}), 400

        # Urutkan data berdasarkan waktu
        df_full_history_input = df_full_history_input.sort_values('time').reset_index(drop=True)


        # Ambil parameter model dari metadata
        window_size = model_info.get('window_size')
        feature_cols = model_info.get('features_used')
        feature_lookback = model_info.get('feature_lookback', 0)

        if window_size is None or feature_cols is None:
             logging.error("❌ Metadata tidak lengkap: window_size atau features_used tidak ada.")
             return jsonify({"error": "Konfigurasi model (window size, features) tidak lengkap. Hubungi administrator."}), 500

        # Validasi minimum data untuk backtest
        # Data yang dibutuhkan adalah lookback_fitur_max + window_size + backtest_window
        min_data_required_bt = feature_lookback + window_size + backtest_window
        if len(df_full_history_input) < min_data_required_bt:
             logging.warning(f"Data historis untuk backtest ({len(df_full_history_input)}) kurang dari minimum yang dibutuhkan ({min_data_required_bt}).")
             return jsonify({"error": f"Data historis yang diberikan untuk backtest kurang. Butuh minimal {min_data_required_bt} candle data."}), 400


        # Jalankan backtest menggunakan fungsi terpisah
        # Fungsi perform_rolling_window_backtest akan melakukan preprocessing internal
        backtest_results = perform_rolling_window_backtest(
            df_full_history_input,
            model,
            scaler_X,
            scaler_y,
            feature_cols,
            window_size,
            backtest_window
        )

        if backtest_results is None:
            return jsonify({"error": "Gagal melakukan backtest."}), 500 # Fungsi backtest sudah log detail error

        # Convert numpy values in results dict to Python native types for JSON serialization
        serializable_results = {
            'mae': float(backtest_results.get('mae', np.nan)),
            'rmse': float(backtest_results.get('rmse', np.nan)),
            'mape': float(backtest_results.get('mape', np.nan)),
            'directional_accuracy': float(backtest_results.get('directional_accuracy', np.nan)),
             # Timestamps, actuals, predictions sudah dalam bentuk list dari fungsi backtest
            'timestamps': backtest_results.get('timestamps', []),
            'actuals': backtest_results.get('actuals', []),
            'predictions': backtest_results.get('predictions', [])
        }


        return jsonify({
            "backtest_metrics": {
                 k: v for k, v in serializable_results.items() if k not in ['timestamps', 'actuals', 'predictions']
            },
            "backtest_window_size": backtest_window,
            "predictions_data": {
                 'timestamps': serializable_results['timestamps'],
                 'actuals': serializable_results['actuals'],
                 'predictions': serializable_results['predictions']
            },
            "timestamp_api_response": pd.Timestamp.now().isoformat()
        })

    except Exception as e:
        logging.error(f"Error saat backtest API: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

def load_model_artifacts():
    global model, scaler_X, scaler_y, model_info

    artifacts_dir = "model_artifacts"
    final_model_path = os.path.join(artifacts_dir, "final_model.h5")
    scaler_x_path = os.path.join(artifacts_dir, "scaler_X.save")
    scaler_y_path = os.path.join(artifacts_dir, "scaler_y.save")
    model_info_path = os.path.join(artifacts_dir, "model_info.json")

    # Check if all necessary files exist
    required_files = [final_model_path, scaler_x_path, scaler_y_path, model_info_path]
    for file_path in required_files:
        if not os.path.exists(file_path):
            logging.error(f"❌ File artefak tidak ditemukan: {file_path}. Jalankan mode 'train' terlebih dahulu.")
            return False # Gagal memuat jika ada file hilang

    try:
        logging.info("⏳ Memuat model dan artefak pendukung...")
        # Memuat scaler
        scaler_X = joblib.load(scaler_x_path)
        scaler_y = joblib.load(scaler_y_path)

        # Memuat metadata model
        with open(model_info_path, "r") as f:
            model_info = json.load(f)

        # Muat custom objects yang dibutuhkan oleh model (seperti Subtract)
        custom_objects_map = {'Subtract': Subtract} # Tambahkan custom layer lain di sini jika ada

        # Memuat model
        model = tf.keras.models.load_model(
            final_model_path,
            custom_objects=custom_objects_map # Pass custom objects
        )

        logging.info("✅ Model dan artefak pendukung berhasil dimuat.")
        logging.info(f"  Metadata model: {model_info}")
        return True

    except Exception as e:
        logging.error(f"❌ Gagal memuat model dan artefak: {e}", exc_info=True)
        return False

# ============ Main Function ============
if __name__ == "__main__":
    # Set seed di main execution path
    np.random.seed(42)
    tf.random.set_seed(42)
    random.seed(42)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'

    # Menggunakan sys.argv untuk mode train
    if len(sys.argv) > 1 and sys.argv[1].lower() == "--train":
        logging.info("🚀 Mode: Training")
        # main_training akan menyimpan model dan artefak, lalu mengembalikan objek model dll.
        # Kita tidak perlu menangkap kembalian jika langsung menjalankan API setelahnya.
        main_training()
        logging.info("✅ Proses training selesai.")

        # Setelah training selesai, API server akan berjalan secara otomatis
        # Panggil load_model_artifacts untuk memuat model yang baru disimpan
        if load_model_artifacts():
            logging.info("🌐 Menjalankan API server setelah training di port 5000...")
            # Gunakan debug=False untuk produksi
            app.run(host='0.0.0.0', port=5000, debug=False)
        else:
            logging.error("❌ API server tidak dapat dijalankan setelah training karena gagal memuat model yang baru disimpan.")
            sys.exit(1)

    # Mode default adalah menjalankan API server
    else:
        logging.info("🚀 Mode: API Server (Default)")
        if load_model_artifacts():
            logging.info("🌐 Menjalankan API server di port 5000...")
            # Gunakan debug=False untuk produksi
            app.run(host='0.0.0.0', port=5000, debug=False)
        else:
            logging.error("❌ API server tidak dapat dijalankan karena gagal memuat model. Pastikan Anda sudah menjalankan mode '--train' terlebih dahulu.")
            sys.exit(1)
