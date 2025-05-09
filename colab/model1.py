# QuantAI End-to-End Colab Script
# Deskripsi: Skrip Python tunggal untuk menjalankan pipeline AI QuantAI di Google Colab.
# Menggabungkan logika dari preprocessing, training, optimasi model, dan konfigurasi.
# Dirancang untuk memproses file CSV 'XAUUSD_M5.csv'.

# ---------------------------------------------------------------------------
# BAGIAN 0: INSTALASI DEPENDENSI
# ---------------------------------------------------------------------------
# Jalankan sel ini terlebih dahulu di Colab untuk menginstal package yang dibutuhkan.
# Hapus komentar pada baris di bawah ini jika menjalankan di Colab.
# !pip install tensorflow>=2.10.0,<2.16.0 tensorflow-model-optimization>=0.7.0,<0.8.0 numpy>=1.23.0,<1.27.0 pandas>=1.5.0,<2.3.0 scikit-learn>=1.1.0,<1.5.0 PyYAML>=6.0,<6.1 joblib>=1.2.0,<1.5.0 matplotlib>=3.5.0,<3.9.0 tensorboard>=2.10.0,<2.16.0

print("Dependensi (jika di-uncomment) akan diinstal. Pastikan menjalankan sel ini.")

# ---------------------------------------------------------------------------
# BAGIAN 1: IMPORT LIBRARY
# ---------------------------------------------------------------------------
import os
import glob # Meskipun hanya satu file, beberapa fungsi internal mungkin masih menggunakannya
import argparse # Tidak digunakan secara aktif di Colab, tapi bagian dari struktur asli
import yaml # Untuk memuat string config, meskipun bisa langsung dict
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_model_optimization as tfmot
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv1D, MaxPooling1D, LSTM, Dense, Dropout, BatchNormalization,
    TimeDistributed, concatenate, Bidirectional
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from tensorflow.keras.mixed_precision import set_global_policy as set_mixed_precision_policy
import datetime
import logging
import joblib
from google.colab import files # Untuk upload file di Colab

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ---------------------------------------------------------------------------
# BAGIAN 2: KONFIGURASI (dari quantai_config.yaml)
# ---------------------------------------------------------------------------
CONFIG_YAML_STRING = """
# config/quantai_config.yaml (Embedded)
dataset:
  # Path ke direktori CSV tidak lagi digunakan secara langsung, file diupload
  file_pattern: "XAUUSD_M5.csv" # Nama file spesifik
  sort_by_column: "date" # Kolom untuk sorting, setelah header ditambahkan
  # Kolom fitur setelah header ditambahkan ke CSV
  feature_columns: ["open", "high", "low", "close", "volume"]
  target_column: "close"
  sequence_length: 60
  prediction_horizon: 5
  train_split: 0.7
  validation_split: 0.15
  test_split: 0.15

model:
  cnn_filters: [64, 128]
  cnn_kernel_size: 3
  cnn_pool_size: 2
  cnn_dropout: 0.2
  cnn_batch_norm: true
  lstm_units: [100, 50]
  lstm_dropout: 0.2
  lstm_batch_norm: true
  dense_units: [64]
  dense_dropout: 0.1

training:
  epochs: 50 # Kurangi untuk testing cepat di Colab jika perlu, misal: 5-10
  batch_size: 32
  learning_rate: 0.001
  optimizer: "adam"
  loss_function: "mse"
  early_stopping_patience: 10
  random_seed: 42

optimization:
  mixed_precision: false # Set true jika Colab GPU mendukung (biasanya T4/P100 mendukung)
  quantization:
    enable: true
    quant_type: "int8"
    use_representative_dataset: true
    num_calibration_samples: 100 # Kurangi jika dataset kecil, misal: 50-100
    int8_fallback_float16: true
    fallback_to_fp16_on_error: true
  pruning:
    enable: false # Set true untuk mencoba, tapi butuh penyesuaian step
    initial_sparsity: 0.25
    final_sparsity: 0.75
    begin_step: 200 # Sesuaikan berdasarkan (num_train_samples / batch_size) * epoch_mulai
    end_step: 1000  # Sesuaikan berdasarkan (num_train_samples / batch_size) * epoch_selesai

environment:
  platform: ["Colab"]
  accelerators: ["CPU", "GPU", "TPU"]
  min_ram_gb: 4

deployment:
  target_env: "tflite_on_colab"
"""

# Fungsi untuk memuat konfigurasi dari string YAML
def load_config_from_string(yaml_string):
    """Memuat konfigurasi YAML dari string."""
    logging.info("Memuat konfigurasi dari string YAML internal.")
    config = yaml.safe_load(yaml_string)
    logging.info("Konfigurasi berhasil dimuat.")
    return config

# ---------------------------------------------------------------------------
# BAGIAN 3: KONTEN WORKFLOW (dari workflows/quantai_train_pipeline.yml)
# ---------------------------------------------------------------------------
# Konten file workflows/quantai_train_pipeline.yml disertakan di sini sebagai
# informasi dan referensi. Ini tidak dieksekusi langsung oleh skrip Python ini.
# File YML ini biasanya digunakan oleh sistem CI/CD seperti GitHub Actions.

QUANTUM_TRAIN_PIPELINE_YML_CONTENT = """
# workflows/quantai_train_pipeline.yml
# Deskripsi: File workflow untuk Continuous Integration/Continuous Deployment (CI/CD)
# pipeline pelatihan model AI.
# File ini mendefinisikan tahapan-tahapan mulai dari persiapan data hingga deployment model.
# Dapat diadaptasi untuk platform CI/CD seperti GitHub Actions, GitLab CI, dll.

name: QuantAI Training Pipeline

on:
  push:
    branches:
      - main
  pull_request:
    branches:
      - main
  workflow_dispatch:

jobs:
  setup_and_preprocess:
    runs-on: ubuntu-latest
    # ... (konten yml lainnya) ...
  train_model:
    needs: setup_and_preprocess
    # ... (konten yml lainnya) ...
  validate_and_quantize_model:
    needs: train_model
    # ... (konten yml lainnya) ...
  package_and_deploy:
    needs: validate_and_quantize_model
    # ... (konten yml lainnya) ...
"""
# logging.info("Konten referensi quantai_train_pipeline.yml:\n" + QUANTUM_TRAIN_PIPELINE_YML_CONTENT)


# ---------------------------------------------------------------------------
# BAGIAN 4: FUNGSI-FUNGSI UTAMA (dari internal_workflow/quantai_main_pipeline.py)
# ---------------------------------------------------------------------------

# Fungsi untuk memuat dan memproses CSV spesifik 'XAUUSD_M5.csv'
def load_and_process_single_csv(file_path, sort_by_col='date', expected_cols=None):
    """
    Memuat satu file CSV tanpa header, menambahkan header, mengurutkan,
    dan memilih kolom fitur.
    """
    if expected_cols is None:
        expected_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
    
    logging.info(f"Memuat file CSV: {file_path}")
    try:
        # Baca CSV tanpa header, dengan pemisah ';'
        df = pd.read_csv(file_path, header=None, sep=';', names=expected_cols)
        logging.info(f"Berhasil memuat {file_path}. Shape awal: {df.shape}")
        logging.info(f"Beberapa baris data awal (sebelum proses):\n{df.head()}")
    except Exception as e:
        logging.error(f"Error saat memuat file CSV {file_path}: {e}")
        raise

    # Pastikan semua kolom yang diharapkan ada
    for col in expected_cols:
        if col not in df.columns:
            logging.error(f"Kolom yang diharapkan '{col}' tidak ditemukan di CSV setelah diberi nama.")
            raise ValueError(f"Kolom '{col}' hilang.")

    # Mengonversi kolom tanggal ke datetime dan mengurutkan
    sort_by_col_lower = sort_by_col.lower()
    if sort_by_col_lower in df.columns:
        try:
            # Coba beberapa format tanggal umum jika konversi standar gagal
            # Format '2023.01.02 00:05:00' atau '20230102 000500' atau timestamp
            # Jika format tanggal adalah YYYY.MM.DD HH:MM:SS
            if '.' in str(df[sort_by_col_lower].iloc[0]) and ':' in str(df[sort_by_col_lower].iloc[0]):
                 df[sort_by_col_lower] = pd.to_datetime(df[sort_by_col_lower], format='%Y.%m.%d %H:%M:%S', errors='coerce')
            # Jika format tanggal adalah YYYYMMDD HHMMSS (tanpa pemisah selain spasi)
            elif ' ' in str(df[sort_by_col_lower].iloc[0]) and str(df[sort_by_col_lower].iloc[0]).replace(' ','').isdigit() and len(str(df[sort_by_col_lower].iloc[0]).replace(' ','')) > 8 :
                 df[sort_by_col_lower] = pd.to_datetime(df[sort_by_col_lower], format='%Y%m%d %H%M%S', errors='coerce')
            else: # Coba infer otomatis, atau jika itu adalah timestamp numerik
                df[sort_by_col_lower] = pd.to_datetime(df[sort_by_col_lower], errors='coerce')

            # Periksa apakah ada NaT setelah konversi
            if df[sort_by_col_lower].isnull().any():
                logging.warning(f"Beberapa nilai di kolom '{sort_by_col_lower}' tidak dapat dikonversi ke datetime. Baris dengan NaT akan dihapus.")
                logging.info(f"Contoh nilai yang gagal dikonversi: {df[df[sort_by_col_lower].isnull()][sort_by_col_lower].head()}")
                df.dropna(subset=[sort_by_col_lower], inplace=True)


            df = df.sort_values(by=sort_by_col_lower).reset_index(drop=True)
            logging.info(f"Data diurutkan berdasarkan kolom: {sort_by_col_lower}")
        except Exception as e:
            logging.error(f"Error saat mengonversi atau mengurutkan berdasarkan kolom tanggal '{sort_by_col_lower}': {e}")
            logging.info(f"Contoh data dari kolom tanggal: {df[sort_by_col_lower].head()}")
            raise
    else:
        logging.warning(f"Kolom untuk pengurutan '{sort_by_col_lower}' tidak ditemukan.")

    # Memilih hanya kolom fitur (tanpa kolom tanggal jika tanggal bukan fitur)
    # Kolom fitur diambil dari config, pastikan sudah di-lowercase
    feature_cols_from_config = [col.lower() for col in CONFIG_DATA['dataset']['feature_columns']]
    
    # Filter df_features agar hanya berisi kolom yang ada di feature_cols_from_config
    df_features = df[[col for col in df.columns if col.lower() in feature_cols_from_config]]
    
    logging.info(f"Bentuk DataFrame Fitur: {df_features.shape}")
    logging.info(f"Menggunakan fitur: {list(df_features.columns)}")
    
    dates_series = df[sort_by_col_lower] if sort_by_col_lower in df.columns else None
    return df_features, dates_series

# Fungsi untuk membuat dataset sekuensial (time series)
def create_sequences(data, n_past, n_future, target_col_index=3):
    X, y = [], []
    for i in range(n_past, len(data) - n_future + 1):
        X.append(data[i - n_past:i, :])
        y.append(data[i:i + n_future, target_col_index])
    return np.array(X), np.array(y)

# Fungsi preprocessing data
def preprocess_data(config, raw_csv_path, output_dir_colab):
    logging.info("Memulai preprocessing data...")
    data_cfg = config['dataset']
    
    # Kolom yang diharapkan ada di CSV (setelah diberi nama)
    expected_csv_cols = ['date'] + data_cfg['feature_columns']
    # Pastikan unik dan lowercase
    expected_csv_cols = sorted(list(set(col.lower() for col in expected_csv_cols)))


    df_features, df_dates = load_and_process_single_csv(
        raw_csv_path,
        sort_by_col=data_cfg['sort_by_column'].lower(),
        expected_cols=expected_csv_cols # Ini adalah nama kolom yang akan kita berikan
    )

    if df_features.empty:
        logging.error("Dataframe fitur kosong setelah dimuat. Membatalkan preprocessing.")
        return None, None # Mengembalikan None jika gagal

    # Pastikan kolom fitur yang digunakan untuk scaling adalah yang ada di df_features
    # dan sesuai dengan feature_columns dari config (setelah lowercase)
    actual_feature_columns_for_scaling = [col for col in df_features.columns if col.lower() in [fc.lower() for fc in data_cfg['feature_columns']]]
    df_features_to_scale = df_features[actual_feature_columns_for_scaling]


    df_features_to_scale.fillna(method='ffill', inplace=True)
    df_features_to_scale.fillna(method='bfill', inplace=True)

    if df_features_to_scale.isnull().any().any():
        logging.warning("Nilai NaN masih ada setelah ffill/bfill. Menghapus baris dengan NaN.")
        df_features_to_scale.dropna(inplace=True)

    if df_features_to_scale.empty:
        logging.error("Dataframe fitur menjadi kosong setelah penanganan NaN. Periksa kualitas data.")
        return None, None

    scaler = MinMaxScaler()
    # Hanya scale kolom yang memang fitur numerik
    scaled_data = scaler.fit_transform(df_features_to_scale)

    target_col_name_config = data_cfg.get('target_column', 'close').lower()
    # Cari index target_col_name_config di dalam actual_feature_columns_for_scaling
    if target_col_name_config not in [col.lower() for col in actual_feature_columns_for_scaling]:
        logging.error(f"Kolom target '{target_col_name_config}' tidak ditemukan dalam daftar fitur aktual yang di-scale: {actual_feature_columns_for_scaling}")
        raise ValueError(f"Kolom target '{target_col_name_config}' tidak ditemukan.")
    target_col_idx = [col.lower() for col in actual_feature_columns_for_scaling].index(target_col_name_config)


    X, y = create_sequences(scaled_data, data_cfg['sequence_length'], data_cfg['prediction_horizon'], target_col_idx)
    
    if X.shape[0] == 0:
        logging.error("Tidak ada sekuens yang dibuat. Periksa panjang data dan parameter sekuens.")
        # raise ValueError("Tidak ada sekuens yang dibuat.") # Bisa jadi error jika data terlalu sedikit
        return None, None

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(1 - data_cfg['train_split']), random_state=config['training']['random_seed'], shuffle=False
    )
    # Hindari error jika X_temp terlalu kecil untuk dibagi lagi
    if X_temp.shape[0] < 2 : # Minimal 2 sampel untuk bisa dibagi
        logging.warning(f"Data temporer (X_temp) terlalu kecil ({X_temp.shape[0]} sampel) untuk dibagi menjadi validation dan test. Menggunakan semua X_temp sebagai validation, test set akan kosong.")
        X_val, y_val = X_temp, y_temp
        X_test, y_test = np.array([]).reshape(0, X.shape[1], X.shape[2]), np.array([]).reshape(0, y.shape[1])
    else:
        val_test_split_ratio = data_cfg['test_split'] / (data_cfg['validation_split'] + data_cfg['test_split'])
        if val_test_split_ratio >= 1.0 or val_test_split_ratio <= 0.0: # Jika test_split 0 atau validation_split 0
            if data_cfg['test_split'] == 0 and data_cfg['validation_split'] > 0 :
                 X_val, X_test, y_val, y_test = X_temp, np.array([]).reshape(0, X.shape[1], X.shape[2]), y_temp, np.array([]).reshape(0, y.shape[1])
            elif data_cfg['validation_split'] == 0 and data_cfg['test_split'] > 0:
                 X_val, X_test, y_val, y_test = np.array([]).reshape(0, X.shape[1], X.shape[2]), X_temp, np.array([]).reshape(0, y.shape[1]), y_temp
            else: # Keduanya 0, tidak ideal
                 X_val, X_test, y_val, y_test = X_temp, np.array([]).reshape(0, X.shape[1], X.shape[2]), y_temp, np.array([]).reshape(0, y.shape[1])
        else:
            X_val, X_test, y_val, y_test = train_test_split(
                X_temp, y_temp, test_size=val_test_split_ratio,
                random_state=config['training']['random_seed'], shuffle=False
            )


    logging.info(f"Bentuk data Train: X={X_train.shape}, y={y_train.shape}")
    logging.info(f"Bentuk data Validasi: X={X_val.shape}, y={y_val.shape}")
    logging.info(f"Bentuk data Test: X={X_test.shape}, y={y_test.shape}")

    os.makedirs(output_dir_colab, exist_ok=True)
    processed_data_path = os.path.join(output_dir_colab, "processed_data.npz")
    np.savez(processed_data_path, X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, X_test=X_test, y_test=y_test, feature_columns=actual_feature_columns_for_scaling)
    
    logging.info(f"Data yang sudah diproses disimpan ke {processed_data_path}")
    scaler_path = os.path.join(output_dir_colab, "scaler.joblib")
    joblib.dump(scaler, scaler_path)
    logging.info(f"Scaler disimpan ke {scaler_path}")
    return processed_data_path, actual_feature_columns_for_scaling # Kembalikan juga nama kolom fitur


# Fungsi untuk membangun model CNN+LSTM (sama seperti sebelumnya)
def build_model(input_shape, n_outputs, model_config, optimization_config):
    logging.info("Membangun model...")
    inputs = Input(shape=input_shape)

    x = Conv1D(filters=model_config['cnn_filters'][0], kernel_size=model_config['cnn_kernel_size'],
               activation='relu', padding='same')(inputs)
    if model_config.get('cnn_batch_norm', False): x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=model_config['cnn_pool_size'])(x)
    if model_config.get('cnn_dropout', 0.0) > 0: x = Dropout(model_config['cnn_dropout'])(x)

    if len(model_config['cnn_filters']) > 1:
        for filters_cnn in model_config['cnn_filters'][1:]:
            x = Conv1D(filters=filters_cnn, kernel_size=model_config['cnn_kernel_size'], activation='relu', padding='same')(x)
            if model_config.get('cnn_batch_norm', False): x = BatchNormalization()(x)
            x = MaxPooling1D(pool_size=model_config['cnn_pool_size'])(x)
            if model_config.get('cnn_dropout', 0.0) > 0: x = Dropout(model_config['cnn_dropout'])(x)
    
    num_lstm_layers = len(model_config['lstm_units'])
    for i, units_lstm in enumerate(model_config['lstm_units']):
        return_sequences_lstm = True if i < num_lstm_layers - 1 else False
        x = Bidirectional(LSTM(units=units_lstm, return_sequences=return_sequences_lstm))(x)
        if model_config.get('lstm_dropout', 0.0) > 0: x = Dropout(model_config['lstm_dropout'])(x)
        if model_config.get('lstm_batch_norm', False): x = BatchNormalization()(x)

    for units_dense in model_config['dense_units']:
        x = Dense(units_dense, activation='relu')(x)
        if model_config.get('dense_dropout', 0.0) > 0: x = Dropout(model_config['dense_dropout'])(x)
    
    outputs = Dense(n_outputs, activation='linear')(x)
    model = Model(inputs=inputs, outputs=outputs)
    
    if optimization_config.get('pruning', {}).get('enable', False):
        pruning_params = {
            'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
                initial_sparsity=optimization_config['pruning']['initial_sparsity'],
                final_sparsity=optimization_config['pruning']['final_sparsity'],
                begin_step=optimization_config['pruning']['begin_step'],
                end_step=optimization_config['pruning']['end_step']      
            )
        }
        model = tfmot.sparsity.keras.prune_low_magnitude(model, **pruning_params)
    
    logging.info("Model berhasil dibangun.")
    return model

# Fungsi untuk melatih model (disesuaikan sedikit untuk path Colab)
def train_model_colab(config, processed_data_path, model_output_dir_colab, metrics_output_dir_colab):
    logging.info("Memulai pelatihan model...")
    
    data = np.load(processed_data_path)
    X_train, y_train = data['X_train'], data['y_train']
    X_val, y_val = data['X_val'], data['y_val']

    if X_train.shape[0] == 0 or X_val.shape[0] == 0 :
        logging.error("Data training atau validasi kosong. Tidak dapat melatih model.")
        return None

    if X_train.ndim != 3:
        raise ValueError(f"X_train harus 3D, tetapi mendapatkan {X_train.ndim} dimensi.")

    input_shape = (X_train.shape[1], X_train.shape[2])
    n_outputs = y_train.shape[1]

    if config['optimization'].get('mixed_precision', False):
        logging.info("Mengaktifkan pelatihan mixed precision (float16).")
        # Cek ketersediaan GPU sebelum set policy
        gpus = tf.config.list_physical_devices('GPU')
        if not gpus:
            logging.warning("Mixed precision diaktifkan, tetapi tidak ada GPU yang terdeteksi oleh TensorFlow. Kebijakan mungkin tidak berpengaruh.")
        else:
            logging.info(f"GPU terdeteksi: {gpus}")
        set_mixed_precision_policy('mixed_float16')


    model = build_model(input_shape, n_outputs, config['model'], config['optimization'])
    
    optimizer_name = config['training']['optimizer'].lower()
    learning_rate = config['training']['learning_rate']
    
    if optimizer_name == 'adam': optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer_name == 'rmsprop': optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    else: optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
        
    model.compile(optimizer=optimizer, loss=config['training']['loss_function'], metrics=['mae', 'mse'])
    model.summary(print_fn=logging.info)

    os.makedirs(model_output_dir_colab, exist_ok=True)
    os.makedirs(metrics_output_dir_colab, exist_ok=True)
    
    checkpoint_path = os.path.join(model_output_dir_colab, "best_model.keras")
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=config['training']['early_stopping_patience'], restore_best_weights=True, verbose=1),
        ModelCheckpoint(filepath=checkpoint_path, monitor='val_loss', save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6, verbose=1),
        TensorBoard(log_dir=os.path.join(metrics_output_dir_colab, 'logs', datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))
    ]
    
    if config['optimization'].get('pruning', {}).get('enable', False):
        callbacks.append(tfmot.sparsity.keras.UpdatePruningStep())
        logging.info("Callback pruning ditambahkan.")

    batch_size = config['training']['batch_size']
    
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    logging.info(f"Pelatihan dengan batch size: {batch_size}, epochs: {config['training']['epochs']}")
    history = model.fit(
        train_dataset,
        epochs=config['training']['epochs'],
        validation_data=val_dataset,
        callbacks=callbacks,
        verbose=1
    )

    final_model_path = os.path.join(model_output_dir_colab, "final_model.keras")
    if config['optimization'].get('pruning', {}).get('enable', False):
        logging.info("Menghapus wrapper pruning dari model untuk disimpan.")
        model_stripped = tfmot.sparsity.keras.strip_pruning(model)
        model_stripped.save(final_model_path)
    else:
        model.save(final_model_path)
    
    logging.info(f"Pelatihan selesai. Model disimpan di {final_model_path}. Model terbaik di {checkpoint_path}")
    
    history_df = pd.DataFrame(history.history)
    history_path = os.path.join(metrics_output_dir_colab, "training_history.csv")
    history_df.to_csv(history_path, index=False)
    logging.info(f"Riwayat pelatihan disimpan di {history_path}")

    # Evaluasi pada test set jika ada
    if 'X_test' in data and data['X_test'].shape[0] > 0:
        X_test, y_test = data['X_test'], data['y_test']
        test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(batch_size)
        
        logging.info("Mengevaluasi model pada set test...")
        best_model = tf.keras.models.load_model(checkpoint_path, compile=False)
        
        if config['optimization'].get('pruning', {}).get('enable', False):
            try:
                best_model = tfmot.sparsity.keras.strip_pruning(best_model)
            except Exception as e:
                logging.warning(f"Tidak dapat strip pruning dari model yang dimuat untuk evaluasi: {e}.")

        best_model.compile(optimizer=optimizer, loss=config['training']['loss_function'], metrics=['mae', 'mse'])
        test_loss, test_mae, test_mse = best_model.evaluate(test_dataset, verbose=0)
        logging.info(f"Evaluasi Set Test - Loss: {test_loss:.4f}, MAE: {test_mae:.4f}, MSE: {test_mse:.4f}")
        
        with open(os.path.join(metrics_output_dir_colab, "test_evaluation.txt"), "w") as f:
            f.write(f"Test Loss: {test_loss}\n")
            f.write(f"Test MAE: {test_mae}\n")
            f.write(f"Test MSE: {test_mse}\n")
    else:
        logging.info("Tidak ada data test untuk evaluasi.")

    return final_model_path

# Fungsi untuk optimasi model (disesuaikan untuk path Colab)
def optimize_model_colab(config, trained_model_path, representative_dataset_source_path, quantized_model_output_dir_colab, num_features_from_preprocessing):
    logging.info("Memulai optimasi model...")
    opt_config = config['optimization']

    model = tf.keras.models.load_model(trained_model_path)
    if model is None:
        logging.error(f"Gagal memuat model dari {trained_model_path}")
        return

    os.makedirs(quantized_model_output_dir_colab, exist_ok=True)

    if opt_config.get('quantization', {}).get('enable', False):
        quant_type = opt_config['quantization'].get('quant_type', 'int8').lower()
        logging.info(f"Menerapkan {quant_type} quantization...")
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        if quant_type == "int8":
            if opt_config['quantization'].get('use_representative_dataset', True):
                def representative_dataset_gen():
                    logging.info(f"Memuat dataset representatif dari sumber: {representative_dataset_source_path}")
                    try:
                        data_rep = np.load(representative_dataset_source_path)
                        cal_data_key = None
                        if 'X_val' in data_rep and data_rep['X_val'].shape[0] > 0: cal_data_key = 'X_val'
                        elif 'X_train' in data_rep and data_rep['X_train'].shape[0] > 0: cal_data_key = 'X_train'
                        
                        if cal_data_key:
                            cal_data_full = data_rep[cal_data_key]
                            num_cal_samples = min(cal_data_full.shape[0], opt_config['quantization'].get('num_calibration_samples', 100))
                            indices = np.random.choice(cal_data_full.shape[0], num_cal_samples, replace=False)
                            cal_data_subset = cal_data_full[indices]
                            logging.info(f"Menggunakan {cal_data_subset.shape[0]} sampel dari '{cal_data_key}' untuk dataset representatif.")
                            for i in range(cal_data_subset.shape[0]):
                                yield [cal_data_subset[i:i+1].astype(np.float32)]
                        else:
                            logging.warning(f"Tidak ada data ('X_train', 'X_val') yang cocok di {representative_dataset_source_path}. Menggunakan data random.")
                            input_spec_shape = (opt_config['quantization'].get('num_calibration_samples', 100),
                                                config['dataset']['sequence_length'],
                                                num_features_from_preprocessing) # Gunakan jumlah fitur dari preprocessing
                            cal_data_random = np.random.rand(*input_spec_shape).astype(np.float32)
                            for i in range(cal_data_random.shape[0]):
                                yield [cal_data_random[i:i+1].astype(np.float32)]
                    except Exception as e_rep:
                        logging.error(f"Error saat memuat/memproses dataset representatif: {e_rep}. Fallback ke data random.")
                        # Fallback jika np.load gagal atau kunci tidak ada
                        input_spec_shape_fallback = (opt_config['quantization'].get('num_calibration_samples', 100),
                                                     config['dataset']['sequence_length'],
                                                     num_features_from_preprocessing)
                        cal_data_random_fallback = np.random.rand(*input_spec_shape_fallback).astype(np.float32)
                        for i in range(cal_data_random_fallback.shape[0]):
                             yield [cal_data_random_fallback[i:i+1].astype(np.float32)]
                
                converter.representative_dataset = representative_dataset_gen
                if opt_config['quantization'].get('int8_fallback_float16', False):
                     converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8, tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_FLOAT16]
                else:
                    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            else:
                 logging.warning("INT8 quantization tanpa dataset representatif.")
                 converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        elif quant_type == "float16":
            converter.target_spec.supported_types = [tf.float16]

        try:
            tflite_quant_model = converter.convert()
            quantized_model_filename = f"model_quant_{quant_type}.tflite"
            quantized_model_path = os.path.join(quantized_model_output_dir_colab, quantized_model_filename)
            with open(quantized_model_path, 'wb') as f: f.write(tflite_quant_model)
            logging.info(f"Model terkuantisasi ({quant_type}) disimpan ke {quantized_model_path}")
            logging.info(f"Ukuran model asli: {os.path.getsize(trained_model_path) / (1024*1024):.2f} MB")
            logging.info(f"Ukuran model terkuantisasi: {len(tflite_quant_model) / (1024*1024):.2f} MB")
            return quantized_model_path # Kembalikan path model terkuantisasi
        except Exception as e_quant:
            logging.error(f"Error selama konversi TFLite ({quant_type}): {e_quant}")
            if quant_type == "int8" and opt_config['quantization'].get('fallback_to_fp16_on_error', True):
                logging.info("Mencoba quantization float16 sebagai fallback...")
                try:
                    converter_fp16 = tf.lite.TFLiteConverter.from_keras_model(model)
                    converter_fp16.optimizations = [tf.lite.Optimize.DEFAULT]
                    converter_fp16.target_spec.supported_types = [tf.float16]
                    tflite_fp16_model = converter_fp16.convert()
                    quantized_fp16_model_path = os.path.join(quantized_model_output_dir_colab, "model_quant_fp16_fallback.tflite")
                    with open(quantized_fp16_model_path, 'wb') as f: f.write(tflite_fp16_model)
                    logging.info(f"Model terkuantisasi float16 (fallback) disimpan ke {quantized_fp16_model_path}")
                    return quantized_fp16_model_path
                except Exception as e_fp16_fallback:
                    logging.error(f"Error selama konversi TFLite FP16 (fallback): {e_fp16_fallback}")
            return None # Gagal kuantisasi
    return None # Kuantisasi tidak diaktifkan

# Fungsi untuk deployment (placeholder)
def deploy_model_colab(config, model_path, deployment_info_path_colab):
    logging.info(f"Memulai deployment model (placeholder)...")
    if model_path is None or not os.path.exists(model_path):
        logging.error(f"Model path tidak valid atau model tidak ada: {model_path}. Deployment dibatalkan.")
        return

    logging.info(f"Model untuk dideploy: {model_path}")
    os.makedirs(os.path.dirname(deployment_info_path_colab), exist_ok=True)
    with open(deployment_info_path_colab, 'w') as f:
        f.write(f"Model dideploy dari: {model_path}\n")
        f.write(f"Timestamp deployment: {datetime.datetime.now()}\n")
        f.write(f"Target environment (konseptual): {config.get('deployment', {}).get('target_env', 'N/A')}\n")
    logging.info(f"Informasi deployment disimpan ke {deployment_info_path_colab}")
    logging.info("Placeholder deployment model selesai.")

# ---------------------------------------------------------------------------
# BAGIAN 5: EKSEKUSI PIPELINE DI COLAB
# ---------------------------------------------------------------------------

# Muat Konfigurasi Global
CONFIG_DATA = load_config_from_string(CONFIG_YAML_STRING)

# Path di lingkungan Colab
COLAB_BASE_DIR = "/content/quantai_colab_workspace"
RAW_DATA_DIR_COLAB = os.path.join(COLAB_BASE_DIR, "raw_data")
PROCESSED_DATA_DIR_COLAB = os.path.join(COLAB_BASE_DIR, "processed_data")
MODEL_OUTPUT_DIR_COLAB = os.path.join(COLAB_BASE_DIR, "models", "trained_model")
METRICS_OUTPUT_DIR_COLAB = os.path.join(COLAB_BASE_DIR, "reports", "metrics")
QUANTIZED_MODEL_OUTPUT_DIR_COLAB = os.path.join(COLAB_BASE_DIR, "models", "quantized_model")
DEPLOYMENT_INFO_DIR_COLAB = os.path.join(COLAB_BASE_DIR, "deploy")

# Buat direktori jika belum ada
os.makedirs(RAW_DATA_DIR_COLAB, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR_COLAB, exist_ok=True)
os.makedirs(MODEL_OUTPUT_DIR_COLAB, exist_ok=True)
os.makedirs(METRICS_OUTPUT_DIR_COLAB, exist_ok=True)
os.makedirs(QUANTIZED_MODEL_OUTPUT_DIR_COLAB, exist_ok=True)
os.makedirs(DEPLOYMENT_INFO_DIR_COLAB, exist_ok=True)

# Nama file CSV yang diharapkan
TARGET_CSV_FILENAME = "XAUUSD_M5.csv"
PATH_TO_UPLOADED_CSV = os.path.join(RAW_DATA_DIR_COLAB, TARGET_CSV_FILENAME) # Path setelah diupload dan dipindah

def run_full_pipeline_colab():
    logging.info("Memulai eksekusi pipeline lengkap di Colab...")

    # 1. Upload File CSV
    logging.info(f"Silakan upload file '{TARGET_CSV_FILENAME}'.")
    logging.info("Pastikan file tidak memiliki header dan menggunakan ';' sebagai pemisah.")
    
    # Hapus file lama jika ada untuk memastikan upload baru yang digunakan
    if os.path.exists(TARGET_CSV_FILENAME): # Cek di /content/
        os.remove(TARGET_CSV_FILENAME)
    if os.path.exists(PATH_TO_UPLOADED_CSV): # Cek di target dir
        os.remove(PATH_TO_UPLOADED_CSV)

    uploaded = files.upload()

    if TARGET_CSV_FILENAME not in uploaded:
        logging.error(f"File '{TARGET_CSV_FILENAME}' tidak ditemukan dalam file yang diupload. Pastikan nama file benar.")
        return
    
    # Pindahkan file yang diupload ke direktori raw_data Kita
    # File yang diupload oleh files.upload() ada di /content/
    os.rename(TARGET_CSV_FILENAME, PATH_TO_UPLOADED_CSV)
    logging.info(f"File '{TARGET_CSV_FILENAME}' telah dipindahkan ke '{PATH_TO_UPLOADED_CSV}'")


    # 2. Preprocessing Data
    logging.info("Tahap 1: Preprocessing Data")
    if not os.path.exists(PATH_TO_UPLOADED_CSV):
        logging.error(f"File CSV yang diharapkan '{PATH_TO_UPLOADED_CSV}' tidak ditemukan. Hentikan preprocessing.")
        return
        
    processed_data_file_path, actual_features = preprocess_data(CONFIG_DATA, PATH_TO_UPLOADED_CSV, PROCESSED_DATA_DIR_COLAB)
    if processed_data_file_path is None:
        logging.error("Preprocessing gagal. Menghentikan pipeline.")
        return
    num_actual_features = len(actual_features) if actual_features else 0
    if num_actual_features == 0:
        logging.error("Tidak ada fitur yang diekstrak selama preprocessing. Menghentikan pipeline.")
        return

    # 3. Training Model
    logging.info("Tahap 2: Training Model")
    trained_model_file_path = train_model_colab(CONFIG_DATA, processed_data_file_path, MODEL_OUTPUT_DIR_COLAB, METRICS_OUTPUT_DIR_COLAB)
    if trained_model_file_path is None:
        logging.error("Training model gagal. Menghentikan pipeline.")
        return

    # 4. Optimasi Model (Quantization)
    logging.info("Tahap 3: Optimasi Model")
    # representative_dataset_source_path adalah path ke file .npz yang berisi X_train/X_val
    quantized_model_file_path = optimize_model_colab(CONFIG_DATA, trained_model_file_path, processed_data_file_path, QUANTIZED_MODEL_OUTPUT_DIR_COLAB, num_actual_features)
    if quantized_model_file_path:
        logging.info(f"Model berhasil dikuantisasi dan disimpan di: {quantized_model_file_path}")
    else:
        logging.warning("Optimasi model (kuantisasi) tidak menghasilkan file output atau dilewati.")


    # 5. Deployment (Placeholder)
    logging.info("Tahap 4: Deployment Model (Placeholder)")
    model_to_deploy = quantized_model_file_path if quantized_model_file_path and os.path.exists(quantized_model_file_path) else trained_model_file_path
    deployment_receipt_path = os.path.join(DEPLOYMENT_INFO_DIR_COLAB, "deployment_receipt.txt")
    deploy_model_colab(CONFIG_DATA, model_to_deploy, deployment_receipt_path)

    logging.info("Pipeline lengkap QuantAI Colab selesai dieksekusi.")
    logging.info(f"Hasil dapat ditemukan di direktori: {COLAB_BASE_DIR}")
    logging.info(f"Model terlatih: {MODEL_OUTPUT_DIR_COLAB}")
    logging.info(f"Model terkuantisasi (jika ada): {QUANTIZED_MODEL_OUTPUT_DIR_COLAB}")
    logging.info(f"Metrik dan log: {METRICS_OUTPUT_DIR_COLAB}")
    logging.info(f"Data terproses: {PROCESSED_DATA_DIR_COLAB}")

# Untuk menjalankan pipeline, panggil fungsi berikut di sel Colab:
# run_full_pipeline_colab()
