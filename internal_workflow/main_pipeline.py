# -*- coding: utf-8 -*-
"""
Kode Pipeline QuantAI End-to-End dalam Satu File.

Kode ini mengimplementasikan seluruh alur kerja ML untuk model QuantAI,
dari pemuatan data mentah hingga penyimpanan model terlatih dan hasilnya.
Dirancang untuk dijalankan di lingkungan standar Python,
termasuk runner GitHub Actions.

Fitur Termasuk:
- Membaca konfigurasi dari file YAML (path diberikan sebagai argumen command-line).
- Setup TensorFlow dan Hardware (GPU/Mixed Precision).
- Pemuatan data (CSV/JSON) dan preprocessing awal (termasuk Feature Engineering aritmatika),
  termasuk penanganan kolom 'Date' untuk pengurutan dan output.
- Pembuatan fitur teks deskriptif.
- Scaling data numerik.
- Pembuatan pipeline tf.data yang efisien (windowing, batching, caching, prefetching),
  termasuk penanganan data teks dan target yang selaras.
- Definisi atau pemuatan model QuantAI (Functional API dengan GRU, LayerNorm, Dense,
  potensi input teks).
- Kompilasi model.
- Pelatihan model (mode initial_train atau incremental_learn) dengan callbacks.
- Evaluasi model akhir.
- Penyimpanan model terlatih (.h5), scaler, file kosakata teks, hasil evaluasi, dan prediksi
  (termasuk kolom 'Date' yang selaras).
- Penggunaan API aritmatika, baca/tulis, dan fondasi belajar mandiri/hybrid.

Untuk menjalankan di GitHub Actions:
1. Pastikan kode ini, quantai_config.yaml, requirements.txt ada di repositori.
2. Pastikan data input ada di path yang sesuai di repositori.
3. Buat file workflow GitHub Actions (.github/workflows/train_pipeline.yml).
4. Workflow akan menjalankan script ini dengan path ke file config sebagai argumen.
"""

import os
import yaml
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import joblib
import logging
import json
import argparse
import traceback # Import traceback module untuk logging error detail

# --- Konfigurasi Global ---
# Path ke file konfigurasi akan diberikan melalui argumen command-line

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Fungsi Pembantu ---

def generate_analysis_text(row):
    """
    Menghasilkan string teks deskriptif berdasarkan nilai numerik baris data.
    Menggunakan operasi aritmatika dan perbandingan.
    """
    text = []
    # Contoh logika sederhana menggunakan nilai Pivot dan harga
    pivot = row.get('Pivot', None)
    close = row.get('Close', None)
    r1 = row.get('R1', None)
    s1 = row.get('S1', None)
    high = row.get('High', None)
    low = row.get('Low', None)

    # Pastikan nilai ada dan bukan NaN sebelum perbandingan/aritmatika
    # Menggunakan Aritmatika untuk perbandingan
    if pivot is not None and close is not None and pd.notna(pivot) and pd.notna(close):
        if close > pivot:
            text.append(f"Close {close:.2f} di atas Pivot {pivot:.2f}, bullish.")
        elif close < pivot:
            text.append(f"Close {close:.2f} di bawah Pivot {pivot:.2f}, bearish.")
        else:
            text.append(f"Close {close:.2f} di sekitar Pivot {pivot:.2f}, netral.")

    if r1 is not None and high is not None and pd.notna(r1) and pd.notna(high):
         # Menggunakan Aritmatika untuk perbandingan
        if high > r1:
            text.append(f"High {high:.2f} menembus R1 {r1:.2f}, potensi naik.")

    if s1 is not None and low is not None and pd.notna(s1) and pd.notna(low):
         # Menggunakan Aritmatika untuk perbandingan
        if low < s1:
            text.append(f"Low {low:.2f} menembus S1 {s1:.2f}, potensi turun.")

    # Tambahkan logika lain sesuai kebutuhan (misal: R2-R5, S2-S5, volatilitas)

    return " ".join(text) if text else "Analisis tidak tersedia." # Gabungkan semua frasa


# Menggunakan @tf.function untuk mengompilasi fungsi pemrosesan teks di pipeline tf.data
@tf.function
def process_text_for_model(text_tensor, max_sequence_length, vocabulary_table):
    """
    Memproses tensor string teks menjadi representasi numerik untuk model.
    Contoh: tokenisasi, lookup ID, padding.
    """
    # Menggunakan API TensorFlow Strings
    tokens = tf.strings.split(text_tensor) # Memecah string menjadi token

    # Menggunakan API TensorFlow Lookup
    # Mengonversi token string menjadi ID integer
    # Use a default value for OOV tokens, e.g., the last index (corresponding to OOV bucket)
    # If using num_oov_buckets=1, OOV tokens map to vocabulary_size - 1.
    # If mask_zero=True is used in Embedding, ID 0 is reserved for padding.
    # Our vocabulary_table maps tokens to 1..vocabulary_size-num_oov_buckets. OOV maps to last index.
    # The lookup function handles OOV mapping automatically.
    token_ids = vocabulary_table.lookup(tokens)


    # Padding atau pemotongan sekuens token ID
    # tf.RaggedTensor.to_tensor pads to the max length in the current batch.
    # If we need a fixed length (max_sequence_length) regardless of batch, we need explicit padding/slicing.
    # Let's pad to the specified max_sequence_length.
    current_len = tf.shape(token_ids)[0]
    padded_token_ids = tf.pad(token_ids, [[0, tf.maximum(0, max_sequence_length - current_len)]], constant_values=0) # Pad with 0
    padded_token_ids = padded_token_ids[:max_sequence_length] # Slice to max_sequence_length


    # Menggunakan API Aritmatika jika ada operasi numerik pada ID token
    # Contoh sederhana: menambahkan 1 ke setiap ID (tidak umum, hanya ilustrasi)
    # padded_token_ids = tf.add(padded_token_ids, 1)

    return padded_token_ids # Output shape: (max_sequence_length,)


def build_quantai_model(config, numeric_input_shape, text_input_shape, num_target_features, vocabulary_size=None):
    """
    Membangun arsitektur model QuantAI menggunakan Functional API.
    Mendukung input numerik dan teks.
    """
    # Menggunakan API Keras Layers: Input
    numeric_input = tf.keras.layers.Input(shape=numeric_input_shape, name='numeric_input')

    # Jalur Pemrosesan Numerik
    x_numeric = numeric_input
    # Menggunakan API Keras Layers: GRU
    # Set return_sequences=True if feeding into Conv1D or another time-distributed layer
    # Set return_sequences=False if feeding directly into Dense or Global Pooling
    # Based on original structure (GRU -> GlobalAveragePooling -> Dense), return_sequences=True
    gru_units = config['model']['architecture'].get('gru_units', 64) # Use .get with default
    dropout_rate = config['model']['architecture'].get('dropout_rate', 0.2) # Use .get with default
    embedding_dim = config['model']['architecture'].get('embedding_dim', 32) # Use .get with default

    x_numeric = tf.keras.layers.GRU(gru_units, return_sequences=True)(x_numeric)
    # Menggunakan API Keras Layers: LayerNormalization
    x_numeric = tf.keras.layers.LayerNormalization()(x_numeric)
    # Menggunakan API Keras Layers: Dropout
    x_numeric = tf.keras.layers.Dropout(dropout_rate)(x_numeric)

    # Opsi: Tambahkan Conv1D untuk pola lokal (TCN-inspired element)
    if config['model']['architecture'].get('use_conv1d', False):
         # Menggunakan API Keras Layers: Conv1D
        conv1d_filters = config['model']['architecture'].get('conv1d_filters', 32)
        conv1d_kernel_size = config['model']['architecture'].get('conv1d_kernel_size', 3)
        conv1d_dilation_rate = config['model']['architecture'].get('conv1d_dilation_rate', 1)
        x_numeric = tf.keras.layers.Conv1D(
            filters=conv1d_filters,
            kernel_size=conv1d_kernel_size,
            activation='relu', # Menggunakan API Aritmatika di dalam layer
            padding='causal', # Penting untuk deret waktu
            dilation_rate=conv1d_dilation_rate
        )(x_numeric)
         # Menggunakan API Keras Layers: LayerNormalization
        x_numeric = tf.keras.layers.LayerNormalization()(x_numeric)
         # Menggunakan API Keras Layers: Dropout
        x_numeric = tf.keras.layers.Dropout(dropout_rate)(x_numeric)

    # Merangkum sekuens numerik
    # Pool or Flatten the sequence output before the final Dense layer
    x_numeric = tf.keras.layers.GlobalAveragePooling1D()(x_numeric)


    # Jalur Pemrosesan Teks (jika digunakan)
    use_text_input = config.get('use_text_input', False)
    if use_text_input and text_input_shape is not None and vocabulary_size is not None and vocabulary_size > 1: # vocab size > 1 because 0 is padding, 1 is OOV if nothing else
        # Input teks adalah sekuens ID integer shape (window_size, max_text_sequence_length)
        # Menggunakan API Keras Layers: Input
        text_input = tf.keras.layers.Input(shape=text_input_shape, name='text_input', dtype=tf.int64) # Input ID integer

        # Menggunakan API Keras Layers: Embedding
        # Masking 0 ID jika 0 digunakan untuk padding
        x_text = tf.keras.layers.Embedding(input_dim=vocabulary_size, # Vocabulary size termasuk OOV bucket + padding 0
                                           output_dim=embedding_dim,
                                           mask_zero=True # Penting jika 0 digunakan untuk padding
                                          )(text_input) # Output shape (window_size, max_text_sequence_length, embedding_dim)

        # Pool embeddings across max_text_sequence_length first, then process resulting sequence (window_size, embedding_dim) with GRU
        x_text = tf.keras.layers.GlobalAveragePooling1D()(x_text) # Pool across max_text_sequence_length. Output shape (window_size, embedding_dim)
         # Menggunakan API Keras Layers: GRU (to process the sequence of pooled text embeddings)
        x_text = tf.keras.layers.GRU(gru_units // 2)(x_text) # Process sequence across window_size. Output shape (units/2)
         # Menggunakan API Keras Layers: LayerNormalization
        x_text = tf.keras.layers.LayerNormalization()(x_text)
         # Menggunakan API Keras Layers: Dropout
        x_text = tf.keras.layers.Dropout(dropout_rate)(x_text)


        # Menggabungkan jalur numerik dan teks
         # Menggunakan API Keras Layers: Concatenate
        combined = tf.keras.layers.Concatenate()([x_numeric, x_text])
    elif use_text_input:
         logger.warning("Fitur teks diaktifkan tetapi tidak dapat membuat jalur teks model (input shape/vocab size issues). Menggunakan hanya jalur numerik.")
         combined = x_numeric # Fallback to numeric only if text path setup fails
    else:
        combined = x_numeric # Hanya jalur numerik jika teks tidak digunakan

    # Layer Akhir
     # Menggunakan API Keras Layers: Dense
    output_layer = tf.keras.layers.Dense(num_target_features, name='output')(combined) # 3 unit untuk HLC

    # Menggunakan API Keras Model: Functional API
    # Define inputs based on whether text path was actually included
    if use_text_input and text_input_shape is not None and vocabulary_size is not None and vocabulary_size > 1:
        model = tf.keras.Model(inputs=[numeric_input, text_input], outputs=output_layer)
    else:
        model = tf.keras.Model(inputs=numeric_input, outputs=output_layer)

    return model

# --- Main Pipeline Function ---

def run_pipeline(config):
    """
    Menjalankan seluruh pipeline ML berdasarkan konfigurasi.
    Ini adalah inti dari script satu file.
    """
    logger.info("Memulai pipeline QuantAI...")

    # --- Langkah 0.5: Setup Hardware ---
    try:
        logger.info("Mengkonfigurasi TensorFlow dan Hardware...")
        physical_devices_gpu = tf.config.list_physical_devices('GPU')
        if physical_devices_gpu:
            logger.info(f"Ditemukan GPU: {len(physical_devices_gpu)}")
            tf.config.set_visible_devices(physical_devices_gpu, 'GPU')
            for gpu in physical_devices_gpu:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info("Memory growth diaktifkan untuk GPU.")
            try: # Mixed precision might not be supported on all GPUs or TF versions
                policy = tf.keras.mixed_precision.Policy('mixed_float16')
                tf.keras.mixed_precision.set_global_policy(policy)
                logger.info("Mixed precision (mixed_float16) diaktifkan.")
            except Exception as mp_e:
                 logger.warning(f"Gagal mengaktifkan mixed precision: {mp_e}")
                 logger.warning("Melanjutkan tanpa mixed precision.")
        else:
            logger.warning("Tidak ada GPU yang terdeteksi. Menggunakan CPU.")
            physical_devices_cpu = tf.config.list_physical_devices('CPU')
            tf.config.set_visible_devices(physical_devices_cpu, 'CPU')

        logger.info(f"Menyetel seed: {config['seed']}")
        np.random.seed(config['seed'])
        tf.random.set_seed(config['seed'])
        logger.info("Setup hardware dan seed selesai.")

    except Exception as e:
        logger.warning(f"Gagal mengkonfigurasi hardware TensorFlow: {e}")


    # --- Langkah 1: Pemuatan Data & Preprocessing Awal ---
    logger.info("Langkah 1: Memuat data dan preprocessing awal...")
    df = None # Inisialisasi df
    original_dates_for_alignment = None # Store dates for aligning predictions
    date_column_name = config['data'].get('date_column', 'Date') # Use configurable date column name
    feature_cols_text = [] # Initialize here

    try:
        data_path = config['data']['raw_path']
        if not os.path.exists(data_path):
             logger.error(f"File data tidak ditemukan di {os.path.abspath(data_path)}. Pastikan path benar dan file ada.")
             return # Hentikan pipeline jika file data tidak ada

        if data_path.endswith('.csv'):
            df = pd.read_csv(data_path)
        elif data_path.endswith('.json'):
            df = pd.read_json(data_path)
        else:
            raise ValueError("Format data tidak didukung. Gunakan .csv atau .json.")

        logger.info(f"Data mentah dimuat dari {data_path}. Jumlah baris: {len(df)}")

        # Pastikan data terurut berdasarkan Date jika ada
        if date_column_name in df.columns:
            df[date_column_name] = pd.to_datetime(df[date_column_name]) # Konversi ke datetime
            df.sort_values(by=date_column_name, inplace=True)
            logger.info(f"Data diurutkan berdasarkan kolom '{date_column_name}'.")
            # Simpan kolom 'Date' asli untuk alignment prediksi nanti
            original_dates_for_alignment = df[date_column_name].copy()
        else:
            logger.warning(f"Kolom '{date_column_name}' tidak ditemukan di data mentah. Prediksi tidak akan bisa di-align dengan tanggal asli.")
            date_column_name = None # Set to None if date column not found


        # Pembersihan data awal
        initial_rows = len(df)
        # Hapus baris dengan NaN di kolom harga atau kolom fitur numerik yang *dipastikan* ada
        cols_to_check_nan = list(config['data']['feature_cols_numeric']) # Start with features from config

        # Option to include date in NaN check
        if date_column_name is not None and config['data'].get('include_date_in_nan_check', False):
             if date_column_name not in cols_to_check_nan: # Avoid adding twice if Date is somehow in feature_cols_numeric
                cols_to_check_nan.append(date_column_name)
                logger.info(f"Menyertakan kolom '{date_column_name}' dalam pengecekan NaN awal.")
        elif date_column_name is None and config['data'].get('include_date_in_nan_check', False):
             logger.warning("include_date_in_nan_check true tetapi kolom Date tidak ditemukan.")

        # Ensure OHLC are in cols_to_check_nan if they are features or needed for pivots
        ohlc_cols_basic = ['Open', 'High', 'Low', 'Close']
        for col in ohlc_cols_basic:
            if col in df.columns and col not in cols_to_check_nan:
                 cols_to_check_nan.append(col) # Ensure OHLC are checked if present

        # Perform initial dropna on specified cols
        if cols_to_check_nan: # Only drop if there are columns to check
            df.dropna(subset=cols_to_check_nan, inplace=True)
            if len(df) < initial_rows:
                logger.warning(f"Menghapus {initial_rows - len(df)} baris dengan NaN di kolom utama: {cols_to_check_nan}")
        else:
            logger.warning("Tidak ada kolom yang ditentukan untuk pengecekan NaN awal.")


        # Pastikan kolom OHLC ada untuk menghitung Pivot Points
        ohlc_cols = ['Open', 'High', 'Low', 'Close']
        if all(col in df.columns for col in ohlc_cols):
             # Feature Engineering (Menggunakan Aritmatika)
             logger.info("Menghitung Pivot Points...")
             df['Pivot'] = (df['High'] + df['Low'] + df['Close']) / 3
             df['R1'] = 2 * df['Pivot'] - df['Low']
             df['S1'] = 2 * df['Pivot'] - df['High']
             df['R2'] = df['Pivot'] + (df['High'] - df['Low'])
             df['S2'] = df['Pivot'] - (df['High'] - df['Low'])
             df['R3'] = df['Pivot'] + 2 * (df['High'] - df['Low'])
             df['S3'] = df['Pivot'] - 2 * (df['High'] - df['Low'])
             # Tambahkan R4/R5, S4/S5 jika perlu
             logger.info("Perhitungan Pivot Points selesai.")
        else:
             logger.warning(f"Kolom OHLC tidak lengkap ({ohlc_cols}) setelah pembersihan data. Tidak dapat menghitung Pivot Points.")


        # Menyiapkan Target (HLC Selanjutnya)
        window_size = config['parameter_windowing']['window_size']
        logger.info(f"Menyiapkan target HLC selanjutnya dengan shift -{window_size}...")
        target_cols = ['High_Next', 'Low_Next', 'Close_Next']
        # Ensure original OHLC cols exist before shifting
        if all(col in df.columns for col in ['High', 'Low', 'Close']):
            df['High_Next'] = df['High'].shift(-window_size)
            df['Low_Next'] = df['Low'].shift(-window_size)
            df['Close_Next'] = df['Close'].shift(-window_size)
            logger.info("Target HLC selanjutnya dibuat.")

            # Menghapus baris dengan NaN di target setelah shift
            initial_rows = len(df)
            if target_cols: # Only drop if target columns were created
                df.dropna(subset=target_cols, inplace=True)
                if len(df) < initial_rows:
                     logger.warning(f"Menghapus {initial_rows - len(df)} baris dengan NaN di target setelah shift (-{window_size}).")
            else:
                logger.warning("Kolom target tidak terdefinisi. Melewatkan dropna target.")

        else:
             logger.error("Kolom High, Low, Close tidak tersedia untuk membuat target. Tidak dapat melanjutkan.")
             return


        # Membuat Fitur Teks Deskriptif (Menulis Teks ke Kolom)
        if config.get('use_text_input', False):
            logger.info("Membuat fitur teks deskriptif...")
            # Check if columns needed for text generation exist AFTER all drops
            cols_needed_for_text = ['High', 'Low', 'Close'] + [col for col in df.columns if col.startswith(('Piv', 'R', 'S')) and col in df.columns] # Include calculated indicators only if they exist
            if all(col in df.columns for col in cols_needed_for_text):
                df['Analysis_Text'] = df.apply(generate_analysis_text, axis=1)
                feature_cols_text = ['Analysis_Text']
                if not df.empty:
                     logger.info(f"Contoh teks analisis: '{df['Analysis_Text'].iloc[0]}'")
                else:
                     logger.warning("DataFrame kosong setelah membuat teks analisis.")
            else:
                 logger.warning(f"Kolom yang dibutuhkan untuk generate_analysis_text tidak lengkap ({cols_needed_for_text}). Fitur teks tidak akan dibuat.")


        # Identifikasi Fitur Input dan Target
        # Pastikan kolom indikator yang dihitung juga masuk fitur numerik
        indicator_cols = [col for col in df.columns if col.startswith(('Piv', 'R', 'S'))]
        # Ambil feature_cols_numeric dari config, lalu tambahkan indicator_cols jika belum ada
        feature_cols_numeric = list(config['data']['feature_cols_numeric'])
        for col in indicator_cols:
            if col not in feature_cols_numeric and col in df.columns:
                feature_cols_numeric.append(col)

        # Remove date column from features if present in config
        if date_column_name is not None and date_column_name in feature_cols_numeric:
             feature_cols_numeric.remove(date_column_name)
             logger.warning(f"Kolom '{date_column_name}' dihapus dari fitur numerik karena biasanya tidak digunakan secara langsung.")


        # Filter DataFrame hanya untuk kolom fitur dan target yang relevan
        all_feature_target_cols = feature_cols_numeric + feature_cols_text + target_cols
        # Exclude date column explicitly if it exists and is not a feature
        if date_column_name is not None and date_column_name in all_feature_target_cols:
             # This case should not happen if we removed it from feature_cols_numeric, but double check
             all_feature_target_cols.remove(date_column_name)


        df_processed = df[all_feature_target_cols].copy()

        # Store the index splits based on the processed DataFrame length
        total_samples_processed = len(df_processed)
        train_split = config['data'].get('train_split', 0.8)
        val_split = config['data'].get('val_split', 0.1)

        train_size = int(np.floor(total_samples_processed * train_split))
        val_size = int(np.floor(total_samples_processed * val_split))
        test_size = total_samples_processed - train_size - val_size

        # Store test set starting index in the processed DataFrame
        test_split_start_idx_processed_df = train_size + val_size


        # Konversi ke NumPy Arrays
        if not df_processed.empty:
            data_numeric = df_processed[feature_cols_numeric].values.astype(np.float32)
            data_text = df_processed[feature_cols_text].values.flatten().astype(str) if feature_cols_text else np.array([], dtype=str)
            data_target = df_processed[target_cols].values.astype(np.float32)

            logger.info(f"Final data numerik shape: {data_numeric.shape}")
            logger.info(f"Final data teks shape: {data_text.shape}")
            logger.info(f"Final data target shape: {data_target.shape}")

        else:
            logger.error("DataFrame kosong setelah preprocessing. Tidak dapat melanjutkan.")
            return


        # Membagi Data (menggunakan slicing NumPy)
        if train_size <= 0 or val_size <= 0 or test_size <= 0:
             logger.error(f"Ukuran set data tidak memadai. Train: {train_size}, Val: {val_size}, Test: {test_size}. Sesuaikan split atau gunakan data lebih banyak.")
             return


        input_train_numeric = data_numeric[:train_size]
        input_val_numeric = data_numeric[train_size:train_size + val_size]
        input_test_numeric = data_numeric[train_size + val_size:]

        input_train_text = data_text[:train_size] if feature_cols_text else np.array([], dtype=str)
        input_val_text = data_text[train_size:train_size + val_size] if feature_cols_text else np.array([], dtype=str)
        input_test_text = data_text[train_size + val_size:] if feature_cols_text else np.array([], dtype=str)

        target_train = data_target[:train_size]
        target_val = data_target[train_size:train_size + val_size]
        target_test = data_target[train_size + val_size:]

        logger.info(f"Train set size: {len(input_train_numeric)}")
        logger.info(f"Validation set size: {len(input_val_numeric)}")
        logger.info(f"Test set size: {len(input_test_numeric)}")

    except Exception as e:
        logger.error(f"Error selama Langkah 1: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return


    # --- Langkah 2: Scaling Data & Pembuatan Pipeline tf.data ---
    logger.info("Langkah 2: Scaling data dan membuat pipeline tf.data...")
    scaler_input = None
    scaler_target = None
    vocabulary_table = None
    vocabulary_size = None
    max_text_sequence_length = config['data'].get('max_text_sequence_length', 20)

    try:
        if input_train_numeric.size > 0:
            scaler_input = MinMaxScaler()
            scaler_target = MinMaxScaler()

            scaled_input_train = scaler_input.fit_transform(input_train_numeric)
            scaled_target_train = scaler_target.fit_transform(target_train)

            scaled_input_val = scaler_input.transform(input_val_numeric)
            scaled_input_test = scaler_input.transform(input_test_numeric)

            scaled_target_val = scaler_target.transform(target_val)
            logger.info("Data numerik berhasil diskalakan.")
        else:
             logger.warning("Tidak ada data numerik untuk scaling.")
             scaled_input_train = input_train_numeric
             scaled_input_val = input_val_numeric
             scaled_input_test = input_test_numeric
             scaled_target_train = target_train
             scaled_target_val = target_val


        if config.get('use_text_input', False) and feature_cols_text and input_train_text.size > 0:
            logger.info("Membuat kosakata dari data pelatihan teks...")
            input_train_text_str = tf.constant(input_train_text.tolist(), dtype=tf.string)
            all_train_tokens_ragged = tf.strings.split(input_train_text_str)
            all_train_tokens_np = all_train_tokens_ragged.values.numpy()
            # Filter out empty strings before finding unique tokens
            unique_tokens_np = np.unique(all_train_tokens_np[all_train_tokens_np != b''])

            if unique_tokens_np.size > 0:
                keys = tf.constant(unique_tokens_np, dtype=tf.string)
                values = tf.range(1, tf.size(keys) + 1, dtype=tf.int64) # Start values from 1 to reserve 0 for padding
                num_oov_buckets = 1
                init = tf.lookup.KeyValueTensorInitializer(keys, values, key_dtype=tf.string, value_dtype=tf.int64)
                vocabulary_table = tf.lookup.StaticVocabularyTable(init, num_oov_buckets=num_oov_buckets)
                vocabulary_size = vocabulary_table.size().numpy() + 1 # Size includes unique keys + OOV bucket + PAD (index 0)
                logger.info(f"Ukuran kosakata teks (termasuk OOV dan PAD 0): {vocabulary_size}")
                logger.info(f"Contoh token unik (mapping value): {[t.decode('utf-8') for t in unique_tokens_np[:5]]} -> {vocabulary_table.lookup(tf.constant(unique_tokens_np[:5])).numpy()}")
            else:
                 logger.warning("Tidak ada token unik yang ditemukan di data teks pelatihan (setelah filter string kosong). Kosakata tidak dibuat.")
                 vocabulary_size = 2 # Minimum size for PAD (0) and OOV (1)
                 vocabulary_table = None

        elif config.get('use_text_input', False) and feature_cols_text:
             logger.warning("Fitur teks diaktifkan tetapi data pelatihan teks kosong. Kosakata tidak dibuat.")
             vocabulary_size = 2
             vocabulary_table = None
        elif config.get('use_text_input', False) and not feature_cols_text:
             logger.warning("Fitur teks diaktifkan di config, tetapi kolom teks ('Analysis_Text') tidak ditemukan atau tidak dibuat.")
             vocabulary_size = 2
             vocabulary_table = None
        else:
             logger.info("Fitur teks tidak digunakan.")
             vocabulary_size = 0
             vocabulary_table = None


        # Membuat tf.data.Dataset dari data yang sudah diproses
        if feature_cols_text:
            dataset_train_raw = tf.data.Dataset.from_tensor_slices(((scaled_input_train, input_train_text), scaled_target_train))
            dataset_val_raw = tf.data.Dataset.from_tensor_slices(((scaled_input_val, input_val_text), scaled_target_val))
            dataset_test_raw = tf.data.Dataset.from_tensor_slices(((scaled_input_test, input_test_text), target_test))
        else:
            dataset_train_raw = tf.data.Dataset.from_tensor_slices((scaled_input_train, scaled_target_train))
            dataset_val_raw = tf.data.Dataset.from_tensor_slices((scaled_input_val, scaled_target_val))
            dataset_test_raw = tf.data.Dataset.from_tensor_slices((scaled_input_test, target_test))


        # Fungsi untuk membuat window dan memproses elemen dataset
        @tf.function
        def process_window_elements(input_elements, target_window, use_text, max_seq_len, vocab_table):
             if use_text:
                 numeric_window, text_window = input_elements
             else:
                 numeric_window = input_elements
                 text_window = None

             final_target = target_window[-1]

             processed_text_window = None

             if use_text and text_window is not None and vocab_table is not None and max_seq_len > 0: # Check max_seq_len > 0
                 processed_text_window = tf.map_fn(
                     lambda text_elem: process_text_for_model(text_elem, max_seq_len, vocab_table),
                     text_window,
                     fn_output_signature=tf.int64
                 )
             elif use_text:
                 logger.warning("Text processing skipped: vocab_table or max_seq_len not valid.")


             if use_text and processed_text_window is not None:
                 return (numeric_window, processed_text_window), final_target
             else:
                 return numeric_window, final_target


        # Functions to create windowed datasets
        def create_window_dataset(dataset_raw, window_size, window_shift, drop_remainder):
             dataset_windowed = dataset_raw.window(size=window_size, shift=window_shift, drop_remainder=drop_remainder)
             dataset_flattened = dataset_windowed.flat_map(lambda window: window.batch(window_size))
             return dataset_flattened

        map_params = {
            'use_text': config.get('use_text_input', False) and feature_cols_text and vocabulary_size is not None and vocabulary_size > 1, # Only use text if configured AND column created AND vocab valid
            'max_seq_len': max_text_sequence_length,
            'vocab_table': vocabulary_table
        }

        window_shift = config['parameter_windowing'].get('window_shift', 1)


        dataset_train_windowed = create_window_dataset(dataset_train_raw, window_size, window_shift, True)
        dataset_train = dataset_train_windowed.map(
            lambda input_elements, target_window: process_window_elements(input_elements, target_window, **map_params),
            num_parallel_calls=tf.data.AUTOTUNE
        ).shuffle(config['data'].get('shuffle_buffer_size', 1000)
        ).batch(config['training'].get('batch_size', 32)
        ).cache().prefetch(tf.data.AUTOTUNE)
        logger.info(f"Pipeline train tf.data: {dataset_train.element_spec}")

        dataset_val_windowed = create_window_dataset(dataset_val_raw, window_size, window_shift, True)
        dataset_val = dataset_val_windowed.map(
             lambda input_elements, target_window: process_window_elements(input_elements, target_window, **map_params),
            num_parallel_calls=tf.data.AUTOTUNE
        ).batch(config['training'].get('batch_size', 32)
        ).cache().prefetch(tf.data.AUTOTUNE)
        logger.info(f"Pipeline val tf.data: {dataset_val.element_spec}")


        dataset_test_windowed = create_window_dataset(dataset_test_raw, window_size, window_shift, True)
        dataset_test = dataset_test_windowed.map(
            lambda input_elements, target_window: process_window_elements(input_elements, target_window, **map_params),
            num_parallel_calls=tf.data.AUTOTUNE
        ).batch(config['training'].get('batch_size', 32)
        ).prefetch(tf.data.AUTOTUNE)
        logger.info(f"Pipeline test tf.data: {dataset_test.element_spec}")


        logger.info("Pipeline tf.data selesai dibuat.")

    except Exception as e:
        logger.error(f"Error selama Langkah 2: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return


    # --- Langkah 3: Definisi & Kompilasi Model ---
    logger.info("Langkah 3: Mendefinisikan atau memuat model...")
    model = None
    numeric_input_shape = (window_size, len(feature_cols_numeric))
    text_input_shape = (window_size, max_text_sequence_length) if config.get('use_text_input', False) and feature_cols_text else None
    num_target_features = len(target_cols)

    try:
        if config['mode'] == 'initial_train':
            logger.info("Mode: initial_train. Mendefinisikan model baru.")
            model = build_quantai_model(config, numeric_input_shape, text_input_shape, num_target_features, vocabulary_size)

            logger.info("Mengkompilasi model.")
            optimizer = tf.keras.optimizers.Adam(learning_rate=config['training'].get('learning_rate', 0.001))
            loss_fn = tf.keras.losses.MeanAbsoluteError()
            metrics = [tf.keras.metrics.RootMeanSquaredError(), tf.keras.metrics.MeanAbsoluteError()]

            model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)

            logger.info("Model baru berhasil didefinisikan dan dikompilasi.")
            model.summary(print_fn=logger.info)

        elif config['mode'] in ['incremental_learn', 'predict_only']:
            load_path = config['model'].get('load_path')
            if not load_path:
                logger.error(f"Mode '{config['mode']}' dipilih, tetapi model.load_path tidak ada di konfigurasi.")
                return

            logger.info(f"Mode: {config['mode']}. Memuat model dari {load_path}")
            if not os.path.exists(load_path):
                 logger.error(f"File model tidak ditemukan di {os.path.abspath(load_path)}. Pastikan path benar dan model telah disimpan sebelumnya.")
                 return

            try:
                # Prioritize loading as H5
                if load_path.lower().endswith('.h5'):
                     model = tf.keras.models.load_model(load_path)
                     logger.info(f"Model berhasil dimuat dari format .h5: {load_path}.")
                else: # Try loading as SavedModel if not .h5
                     model = tf.saved_model.load(load_path)
                     logger.info(f"Model berhasil dimuat dari format SavedModel: {load_path}.")
                     if config['mode'] == 'incremental_learn' and not isinstance(model, tf.keras.Model):
                         logger.error("Model dimuat sebagai SavedModel tetapi bukan Keras Model. Tidak dapat melanjutkan pelatihan inkremental.")
                         model = None

            except Exception as e:
                logger.error(f"Gagal memuat model dari {load_path}: {e}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                model = None

            if model is not None:
                if config['mode'] == 'incremental_learn' and isinstance(model, tf.keras.Model):
                    logger.info("Mengkompilasi ulang model untuk incremental_learn.")
                    optimizer = tf.keras.optimizers.Adam(learning_rate=config['training'].get('learning_rate', 0.001))
                    loss_fn = tf.keras.losses.MeanAbsoluteError()
                    metrics = [tf.keras.metrics.RootMeanSquaredError(), tf.keras.metrics.MeanAbsoluteError()]
                    model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)
                    logger.info("Model berhasil dikompilasi ulang.")
                elif config['mode'] == 'incremental_learn' and not isinstance(model, tf.keras.Model):
                     pass # Error logged above
                else:
                    pass # predict_only mode, no recompilation needed
            else:
                 logger.error("Model object is None after attempting to load.")


        else:
            logger.error(f"Mode operasional tidak valid: {config['mode']}")
            return

    except Exception as e:
        logger.error(f"Error selama Langkah 3: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return

    if model is None:
         logger.error("Model tidak tersedia setelah Langkah 3. Menghentikan pipeline.")
         return


    # --- Langkah 4: Pelatihan Model (Belajar Mandiri/Hybrid/Otonom) ---
    if config['mode'] in ['initial_train', 'incremental_learn'] and isinstance(model, tf.keras.Model):
        logger.info(f"Langkah 4: Memulai pelatihan model dalam mode {config['mode']}...")
        try:
            output_dir = config['output']['base_dir']
            model_save_file = config['output'].get('model_save_file', 'saved_model/best_model.h5')
            model_save_path_full = os.path.join(output_dir, model_save_file)
            scaler_save_dir_full = os.path.join(output_dir, config['output'].get('scaler_subdir', 'scalers'))
            tensorboard_log_dir_full = os.path.join(output_dir, config['output'].get('tensorboard_log_dir', 'logs/tensorboard'))

            tf.io.gfile.makedirs(os.path.dirname(model_save_path_full))
            tf.io.gfile.makedirs(scaler_save_dir_full)
            tf.io.gfile.makedirs(tensorboard_log_dir_full)


            callbacks = [
                tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=config['training'].get('early_stopping_patience', 10), restore_best_weights=True),
                tf.keras.callbacks.ModelCheckpoint(filepath=model_save_path_full, monitor='val_loss', save_best_only=True, save_format='h5'), # <-- save_format='h5'
                tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=config['training'].get('lr_reduce_factor', 0.5), patience=config['training'].get('lr_reduce_patience', 5), min_lr=config['training'].get('min_lr', 0.0001)),
                tf.keras.callbacks.TensorBoard(log_dir=tensorboard_log_dir_full)
            ]

            epochs_to_run = config['training'].get('epochs', 100) if config['mode'] == 'initial_train' else config['training'].get('incremental_epochs', 10)

            logger.info(f"Melatih selama {epochs_to_run} epoch.")
            history = model.fit(
                dataset_train,
                epochs=epochs_to_run,
                validation_data=dataset_val,
                callbacks=callbacks
            )
            logger.info("Pelatihan selesai.")

            # Muat kembali model terbaik setelah pelatihan selesai (ModelCheckpoint menyimpannya)
            logger.info(f"Memuat model terbaik dari {model_save_path_full}")
            try:
                model = tf.keras.models.load_model(model_save_path_full)
                logger.info("Model terbaik berhasil dimuat setelah pelatihan.")
            except Exception as e:
                 logger.warning(f"Gagal memuat model terbaik setelah pelatihan dari {model_save_path_full}: {e}")
                 logger.warning(f"Traceback: {traceback.format_exc()}")
                 pass # Do not set model to None


            # --- Fondasi Belajar Mandiri/Hybrid/Otonom (Loop Kustom Opsional) ---
            # (Pseudocode placeholders remain)

        except Exception as e:
            logger.error(f"Error selama Langkah 4: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            # Lanjutkan ke langkah berikutnya meskipun ada error pelatihan

    elif config['mode'] in ['initial_train', 'incremental_learn']:
         logger.warning(f"Mode '{config['mode']}' dipilih tetapi model bukan instance tf.keras.Model. Melewatkan pelatihan.")


    # --- Langkah 5: Evaluation Model Akhir ---
    eval_results = None
    if config['mode'] in ['initial_train', 'incremental_learn'] and isinstance(model, tf.keras.Model) and hasattr(model, 'evaluate'):
        logger.info("Langkah 5: Mengevaluasi model akhir...")
        try:
            if 'dataset_test' in locals() and dataset_test is not None:
                logger.info(f"Dataset test size untuk evaluasi: {tf.data.Dataset.cardinality(dataset_test).numpy()} batches")
                eval_results = model.evaluate(dataset_test)
                if hasattr(model, 'metrics_names'):
                     logger.info(f"Hasil Evaluasi Akhir: {dict(zip(model.metrics_names, eval_results))}")
                else:
                     logger.info(f"Hasil Evaluasi Akhir (metrik tidak tersedia): {eval_results}")

            else:
                 logger.warning("Dataset test tidak tersedia. Tidak dapat melakukan evaluasi.")

        except Exception as e:
            logger.error(f"Error selama Langkah 5: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            # Lanjutkan ke langkah berikutnya meskipun ada error evaluasi
    elif config['mode'] in ['initial_train', 'incremental_learn']:
         logger.warning("Mode training tetapi model bukan Keras Model atau tidak memiliki metode evaluate. Melewatkan evaluasi.")
    else:
         logger.info("Mode predict_only atau tidak ada model Keras. Melewatkan evaluasi pelatihan.")


    # --- Langkah 6: Penyimpanan Aset & Hasil (Menulis) ---
    logger.info("Langkah 6: Menyimpan aset dan hasil...")
    try:
        output_dir = config['output']['base_dir']
        scaler_subdir = config['output'].get('scaler_subdir', 'scalers')
        eval_results_file = config['output'].get('eval_results_file', 'evaluation_metrics/eval_results.json')
        predictions_file = config['output'].get('predictions_file', 'predictions/predictions.csv')

        tf.io.gfile.makedirs(os.path.join(output_dir, scaler_subdir))
        tf.io.gfile.makedirs(os.path.dirname(os.path.join(output_dir, eval_results_file)))
        tf.io.gfile.makedirs(os.path.dirname(os.path.join(output_dir, predictions_file)))


        if config['mode'] in ['initial_train', 'incremental_learn']:
            model_save_file = config['output'].get('model_save_file', 'saved_model/best_model.h5')
            model_save_path_full = os.path.join(output_dir, model_save_file)
            if os.path.exists(model_save_path_full):
                 logger.info(f"Model terbaik dalam format .h5 sudah tersimpan di Langkah 4: {model_save_path_full}.")
            else:
                 logger.warning(f"Model file ({model_save_path_full}) tidak ditemukan setelah pelatihan. Model mungkin tidak tersimpan.")


            # Menyimpan Scaler
            logger.info(f"Menyimpan scaler ke {os.path.join(output_dir, scaler_subdir)}...")
            if 'scaler_input' in locals() and scaler_input is not None and 'scaler_target' in locals() and scaler_target is not None:
                 joblib.dump(scaler_input, os.path.join(output_dir, scaler_subdir, 'scaler_input.pkl'))
                 joblib.dump(scaler_target, os.path.join(output_dir, scaler_subdir, 'scaler_target.pkl'))
                 logger.info("Scaler berhasil disimpan.")
            else:
                 logger.warning("Scaler tidak tersedia. Tidak dapat menyimpan scaler.")

            # Menyimpan Kosakata Teks
            if config.get('use_text_input', False) and feature_cols_text and 'vocabulary_table' in locals() and vocabulary_table is not None:
                 vocab_file_path_config = config['output'].get('vocabulary_file', os.path.join(scaler_subdir, 'vocabulary.txt'))
                 vocab_file_path_full = os.path.join(output_dir, vocab_file_path_config)
                 tf.io.gfile.makedirs(os.path.dirname(vocab_file_path_full))
                 try:
                    if 'unique_tokens_np' in locals() and unique_tokens_np is not None and unique_tokens_np.size > 0: # Check if unique_tokens_np is not empty
                        with open(vocab_file_path_full, 'w', encoding='utf-8') as f:
                            for token in unique_tokens_np:
                                f.write(token.decode('utf-8') + '\n')
                        logger.info(f"Kosakata teks berhasil disimpan ke {vocab_file_path_full}.")
                    elif 'unique_tokens_np' in locals() and unique_tokens_np is not None:
                         logger.warning("Daftar token unik kosong. Tidak dapat menyimpan kosakata.")
                    else:
                         logger.warning("Daftar token unik untuk kosakata tidak tersedia. Tidak dapat menyimpan kosakata.")
                 except Exception as e:
                     logger.warning(f"Gagal menyimpan kosakata teks: {e}")
                     logger.warning(f"Traceback: {traceback.format_exc()}")


            # Menyimpan Hasil Evaluasi
            if eval_results is not None and model is not None and hasattr(model, 'metrics_names'):
                logger.info(f"Menyimpan hasil evaluasi ke {os.path.join(output_dir, eval_results_file)}...")
                eval_dict = dict(zip(model.metrics_names, eval_results))
                with open(os.path.join(output_dir, eval_results_file), 'w') as f:
                    json.dump(eval_dict, f, indent=4)
                logger.info("Hasil evaluasi berhasil disimpan.")
            elif eval_results is not None:
                 logger.warning("Nama metrik model tidak tersedia. Tidak dapat menyimpan hasil evaluasi dalam format dictionary.")


        # Menyimpan Prediksi (Output Hasil, termasuk Date)
        if config['output'].get('save_predictions', False) and model is not None and hasattr(model, 'predict'):
             if 'dataset_test' in locals() and dataset_test is not None:
                logger.info(f"Membuat dan menyimpan prediksi ke {os.path.join(output_dir, predictions_file)}...")
                predictions_scaled = model.predict(dataset_test)

                if 'scaler_target' in locals() and scaler_target is not None:
                    predictions_original_scale = scaler_target.inverse_transform(predictions_scaled)

                    df_predictions = pd.DataFrame(predictions_original_scale, columns=[f'{col}_Pred' for col in target_cols])

                    # --- Menambahkan Kolom Date ke Hasil Prediksi ---
                    if original_dates_for_alignment is not None and date_column_name is not None:
                         window_size = config['parameter_windowing']['window_size']
                         window_shift = config['parameter_windowing'].get('window_shift', 1)
                         num_predictions = len(df_predictions) # Number of predictions is number of test windows

                         # Calculate the starting index in the original_dates_for_alignment series
                         # This series has the same index as the df after initial dropna but before target shift.
                         # The first data point in the test split of df_processed corresponds to index test_split_start_idx_processed_df in df_processed.
                         # This index test_split_start_idx_processed_df also corresponds to the index in the original_dates_for_alignment series
                         # that marks the start of the data *after* initial dropna.
                         # The prediction for a window ending at original index `i` is for date at original index `i + window_size`.
                         # The first window in test starts at index test_split_start_idx_processed_df in df_processed.
                         # The original index corresponding to this first data point in df_processed is the index in original_dates_for_alignment at test_split_start_idx_processed_df.
                         # The prediction for this window is for the date at index test_split_start_idx_processed_df + window_size in original_dates_for_alignment.

                         date_indices_for_predictions_in_original = np.arange(0, num_predictions) * window_shift + (test_split_start_idx_processed_df + window_size)

                         # Ensure indices are within bounds of original_dates_for_alignment
                         valid_date_indices_in_original = date_indices_for_predictions_in_original[date_indices_for_predictions_in_original < len(original_dates_for_alignment)]

                         if len(valid_date_indices_in_original) == num_predictions:
                             predicted_dates = original_dates_for_alignment.iloc[valid_date_indices_in_original].reset_index(drop=True)
                             df_predictions.insert(0, date_column_name, predicted_dates)
                             logger.info(f"Kolom '{date_column_name}' berhasil ditambahkan ke hasil prediksi.")
                         else:
                              logger.warning(f"Jumlah prediksi ({num_predictions}) tidak sesuai dengan jumlah tanggal yang valid ({len(valid_date_indices_in_original)}). Tidak dapat menambahkan kolom Date dengan benar.")
                              logger.warning(f"Indeks tanggal asli yang dicoba: {date_indices_for_predictions_in_original}")
                              logger.warning(f"Panjang original_dates_for_alignment: {len(original_dates_for_alignment)}")

                    else:
                         logger.warning(f"Kolom '{date_column_name}' asli tidak tersedia atau tidak ditemukan. Tidak dapat menambahkan kolom Date ke prediksi.")


                    predictions_file_full = os.path.join(output_dir, predictions_file)
                    df_predictions.to_csv(predictions_file_full, index=False)
                    logger.info(f"Prediksi berhasil disimpan ke {predictions_file_full}.")

                elif model is None or not hasattr(model, 'predict'):
                     logger.warning("Model tidak tersedia atau tidak memiliki metode predict. Tidak dapat membuat prediksi.")
                elif ('scaler_target' in locals() and scaler_target is None) or ('scaler_target' not in locals()):
                     logger.warning("Scaler target tidak tersedia. Tidak dapat melakukan inverse transform atau menyimpan prediksi.")

             elif 'dataset_test' not in locals() or dataset_test is None:
                  logger.warning("Dataset test tidak tersedia. Tidak dapat membuat prediksi.")
             else:
                  logger.info("Prediksi tidak disimpan karena save_predictions diatur ke False di konfigurasi.")


        logger.info("Langkah 6 selesai.")

    except Exception as e:
        logger.error(f"Error selama Langkah 6: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")


    logger.info("Pipeline QuantAI selesai.")

# --- Eksekusi Pipeline ---

if __name__ == "__main__":
    # traceback already imported at the top

    parser = argparse.ArgumentParser(description='Run QuantAI ML Pipeline.')
    parser.add_argument('--config', type=str, required=True, help='Path to the YAML configuration file.')
    args = parser.parse_args()

    config_path = args.config

    config = None
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Konfigurasi berhasil dimuat dari {config_path}")

    except FileNotFoundError:
        logger.error(f"File konfigurasi tidak ditemukan di {config_path}. Pastikan path benar di workflow atau command-line.")
        exit(1)

    except yaml.YAMLError as e:
        logger.error(f"Error mem-parsing file konfigurasi YAML dari {config_path}: {e}")
        exit(1)

    if config:
        # --- Validasi Konfigurasi Esensial ---
        essential_keys = ['data', 'model', 'training', 'output', 'parameter_windowing', 'seed']
        missing_keys = [key for key in essential_keys if key not in config or config[key] is None]

        if missing_keys:
            logger.error(f"File konfigurasi ({config_path}) tidak lengkap atau rusak. Kunci utama yang hilang atau kosong: {missing_keys}")
            logger.error("Pastikan file konfigurasi YAML memiliki semua bagian utama (data, model, training, output, parameter_windowing, seed) dan tidak kosong.")
            exit(1)

        # --- Validasi Format Kunci Utama (Harus Dictionary) ---
        if not isinstance(config.get('data'), dict):
             logger.error("Kunci 'data' ada tetapi bukan dictionary. Pastikan format file konfigurasi benar.")
             exit(1)
        if not isinstance(config.get('model'), dict):
             logger.error("Kunci 'model' ada tetapi bukan dictionary. Pastikan format file konfigurasi benar.")
             exit(1)
        if not isinstance(config.get('training'), dict):
             logger.error("Kunci 'training' ada tetapi bukan dictionary. Pastikan format file konfigurasi benar.")
             exit(1)
        if not isinstance(config.get('output'), dict):
             logger.error("Kunci 'output' ada tetapi bukan dictionary. Pastikan format file konfigurasi benar.")
             exit(1)
        if not isinstance(config.get('parameter_windowing'), dict):
             logger.error("Kunci 'parameter_windowing' ada tetapi bukan dictionary. Pastikan format file konfigurasi benar.")
             exit(1)

        # --- Menambahkan Path Default jika tidak ada di config['output'] ---
        # Use .get() for sub-keys within 'output' as they might be optional in the config file itself
        output_config = config['output']
        base_dir = output_config.get('base_dir', '') # Default base_dir to empty string if missing

        if config.get('use_text_input', False): # Only add vocab file if text input is enabled
            if 'vocabulary_file' not in output_config:
                 scaler_subdir = output_config.get('scaler_subdir', 'scalers')
                 output_config['vocabulary_file'] = os.path.join(base_dir, scaler_subdir, 'vocabulary.txt')
                 logger.info(f"Menambahkan path default vocabulary_file: {output_config['vocabulary_file']}")

        if 'model_save_file' not in output_config:
             model_subdir = output_config.get('model_subdir', 'saved_model')
             output_config['model_save_file'] = os.path.join(base_dir, model_subdir, 'best_model.h5')
             logger.info(f"Menambahkan path default model_save_file: {output_config['model_save_file']}")


        # --- Validasi Mode dan load_path ---
        if config['mode'] in ['incremental_learn', 'predict_only']:
            if 'load_path' not in config['model'] or not config['model']['load_path']:
                 logger.error(f"Mode '{config['mode']}' dipilih, tetapi 'model.load_path' tidak ada/kosong di konfigurasi.")
                 exit(1)
            # Optional: Warning if load_path doesn't match expected save format (.h5)
            # Assumes model_save_file default/configured path dictates the format
            model_save_file_expected = output_config.get('model_save_file', 'saved_model/best_model.h5')
            if model_save_file_expected.endswith('.h5'):
                 if not config['model']['load_path'].lower().endswith('.h5'):
                      logger.warning(f"Mode '{config['mode']}' dipilih dan model akan disimpan sebagai .h5, tetapi model.load_path ({config['model']['load_path']}) tidak berakhir dengan .h5. Pastikan path memuat file .h5 yang benar.")
            # If the save format were SavedModel (directory)
            # else: # Assumes SavedModel is saved
            #      if config['model']['load_path'].lower().endswith('.h5'):
            #           logger.warning(f"Mode '{config['mode']}' dipilih dan model akan disimpan sebagai SavedModel, tetapi model.load_path ({config['model']['load_path']}) berakhir dengan .h5. Pastikan path memuat direktori SavedModel yang benar.")


        # --- Jalankan Pipeline Utama ---
        # base_dir is checked by essential_keys and its dict format is checked above
        run_pipeline(config)

    else:
        logger.error("Tidak ada konfigurasi yang tersedia setelah mencoba memuat file.")
        exit(1)
