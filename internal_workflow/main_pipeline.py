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
- Pemuatan data (CSV/JSON) dan preprocessing awal (termasuk Feature Engineering aritmatika).
- Pembuatan fitur teks deskriptif.
- Scaling data numerik.
- Pembuatan pipeline tf.data yang efisien (windowing, batching, caching, prefetching),
  termasuk penanganan data teks.
- Definisi atau pemuatan model QuantAI (Functional API dengan GRU, LayerNorm, Dense,
  potensi input teks).
- Kompilasi model.
- Pelatihan model (mode initial_train atau incremental_learn) dengan callbacks.
- Evaluasi model akhir.
- Penyimpanan model terlatih (.h5), scaler, hasil evaluasi, dan prediksi.
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
import json # Untuk menyimpan hasil evaluasi/prediksi dalam JSON
import argparse # Untuk membaca argumen command-line

# --- Konfigurasi Global ---
# Path ke file konfigurasi akan diberikan melalui argumen command-line
# CONFIG_PATH = 'quantai_config.yaml' # Tidak lagi hardcoded di sini

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
    token_ids = vocabulary_table.lookup(tokens)

    # Padding atau pemotongan sekuens token ID
    # Menggunakan API TensorFlow RaggedTensor untuk padding
    # Konversi RaggedTensor ke DenseTensor dengan padding
    # Menggunakan API Aritmatika untuk menghitung ukuran padding/pemotongan jika diperlukan
    # Misalnya: tf.minimum(tf.size(token_ids), max_sequence_length)
    # Note: RaggedTensor.to_tensor automatically handles padding to the max size in the batch,
    # but we can also pad to a fixed shape. Let's use a fixed shape for consistency.
    padded_token_ids = token_ids.to_tensor(default_value=0) # Pad with 0 (assuming 0 is not a valid token ID)
    # If we need a fixed shape regardless of batch max, we can slice/pad explicitly:
    # current_len = tf.shape(padded_token_ids)[0]
    # padded_token_ids = tf.pad(padded_token_ids, [[0, tf.maximum(0, max_sequence_length - current_len)]], constant_values=0)
    # padded_token_ids = padded_token_ids[:max_sequence_length]

    # Menggunakan API Aritmatika jika ada operasi numerik pada ID token
    # Contoh sederhana: menambahkan 1 ke setiap ID (tidak umum, hanya ilustrasi)
    # padded_token_ids = tf.add(padded_token_ids, 1)

    return padded_token_ids


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
    x_numeric = tf.keras.layers.GRU(config['model']['architecture']['gru_units'], return_sequences=True)(x_numeric)
    # Menggunakan API Keras Layers: LayerNormalization
    x_numeric = tf.keras.layers.LayerNormalization()(x_numeric)
    # Menggunakan API Keras Layers: Dropout
    x_numeric = tf.keras.layers.Dropout(config['model']['architecture']['dropout_rate'])(x_numeric)

    # Opsi: Tambahkan Conv1D untuk pola lokal (TCN-inspired element)
    if config['model']['architecture'].get('use_conv1d', False):
         # Menggunakan API Keras Layers: Conv1D
        x_numeric = tf.keras.layers.Conv1D(
            filters=config['model']['architecture']['conv1d_filters'],
            kernel_size=config['model']['architecture']['conv1d_kernel_size'],
            activation='relu', # Menggunakan API Aritmatika di dalam layer
            padding='causal', # Penting untuk deret waktu
            dilation_rate=config['model']['architecture'].get('conv1d_dilation_rate', 1) # Untuk TCN
        )(x_numeric)
         # Menggunakan API Keras Layers: LayerNormalization
        x_numeric = tf.keras.layers.LayerNormalization()(x_numeric)
         # Menggunakan API Keras Layers: Dropout
        x_numeric = tf.keras.layers.Dropout(config['model']['architecture']['dropout_rate'])(x_numeric)

    # Merangkum sekuens numerik
     # Menggunakan API Keras Layers: GlobalAveragePooling1D atau Flatten jika Conv1D last layer, atau GlobalAveragePooling1D setelah GRU
    # Jika GRU return_sequences=True, butuh pooling/flatten sebelum Dense
    x_numeric = tf.keras.layers.GlobalAveragePooling1D()(x_numeric) # Pooling output GRU
    # Jika ingin menggunakan output terakhir GRU saja (GRU return_sequences=False), hapus pooling


    # Jalur Pemrosesan Teks (jika digunakan)
    if config['use_text_input'] and vocabulary_size is not None:
        # Input teks adalah sekuens ID integer
        # Menggunakan API Keras Layers: Input
        text_input = tf.keras.layers.Input(shape=text_input_shape, name='text_input', dtype=tf.int64) # Input ID integer

        # Menggunakan API Keras Layers: Embedding
        # Menggunakan API Aritmatika di dalam layer Embedding
        # Masking 0 ID jika 0 digunakan untuk padding
        x_text = tf.keras.layers.Embedding(input_dim=vocabulary_size, # Vocabulary size termasuk OOV
                                           output_dim=config['model']['architecture']['embedding_dim'],
                                           mask_zero=True # Penting jika 0 digunakan untuk padding
                                          )(text_input)

        # Opsi: Tambahkan GRU atau Conv1D pada embedding teks
         # Menggunakan API Keras Layers: GRU
        x_text = tf.keras.layers.GRU(config['model']['architecture']['gru_units'] // 2)(x_text) # Setengah unit GRU
         # Menggunakan API Keras Layers: LayerNormalization
        x_text = tf.keras.layers.LayerNormalization()(x_text)
         # Menggunakan API Keras Layers: Dropout
        x_text = tf.keras.layers.Dropout(config['model']['architecture']['dropout_rate'])(x_text)

        # Menggabungkan jalur numerik dan teks
         # Menggunakan API Keras Layers: Concatenate
        combined = tf.keras.layers.Concatenate()([x_numeric, x_text])
    else:
        combined = x_numeric # Hanya jalur numerik jika teks tidak digunakan

    # Layer Akhir
     # Menggunakan API Keras Layers: Dense
    output_layer = tf.keras.layers.Dense(num_target_features, name='output')(combined) # 3 unit untuk HLC

    # Menggunakan API Keras Model: Functional API
    if config['use_text_input'] and vocabulary_size is not None:
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
    # Konfigurasi TensorFlow dan Hardware:
    try:
        logger.info("Mengkonfigurasi TensorFlow dan Hardware...")
        # Deteksi GPU:
        physical_devices_gpu = tf.config.list_physical_devices('GPU')
        if physical_devices_gpu:
            logger.info(f"Ditemukan GPU: {len(physical_devices_gpu)}")
            # Setel Perangkat Terlihat (Gunakan semua GPU yang terdeteksi):
            tf.config.set_visible_devices(physical_devices_gpu, 'GPU')
            # Atur Memori Growth untuk setiap GPU:
            for gpu in physical_devices_gpu:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info("Memory growth diaktifkan untuk GPU.")
            # Aktifkan Mixed Precision jika menggunakan GPU:
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            logger.info("Mixed precision (mixed_float16) diaktifkan.")
        else:
            logger.warning("Tidak ada GPU yang terdeteksi. Menggunakan CPU.")
            # Setel Perangkat Terlihat (Hanya gunakan CPU jika tidak ada GPU):
            physical_devices_cpu = tf.config.list_physical_devices('CPU')
            tf.config.set_visible_devices(physical_devices_cpu, 'CPU')
            # Mixed precision tidak relevan untuk CPU

        # Setel seed untuk reproduksibilitas:
        logger.info(f"Menyetel seed: {config['seed']}")
        np.random.seed(config['seed'])
        tf.random.set_seed(config['seed'])
        logger.info("Setup hardware dan seed selesai.")

    except Exception as e:
        logger.warning(f"Gagal mengkonfigurasi hardware TensorFlow: {e}")
        # Pipeline akan tetap berjalan di CPU jika konfigurasi GPU gagal


    # --- Langkah 1: Pemuatan Data & Preprocessing Awal ---
    logger.info("Langkah 1: Memuat data dan preprocessing awal...")
    df = None # Inisialisasi df
    try:
        # Menggunakan Pandas untuk membaca data (Membaca)
        data_path = config['data']['raw_path']
        if not os.path.exists(data_path):
             logger.error(f"File data tidak ditemukan di {os.path.abspath(data_path)}. Pastikan path benar dan file ada.")
             # Di GitHub Actions, ini akan menyebabkan workflow gagal.
             return # Hentikan pipeline jika file data tidak ada

        if data_path.endswith('.csv'):
            df = pd.read_csv(data_path)
        elif data_path.endswith('.json'):
            df = pd.read_json(data_path)
        else:
            raise ValueError("Format data tidak didukung. Gunakan .csv atau .json.")

        logger.info(f"Data mentah dimuat dari {data_path}. Jumlah baris: {len(df)}")

        # Pastikan data terurut berdasarkan Date jika ada (Opsional tapi direkomendasikan)
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date']) # Konversi ke datetime
            df.sort_values(by='Date', inplace=True)
            logger.info("Data diurutkan berdasarkan kolom 'Date'.")
            # Opsi: Set 'Date' sebagai index jika diinginkan, tapi pastikan kolom 'Date' tidak menjadi fitur input
            # df.set_index('Date', inplace=True)


        # Pembersihan data awal
        initial_rows = len(df)
        # Hapus baris dengan NaN di kolom harga atau kolom fitur numerik yang *dipastikan* ada
        cols_to_check_nan = config['data']['feature_cols_numeric'] # Pastikan ini termasuk OHLCV
        df.dropna(subset=cols_to_check_nan, inplace=True)
        if len(df) < initial_rows:
            logger.warning(f"Menghapus {initial_rows - len(df)} baris dengan NaN di kolom fitur numerik utama.")


        # Pastikan kolom harga adalah numerik
        # already handled by dropna(subset=feature_cols_numeric) assuming OHLCV are in feature_cols_numeric
        # price_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        # for col in price_cols:
        #     if col in df.columns:
        #         df[col] = pd.to_numeric(df[col], errors='coerce')
        # initial_rows = len(df)
        # df.dropna(subset=price_cols, inplace=True) # Hapus baris jika kolom harga jadi NaN setelah konversi
        # if len(df) < initial_rows:
        #      logger.warning(f"Menghapus {initial_rows - len(df)} baris setelah konversi numerik.")


        # Feature Engineering (Menggunakan Aritmatika)
        # Contoh: Menghitung Pivot Points
        logger.info("Menghitung Pivot Points...")
        # Menggunakan Aritmatika Pandas Series
        # Pastikan OHLC ada setelah dropna
        if all(col in df.columns for col in ['High', 'Low', 'Close']):
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
             logger.warning("Kolom OHLC tidak lengkap untuk menghitung Pivot Points.")


        # Menyiapkan Target (HLC Selanjutnya)
        logger.info(f"Menyiapkan target HLC selanjutnya dengan shift -{config['parameter_windowing']['window_size']}...")
        # Menggunakan Pandas shift. Shift negatif menarik data masa depan ke baris saat ini.
        # Jendela input size=N di baris T akan memprediksi target di baris T+N.
        # Jadi, target di baris T haruslah HLC di baris T+N.
        # shift(-N) menggeser nilai dari N baris ke bawah ke baris saat ini.
        df['High_Next'] = df['High'].shift(-config['parameter_windowing']['window_size'])
        df['Low_Next'] = df['Low'].shift(-config['parameter_windowing']['window_size'])
        df['Close_Next'] = df['Close'].shift(-config['parameter_windowing']['window_size'])

        # Menghapus baris terakhir yang memiliki NaN setelah shift
        # Jumlah baris yang dihapus akan sama dengan window_size
        initial_rows = len(df)
        df.dropna(subset=['High_Next', 'Low_Next', 'Close_Next'], inplace=True)
        if len(df) < initial_rows:
             logger.warning(f"Menghapus {initial_rows - len(df)} baris dengan NaN di target setelah shift (-{config['parameter_windowing']['window_size']}).")


        # Membuat Fitur Teks Deskriptif (Menulis Teks ke Kolom)
        if config['use_text_input']:
            logger.info("Membuat fitur teks deskriptif...")
            # Menggunakan Pandas apply dan fungsi kustom generate_analysis_text
            # Pastikan kolom yang dibutuhkan generate_analysis_text ada setelah dropna
            if all(col in df.columns for col in ['High', 'Low', 'Close', 'Pivot', 'R1', 'S1']): # Sesuaikan dengan kolom yang dipakai di generate_analysis_text
                df['Analysis_Text'] = df.apply(generate_analysis_text, axis=1)
                if not df.empty:
                     logger.info(f"Contoh teks analisis: {df['Analysis_Text'].iloc[0]}")
                else:
                     logger.warning("DataFrame kosong setelah preprocessing. Tidak dapat membuat teks analisis.")
            else:
                 logger.warning("Kolom yang dibutuhkan untuk generate_analysis_text tidak lengkap setelah preprocessing.")


        # Identifikasi Fitur Input dan Target
        # Pastikan kolom indikator yang dihitung juga masuk fitur numerik
        indicator_cols = [col for col in df.columns if col.startswith(('Piv', 'R', 'S'))] # Match 'Pivot', 'R*', 'S*'
        # Ambil feature_cols_numeric dari config, lalu tambahkan indicator_cols jika belum ada
        feature_cols_numeric = list(config['data']['feature_cols_numeric']) # Mulai dengan kolom dari config
        for col in indicator_cols:
            if col not in feature_cols_numeric:
                feature_cols_numeric.append(col) # Tambahkan kolom indikator yang dihitung

        feature_cols_text = ['Analysis_Text'] if config['use_text_input'] and 'Analysis_Text' in df.columns else []
        target_cols = ['High_Next', 'Low_Next', 'Close_Next'] # Pastikan ini ada setelah shift dan dropna

        # Filter DataFrame hanya untuk kolom yang relevan sebelum konversi ke NumPy
        # Ini memastikan urutan kolom konsisten
        all_relevant_cols = feature_cols_numeric + feature_cols_text + target_cols + (['Date'] if 'Date' in df.columns else [])
        df_processed = df[all_relevant_cols].copy() # Buat salinan untuk menghindari SettingWithCopyWarning


        # Konversi ke NumPy Arrays
        # Menggunakan NumPy untuk konversi
        if not df_processed.empty:
            # Konversi hanya kolom fitur dan target yang relevan
            data_numeric = df_processed[feature_cols_numeric].values.astype(np.float32)
            data_text = df_processed[feature_cols_text].values.flatten().astype(str) if feature_cols_text else np.array([], dtype=str)
            data_target = df_processed[target_cols].values.astype(np.float32)

            logger.info(f"Final data numerik shape: {data_numeric.shape}")
            logger.info(f"Final data teks shape: {data_text.shape}")
            logger.info(f"Final data target shape: {data_target.shape}")
        else:
            logger.error("DataFrame kosong setelah preprocessing. Tidak dapat melanjutkan.")
            return # Hentikan pipeline jika data kosong setelah preprocessing


        # Membagi Data (menggunakan slicing NumPy)
        total_samples = len(df_processed)
        # Perhitungan ukuran split harus memastikan setidaknya ada 1 sample di setiap set
        # Menggunakan floor untuk train/val dan sisa untuk test lebih aman
        train_size = int(np.floor(total_samples * config['data']['train_split']))
        val_size = int(np.floor(total_samples * config['data']['val_split']))
        test_size = total_samples - train_size - val_size # Ukuran test adalah sisanya

        if train_size <= 0 or val_size <= 0 or test_size <= 0:
             logger.error(f"Ukuran set data tidak memadai. Train: {train_size}, Val: {val_size}, Test: {test_size}. Sesuaikan split atau gunakan data lebih banyak.")
             return # Hentikan pipeline jika ukuran data tidak memadai


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
        return # Hentikan pipeline jika ada error fatal

    # --- Langkah 2: Scaling Data & Pembuatan Pipeline tf.data ---
    logger.info("Langkah 2: Scaling data dan membuat pipeline tf.data...")
    scaler_input = None # Inisialisasi scaler
    scaler_target = None
    vocabulary_table = None
    vocabulary_size = None
    try:
        # Scaling Data Numerik (Menggunakan Aritmatika di dalam scaler)
        scaler_input = MinMaxScaler()
        scaler_target = MinMaxScaler()

        # Latih scaler hanya pada data pelatihan
        scaled_input_train = scaler_input.fit_transform(input_train_numeric)
        scaled_target_train = scaler_target.fit_transform(target_train)

        # Terapkan scaler pada data validasi dan pengujian
        scaled_input_val = scaler_input.transform(input_val_numeric)
        scaled_input_test = scaler_input.transform(input_test_numeric)

        scaled_target_val = scaler_target.transform(target_val)
        # target_test tidak perlu diskalakan di sini untuk evaluasi/inverse transform nanti,
        # hanya input_test_numeric yang diskalakan untuk dimasukkan ke model.

        # Membuat Kosakata untuk Teks (jika digunakan)
        if config['use_text_input'] and feature_cols_text:
            logger.info("Membuat kosakata dari data pelatihan teks...")
            # Mengumpulkan semua token unik dari data pelatihan teks
            # Menggunakan API TensorFlow Strings: split
            # Ensure tokens are string before converting to tensor
            input_train_text_str = tf.constant(input_train_text.tolist(), dtype=tf.string)
            all_train_tokens_ragged = tf.strings.split(input_train_text_str)
            # Add a placeholder for padding (if 0 is used for padding) and OOV bucket
            # unique_tokens, _ = tf.unique(all_train_tokens_ragged.values)
            # tf.unique requires session/eager mode. Use numpy for simplicity here.
            all_train_tokens_np = all_train_tokens_ragged.values.numpy() # Ambil nilai tensor datar
            unique_tokens_np = np.unique(all_train_tokens_np[all_train_tokens_np != b'']) # Hilangkan token kosong

            # Add special tokens if needed (e.g., PAD, OOV, START, END)
            # For simplicity with mask_zero=True and num_oov_buckets, TensorFlow handles PAD/OOV.
            # PAD is typically 0. OOV is handled by num_oov_buckets.

            # Menggunakan API TensorFlow Lookup: StaticVocabularyTable
            # Membuat tabel lookup dari kosakata unik
            # Index 0 will be used for padding by default if mask_zero=True and input_dim includes 0.
            # Let's shift unique tokens by 1 to reserve 0 for padding. OOV will get the last index.
            keys = tf.constant(unique_tokens_np, dtype=tf.string)
            values = tf.range(1, tf.size(keys) + 1, dtype=tf.int64) # Start values from 1
            # Add OOV bucket. Size of OOV bucket is num_oov_buckets. Indices for OOV start after unique tokens.
            num_oov_buckets = 1
            oov_start_index = tf.size(keys) + 1 # OOV index will be this + offset (usually just this value)

            init = tf.lookup.KeyValueTensorInitializer(keys, values, key_dtype=tf.string, value_dtype=tf.int64)
            # num_oov_buckets=1: 1 bucket untuk token di luar kosakata.
            # Token di luar kosakata akan dipetakan ke index `vocabulary_size - 1` jika num_oov_buckets=1
            vocabulary_table = tf.lookup.StaticVocabularyTable(init, num_oov_buckets=num_oov_buckets)
            vocabulary_size = vocabulary_table.size().numpy() # Size includes unique keys + num_oov_buckets

            # The ID 0 is reserved for padding when mask_zero=True. Ensure it's not in our vocab values.
            # Our values start from 1, so 0 is implicitly reserved.

            logger.info(f"Ukuran kosakata teks (termasuk OOV): {vocabulary_size}")
            if len(unique_tokens_np) > 0:
                logger.info(f"Contoh token unik (mapping value): {unique_tokens_np[:10]} -> {vocabulary_table.lookup(tf.constant(unique_tokens_np[:10])).numpy()}")


        # Membuat tf.data.Dataset dari data yang sudah diproses
        # Menggunakan API TensorFlow Data: from_tensor_slices
        if config['use_text_input'] and feature_cols_text:
            dataset_train_raw = tf.data.Dataset.from_tensor_slices(((scaled_input_train, input_train_text), scaled_target_train))
            dataset_val_raw = tf.data.Dataset.from_tensor_slices(((scaled_input_val, input_val_text), scaled_target_val))
            # Target test tidak diskalakan, jadi gunakan target_test asli
            dataset_test_raw = tf.data.Dataset.from_tensor_slices(((scaled_input_test, input_test_text), target_test))
        else:
             # Menggunakan API TensorFlow Data: from_tensor_slices
            dataset_train_raw = tf.data.Dataset.from_tensor_slices((scaled_input_train, scaled_target_train))
            dataset_val_raw = tf.data.Dataset.from_tensor_slices((scaled_input_val, scaled_target_val))
            # Target test tidak diskalakan, jadi gunakan target_test asli
            dataset_test_raw = tf.data.Dataset.from_tensor_slices((scaled_input_test, target_test))


        # Fungsi untuk membuat window dan memproses elemen dataset
        # Menggunakan @tf.function untuk kompilasi grafik
        # Perlu sedikit modifikasi fungsi ini untuk menangani structure_split_inputs
        # yang dihasilkan oleh window().flat_map(batch).map(...)
        @tf.function
        def process_window_elements(numeric_window, text_window, target_window):
             """Memproses window data (termasuk teks) di dalam pipeline map."""
             # numeric_window shape: (window_size, num_numeric_features)
             # text_window shape: (window_size,) # string tensors
             # target_window shape: (window_size, num_target_features) - we only need the LAST target!

             # Get the last target element for this window
             final_target = target_window[-1] # Use Aritmatika Indexing

             processed_text_window = text_window # Default if text not used or no text column

             if config['use_text_input'] and feature_cols_text and vocabulary_table is not None:
                 # Process each text element in the window
                 # Map the text processing function over the window dimension
                 # Use a wrapper function or lambda for the map call
                 # Menggunakan API TensorFlow Strings dan Lookup di dalam process_text_for_model
                 processed_text_window = tf.map_fn(
                     lambda text_elem: process_text_for_model(text_elem, config['data']['max_text_sequence_length'], vocabulary_table),
                     text_window,
                     fn_output_signature=tf.int64 # Expected output dtype of the map function
                 )
                 # processed_text_window shape should now be (window_size, max_text_sequence_length)


             # Return tuple that matches model input structure and the *final* target
             if config['use_text_input'] and feature_cols_text:
                 return (numeric_window, processed_text_window), final_target
             else:
                 return numeric_window, final_target


        # Fungsi pembantu untuk windowing dan batching
        def create_window_dataset(dataset_raw, window_size, window_shift, drop_remainder):
             # Menggunakan API TensorFlow Data: window
            dataset_windowed = dataset_raw.window(size=window_size, shift=window_shift, drop_remainder=drop_remainder)
             # Menggunakan API TensorFlow Data: flat_map
             # Menggunakan API TensorFlow Data: batch (untuk mengumpulkan elemen dalam window)
            # flat_map must return a dataset. We apply batch(window_size) to collect elements of each window.
            # Then flatten these window-batches into a dataset of window-tensors.
            dataset_flattened = dataset_windowed.flat_map(lambda window: window.batch(window_size))
            return dataset_flattened


        # Membuat Pipeline tf.data (Windowing, Batching, Caching, Prefetching)
        # Menggunakan API TensorFlow Data: window
        # Menggunakan API TensorFlow Data: flat_map
        # Menggunakan API TensorFlow Data: shuffle
        # Menggunakan API TensorFlow Data: batch
        # Menggunakan API TensorFlow Data: cache
        # Menggunakan API TensorFlow Data: prefetch
        # Menggunakan API TensorFlow Data: AUTOTUNE
        # Menggunakan API TensorFlow Data: map (untuk memproses teks dan memilih target)

        # Pipeline Pelatihan
        dataset_train_windowed = create_window_dataset(dataset_train_raw, config['parameter_windowing']['window_size'], config['parameter_windowing']['window_shift'], True)
        dataset_train = dataset_train_windowed.map(
            lambda input_tuple, target_window: process_window_elements(*input_tuple, target_window) if config['use_text_input'] and feature_cols_text else process_window_elements(input_tuple, None, target_window), # Pass None for text if not used
            num_parallel_calls=tf.data.AUTOTUNE
        )
        dataset_train = dataset_train.shuffle(config['data']['shuffle_buffer_size'])
        dataset_train = dataset_train.batch(config['training']['batch_size']).cache().prefetch(tf.data.AUTOTUNE)


        # Pipeline Validasi (tanpa shuffle)
        dataset_val_windowed = create_window_dataset(dataset_val_raw, config['parameter_windowing']['window_size'], config['parameter_windowing']['window_shift'], True)
        dataset_val = dataset_val_windowed.map(
             lambda input_tuple, target_window: process_window_elements(*input_tuple, target_window) if config['use_text_input'] and feature_cols_text else process_window_elements(input_tuple, None, target_window),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        dataset_val = dataset_val.batch(config['training']['batch_size']).cache().prefetch(tf.data.AUTOTUNE)


        # Pipeline Pengujian (tanpa shuffle, tanpa cache jika data test besar)
        # Cache data test hanya jika ukurannya relatif kecil
        dataset_test_windowed = create_window_dataset(dataset_test_raw, config['parameter_windowing']['window_size'], config['parameter_windowing']['window_shift'], True)
        dataset_test = dataset_test_windowed.map(
            lambda input_tuple, target_window: process_window_elements(*input_tuple, target_window) if config['use_text_input'] and feature_cols_text else process_window_elements(input_tuple, None, target_window),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        dataset_test = dataset_test.batch(config['training']['batch_size']).prefetch(tf.data.AUTOTUNE)


        logger.info("Pipeline tf.data selesai dibuat.")

    except Exception as e:
        logger.error(f"Error selama Langkah 2: {e}")
        return # Hentikan pipeline jika ada error fatal


    # --- Langkah 3: Definisi & Kompilasi Model ---
    logger.info("Langkah 3: Mendefinisikan atau memuat model...")
    model = None
    # Define model architecture parameters
    numeric_input_shape = (config['parameter_windowing']['window_size'], len(feature_cols_numeric))
    # Text input shape is (window_size, max_text_sequence_length) after processing
    text_input_shape = (config['parameter_windowing']['window_size'], config['data']['max_text_sequence_length']) if config['use_text_input'] and feature_cols_text else None
    num_target_features = len(target_cols)

    try:
        if config['mode'] == 'initial_train':
            logger.info("Mode: initial_train. Mendefinisikan model baru.")
            # Menggunakan API Keras Model dan Layers
            # Mendefinisikan model QuantAI baru
            model = build_quantai_model(config, numeric_input_shape, text_input_shape, num_target_features, vocabulary_size)

            # Kompilasi Model
            logger.info("Mengkompilasi model.")
             # Menggunakan API Keras Optimizers
            optimizer = tf.keras.optimizers.Adam(learning_rate=config['training']['learning_rate'])
             # Menggunakan API Keras Losses
            loss_fn = tf.keras.losses.MeanAbsoluteError() # Menggunakan Aritmatika
             # Menggunakan API Keras Metrics
            metrics = [tf.keras.metrics.RootMeanSquaredError(), tf.keras.metrics.MeanAbsoluteError()] # Menggunakan Aritmatika

            # Menggunakan API Keras Model: compile
            model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)

            logger.info("Model baru berhasil didefinisikan dan dikompilasi.")
            model.summary(print_fn=logger.info) # Cetak ringkasan model ke log

        elif config['mode'] in ['incremental_learn', 'predict_only']:
            load_path = config['model']['load_path']
            logger.info(f"Mode: {config['mode']}. Memuat model dari {load_path}")
            # Menggunakan API Keras Models: load_model untuk format .h5
            if not os.path.exists(load_path):
                 logger.error(f"File model tidak ditemukan di {os.path.abspath(load_path)}. Pastikan path benar dan model telah disimpan sebelumnya.")
                 return # Hentikan pipeline jika model tidak ditemukan untuk dimuat

            try:
                 # Menggunakan API Keras Models: load_model (untuk format .h5)
                # custom_objects diperlukan jika model menggunakan layer/fungsi kustom yang tidak standar Keras
                # Saat ini, kita hanya pakai layer standar, jadi tidak perlu custom_objects
                model = tf.keras.models.load_model(load_path)
                logger.info("Model berhasil dimuat dari format .h5 menggunakan tf.keras.models.load_model.")
            except Exception as e:
                logger.error(f"Gagal memuat model dari {load_path}: {e}")
                model = None # Set model ke None jika gagal memuat

            if model is not None:
                logger.info("Model berhasil dimuat.")
                # Ringkasan mungkin tidak selalu akurat setelah dimuat, tapi coba cetak
                # model.summary(print_fn=logger.info)

                # Kompilasi ulang jika mode incremental_learn (diperlukan untuk training)
                if config['mode'] == 'incremental_learn':
                    logger.info("Mengkompilasi ulang model untuk incremental_learn.")
                    # Menggunakan API Keras Optimizers
                    optimizer = tf.keras.optimizers.Adam(learning_rate=config['training']['learning_rate'])
                    # Menggunakan API Keras Losses
                    loss_fn = tf.keras.losses.MeanAbsoluteError()
                    # Menggunakan API Keras Metrics
                    metrics = [tf.keras.metrics.RootMeanSquaredError(), tf.keras.metrics.MeanAbsoluteError()]
                    model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)
                    logger.info("Model berhasil dikompilasi ulang.")


        else:
            logger.error(f"Mode operasional tidak valid: {config['mode']}")
            return # Hentikan pipeline jika mode tidak valid

    except Exception as e:
        logger.error(f"Error selama Langkah 3: {e}")
        return # Hentikan pipeline jika ada error fatal

    if model is None:
         logger.error("Model tidak tersedia setelah Langkah 3. Menghentikan pipeline.")
         return

    # --- Langkah 4: Pelatihan Model (Belajar Mandiri/Hybrid/Otonom) ---
    if config['mode'] in ['initial_train', 'incremental_learn']:
        logger.info(f"Langkah 4: Memulai pelatihan model dalam mode {config['mode']}...")
        try:
            # Menggunakan API Keras Model: fit
            # Menggunakan API Keras Callbacks: EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard

            # Pastikan direktori save_path dan tensorboard_log_dir sudah dibuat di Langkah 6
            # atau buat di sini jika tidak ingin menunggu Langkah 6
            output_dir = config['output']['base_dir']
            model_save_path_full = os.path.join(output_dir, config['output']['model_save_file']) # Path ke file .h5
            scaler_save_dir_full = os.path.join(output_dir, config['output']['scaler_subdir'])
            tensorboard_log_dir_full = os.path.join(output_dir, config['output']['tensorboard_log_dir'])
            # Membuat direktori induk untuk file model .h5
            tf.io.gfile.makedirs(os.path.dirname(model_save_path_full))
            tf.io.gfile.makedirs(scaler_save_dir_full)
            tf.io.gfile.makedirs(tensorboard_log_dir_full)


            callbacks = [
                 # Menggunakan API Keras Callbacks: EarlyStopping
                tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=config['training']['early_stopping_patience'], restore_best_weights=True),
                 # Menggunakan API Keras Callbacks: ModelCheckpoint
                # Menyimpan model terbaik dalam format .h5
                tf.keras.callbacks.ModelCheckpoint(filepath=model_save_path_full, monitor='val_loss', save_best_only=True, save_format='h5'), # <-- save_format='h5'
                 # Menggunakan API Keras Callbacks: ReduceLROnPlateau (Menggunakan Aritmatika)
                tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=config['training']['lr_reduce_factor'], patience=config['training']['lr_reduce_patience'], min_lr=config['training']['min_lr']),
                 # Menggunakan API Keras Callbacks: TensorBoard
                tf.keras.callbacks.TensorBoard(log_dir=tensorboard_log_dir_full)
            ]

            epochs_to_run = config['training']['epochs'] if config['mode'] == 'initial_train' else config['training']['incremental_epochs']

            logger.info(f"Melatih selama {epochs_to_run} epoch.")
            history = model.fit(
                dataset_train,
                epochs=epochs_to_run,
                validation_data=dataset_val,
                callbacks=callbacks
            )
            logger.info("Pelatihan selesai.")

            # Muat kembali model terbaik setelah pelatihan selesai (ModelCheckpoint menyimpannya)
            # Ini penting jika EarlyStopping menghentikan pelatihan sebelum epoch terakhir
            logger.info(f"Memuat model terbaik dari {model_save_path_full}")
             # Menggunakan API Keras Models: load_model (untuk format .h5)
            try:
                # custom_objects mungkin diperlukan di sini jika model menggunakan komponen kustom
                model = tf.keras.models.load_model(model_save_path_full)
                logger.info("Model terbaik berhasil dimuat setelah pelatihan.")
            except Exception as e:
                 logger.warning(f"Gagal memuat model terbaik setelah pelatihan dari {model_save_path_full}: {e}. Menggunakan model akhir dari fit().")
                 # Jika gagal memuat model terbaik, gunakan model yang ada setelah fit()


            # --- Fondasi Belajar Mandiri/Hybrid/Otonom (Loop Kustom Opsional) ---
            # Jika diperlukan logika update bobot yang lebih granular dari model.fit,
            # implementasikan loop kustom di sini menggunakan:
            # @tf.function, tf.GradientTape, optimizer.apply_gradients, model.trainable_variables,
            # tf.cond, tf.while_loop, tf.py_function, model.train_on_batch
            # Contoh (pseudocode):
            # @tf.function
            # def custom_train_step(inputs, targets):
            #     with tf.GradientTape() as tape:
            #         predictions = model(inputs, training=True)
            #         loss = model.compiled_loss(targets, predictions) # Menggunakan loss yang dikompilasi
            #         # Tambahkan loss kustom jika ada: loss += sum(model.losses)
            #     gradients = tape.gradient(loss, model.trainable_variables)
            #     model.optimizer.apply_gradients(zip(gradients, model.trainable_variables)) # Menggunakan optimizer yang dikompilasi
            #     # Update metrik kompilasi jika ada: model.compiled_metrics.update_state(targets, predictions)
            #     return loss
            #
            # if config['mode'] == 'incremental_learn_custom': # Mode kustom baru
            #     logger.info("Memulai pelatihan inkremental dengan loop kustom.")
            #     # Pastikan dataset_new_data disiapkan di Langkah 2
            #     if 'dataset_new_data' in locals():
            #         for epoch in range(config['training']['incremental_epochs_custom']):
            #             logger.info(f"Epoch inkremental kustom {epoch+1}/{config['training']['incremental_epochs_custom']}")
            #             # Reset metrik jika menggunakan metrik kompilasi
            #             # for metric in model.metrics: metric.reset_states()
            #             for batch_inputs, batch_targets in dataset_new_data:
            #                 batch_loss = custom_train_step(batch_inputs, batch_targets)
            #                 # Logika otonom: if batch_loss < threshold: ...
            #                 # Log loss atau metrik batch
            #             # Log metrik epoch jika menggunakan metrik kompilasi
            #             # eval_on_val_set() # Evaluasi pada set validasi secara berkala
            #         logger.info("Pelatihan inkremental kustom selesai.")
            #     else:
            #          logger.warning("Dataset data baru tidak tersedia untuk pelatihan inkremental kustom.")


        except Exception as e:
            logger.error(f"Error selama Langkah 4: {e}")
            # Lanjutkan ke langkah berikutnya meskipun ada error pelatihan,
            # mungkin kita masih ingin menyimpan model yang dimuat atau melakukan prediksi.


    # --- Langkah 5: Evaluation Model Akhir ---
    eval_results = None
    # Hanya lakukan evaluasi jika mode bukan 'predict_only' dan model berhasil dimuat/dilatih
    if config['mode'] in ['initial_train', 'incremental_learn'] and model is not None:
        logger.info("Langkah 5: Mengevaluasi model akhir...")
        try:
            # Menggunakan API Keras Model: evaluate
            # Menggunakan Aritmatika di dalam metrik
            # Pastikan model dikompilasi jika mode incremental_learn dan dimuat dengan tf.saved_model.load tanpa compile
            # (Ini sudah ditangani di Langkah 3)
            # Pastikan dataset_test tersedia
            if hasattr(model, 'evaluate') and 'dataset_test' in locals():
                logger.info(f"Dataset test size untuk evaluasi: {tf.data.Dataset.cardinality(dataset_test).numpy()} batches") # Log dataset size
                eval_results = model.evaluate(dataset_test)
                # model.metrics_names seharusnya tersedia setelah kompilasi
                if hasattr(model, 'metrics_names'):
                     logger.info(f"Hasil Evaluasi Akhir: {dict(zip(model.metrics_names, eval_results))}")
                else:
                     logger.info(f"Hasil Evaluasi Akhir (metrik tidak tersedia): {eval_results}")

            elif 'dataset_test' not in locals():
                 logger.warning("Dataset test tidak tersedia. Tidak dapat melakukan evaluasi.")
            else:
                 logger.error("Model tidak memiliki metode evaluate. Tidak dapat melakukan evaluasi.")


        except Exception as e:
            logger.error(f"Error selama Langkah 5: {e}")
            # Lanjutkan ke langkah berikutnya meskipun ada error evaluasi

    # --- Langkah 6: Penyimpanan Aset & Hasil (Menulis) ---
    logger.info("Langkah 6: Menyimpan aset dan hasil...")
    try:
        # Pastikan direktori output ada (Menggunakan API TensorFlow IO GFile)
        output_dir = config['output']['base_dir']
        # Model .h5 disimpan oleh ModelCheckpoint, path diambil dari config['output']['model_save_file']
        # scaler_save_dir_full = os.path.join(output_dir, config['output']['scaler_subdir'])
        eval_results_path_full = os.path.join(output_dir, config['output']['eval_results_file'])
        predictions_path_full = os.path.join(output_dir, config['output']['predictions_file'])
        # tensorboard_log_dir_full = os.path.join(output_dir, config['output']['tensorboard_log_dir']) # Path ini digunakan oleh callback


        # Membuat direktori induk untuk semua output file (Menggunakan API TensorFlow IO GFile: makedirs)
        # Pastikan direktori dibuat sebelum menyimpan file di dalamnya
        # Buat base dir dan sub-direktori scaler dan tensorboard logs
        tf.io.gfile.makedirs(os.path.join(output_dir, config['output']['scaler_subdir']))
        tf.io.gfile.makedirs(os.path.join(output_dir, config['output']['tensorboard_log_dir']))
        # Buat direktori induk untuk file eval_results dan predictions
        tf.io.gfile.makedirs(os.path.dirname(eval_results_path_full))
        tf.io.gfile.makedirs(os.path.dirname(predictions_path_full))


        if config['mode'] in ['initial_train', 'incremental_learn']:
            # Menyimpan Model Terlatih (Hasil Pelatihan, Jalan Offline/Deploy Anywhere)
            # Model terbaik sudah disimpan oleh ModelCheckpoint di Langkah 4 dalam format .h5
            # Path: config['output']['model_save_file']
            logger.info(f"Model terbaik dalam format .h5 sudah disimpan oleh ModelCheckpoint di Langkah 4.")

            # Menyimpan Scaler (Pendukung Offline/Deploy Anywhere)
            logger.info(f"Menyimpan scaler ke {os.path.join(output_dir, config['output']['scaler_subdir'])}...")
             # Menggunakan Joblib: dump
            # Pastikan scaler_input dan scaler_target tersedia (dilatih di Langkah 2)
            if 'scaler_input' in locals() and scaler_input is not None and 'scaler_target' in locals() and scaler_target is not None:
                 joblib.dump(scaler_input, os.path.join(output_dir, config['output']['scaler_subdir'], 'scaler_input.pkl'))
                 joblib.dump(scaler_target, os.path.join(output_dir, config['output']['scaler_subdir'], 'scaler_target.pkl'))
                 logger.info("Scaler berhasil disimpan.")
            else:
                 logger.warning("Scaler tidak tersedia. Tidak dapat menyimpan scaler.")

            # Menyimpan Kosakata Teks (jika digunakan)
            if config['use_text_input'] and feature_cols_text and vocabulary_table is not None:
                 vocab_file_path = os.path.join(output_dir, config['output'].get('vocabulary_file', 'scalers/vocabulary.txt')) # tambahkan path ke config jika perlu
                 tf.io.gfile.makedirs(os.path.dirname(vocab_file_path)) # Pastikan dir ada
                 # TensorFlow vocabulary_table tidak langsung menyimpan kosakata string aslinya
                 # Kita perlu mendapatkan keys dari table dan menyimpannya
                 try:
                    # Mendapatkan keys dari lookup table
                    # Ini agak tricky, mungkin perlu membuat map terbalik dari ID ke string jika table tidak mengekspos keys
                    # Atau, simpan list unique_tokens_np yang digunakan untuk membuat table
                    if 'unique_tokens_np' in locals():
                        with open(vocab_file_path, 'w', encoding='utf-8') as f:
                            for token in unique_tokens_np:
                                f.write(token.decode('utf-8') + '\n')
                        logger.info(f"Kosakata teks berhasil disimpan ke {vocab_file_path}.")
                    else:
                         logger.warning("Daftar token unik untuk kosakata tidak tersedia. Tidak dapat menyimpan kosakata.")

                 except Exception as e:
                     logger.warning(f"Gagal menyimpan kosakata teks: {e}")


            # Menyimpan Hasil Evaluasi (Hasil Output)
            if eval_results is not None and model is not None and hasattr(model, 'metrics_names'):
                logger.info(f"Menyimpan hasil evaluasi ke {eval_results_path_full}...")
                # Pastikan model.metrics_names tersedia jika eval_results bukan None
                eval_dict = dict(zip(model.metrics_names, eval_results))
                # Menggunakan Python Standard Library: open dan json.dump (Menulis)
                with open(eval_results_path_full, 'w') as f:
                    json.dump(eval_dict, f, indent=4)
                logger.info("Hasil evaluasi berhasil disimpan.")
            elif eval_results is not None:
                 logger.warning("Nama metrik model tidak tersedia. Tidak dapat menyimpan hasil evaluasi dalam format dictionary.")


        # Menyimpan Prediksi (Hasil Output, Opsional untuk semua mode)
        # Hanya simpan prediksi jika save_predictions True DAN model berhasil dimuat/dilatih
        if config['output'].get('save_predictions', False) and model is not None:
             # Tentukan dataset mana yang akan diprediksi berdasarkan mode
             # Dalam semua mode, kita akan melakukan prediksi pada dataset test jika tersedia
             dataset_to_predict = None
             if 'dataset_test' in locals():
                 dataset_to_predict = dataset_test
                 logger.info("Membuat prediksi pada dataset test.")
             else:
                  logger.warning("Dataset test tidak tersedia untuk prediksi.")


             if dataset_to_predict is not None and 'scaler_target' in locals() and scaler_target is not None:
                logger.info(f"Membuat dan menyimpan prediksi ke {predictions_path_full}...")
                # Menggunakan API Keras Model: predict
                predictions_scaled = model.predict(dataset_to_predict)

                # Pastikan scaler_target tersedia untuk inverse transform
                # Menggunakan Aritmatika di dalam scaler inverse_transform
                predictions_original_scale = scaler_target.inverse_transform(predictions_scaled)

                # Membuat DataFrame hasil prediksi
                df_predictions = pd.DataFrame(predictions_original_scale, columns=[f'{col}_Pred' for col in target_cols])

                # Opsi: Gabungkan dengan data test asli (membutuhkan penanganan indeks)
                # Ini bisa rumit dengan windowing. Cara paling aman adalah menyimpan prediksi saja
                # atau menggabungkan data asli dan prediksi di luar pipeline ini.
                # Jika perlu menggabungkan, Anda butuh data_test *asli* (sebelum windowing/batching)
                # dan melakukan slicing/indexing yang tepat.
                # Contoh sederhana (perlu disesuaikan dengan indexing window):
                # try:
                #     # Assume predictions align row-wise with the LAST element of each window in the original test data
                #     # This requires careful handling of window_shift and drop_remainder
                #     original_test_data_sliced = df_processed[train_size + val_size + config['parameter_windowing']['window_size'] -1 :][::config['parameter_windowing']['window_shift']]
                #     df_predictions.index = original_test_data_sliced.index[:len(df_predictions)]
                #     df_predictions = original_test_data_sliced.join(df_predictions)
                #     logger.info("Prediksi digabungkan dengan data test asli.")
                # except Exception as e_join:
                #     logger.warning(f"Gagal menggabungkan prediksi dengan data test asli: {e_join}")


                # Menggunakan Pandas to_csv (Menulis)
                df_predictions.to_csv(predictions_path_full, index=False) # Set index=True jika ingin menyimpan index Date
                logger.info("Prediksi berhasil disimpan.")
             elif model is None:
                  logger.warning("Model tidak tersedia. Tidak dapat membuat prediksi.")
             elif ('scaler_target' in locals() and scaler_target is None) or ('scaler_target' not in locals()):
                  logger.warning("Scaler target tidak tersedia. Tidak dapat melakukan inverse transform atau menyimpan prediksi.")
             # else: # dataset_to_predict is None handled earlier
             #      logger.warning("Tidak ada dataset yang ditentukan untuk prediksi.")


        logger.info("Langkah 6 selesai.")

    except Exception as e:
        logger.error(f"Error selama Langkah 6: {e}")


    logger.info("Pipeline QuantAI selesai.")

# --- Eksekusi Pipeline ---

if __name__ == "__main__":
    # Menggunakan argparse untuk membaca path konfigurasi dari command-line
    parser = argparse.ArgumentParser(description='Run QuantAI ML Pipeline.')
    parser.add_argument('--config', type=str, required=True, help='Path to the YAML configuration file.')
    args = parser.parse_args()

    config_path = args.config

    # Membaca konfigurasi saat script dijalankan
    config = None
    try:
        # Menggunakan Python Standard Library: open
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Konfigurasi berhasil dimuat dari {config_path}")
        # logger.info(f"Konfigurasi: {config}") # Opsi: log seluruh config
    except FileNotFoundError:
        logger.error(f"File konfigurasi tidak ditemukan di {config_path}. Pastikan path benar di workflow atau command-line.")
        exit(1) # Keluar dengan kode error jika config tidak ditemukan

    except yaml.YAMLError as e:
        logger.error(f"Error mem-parsing file konfigurasi YAML dari {config_path}: {e}")
        exit(1) # Keluar dengan kode error jika file config error

    # Jalankan pipeline utama hanya jika config berhasil dimuat
    if config:
        # --- Validasi Konfigurasi Esensial ---
        essential_keys = ['data', 'model', 'training', 'output', 'parameter_windowing'] # Menambahkan parameter_windowing
        missing_keys = [key for key in essential_keys if key not in config or config[key] is None] # Cek juga jika kunci ada tapi nilainya None

        if missing_keys:
            logger.error(f"File konfigurasi ({config_path}) tidak lengkap atau rusak. Kunci utama yang hilang atau kosong: {missing_keys}")
            logger.error("Pastikan file konfigurasi YAML memiliki semua bagian utama (data, model, training, output, parameter_windowing) dan tidak kosong.")
            exit(1) # Keluar jika konfigurasi tidak valid

        # --- Menambahkan Path Default (jika belum ada) ---
        # Sekarang kita yakin kunci utama ada karena sudah divalidasi di atas
        # Gunakan .get() untuk mengakses sub-kunci dengan aman jika sub-kunci tersebut opsional atau mungkin hilang

        # Tambahkan path 'vocabulary_file' default ke config['output'] jika belum ada dan use_text_input True
        # Gunakan .get() untuk mengakses 'scaler_subdir' dengan aman jika 'output' ada tapi 'scaler_subdir' tidak
        # Pastikan config['output'] adalah dictionary
        if isinstance(config['output'], dict):
             if config.get('use_text_input', False) and 'vocabulary_file' not in config['output']:
                  scaler_subdir = config['output'].get('scaler_subdir', 'scalers') # Gunakan default 'scalers' jika scaler_subdir tidak ada
                  # Gunakan base_dir dari config['output'] untuk membangun path lengkap
                  base_dir = config['output'].get('base_dir', '') # Gunakan default '' jika base_dir tidak ada
                  config['output']['vocabulary_file'] = os.path.join(base_dir, scaler_subdir, 'vocabulary.txt')
                  logger.info(f"Menambahkan path default vocabulary_file: {config['output']['vocabulary_file']}")

             # Tambahkan path 'model_save_file' default ke config['output'] jika belum ada
             # Gunakan .get() untuk mengakses 'model_subdir' dengan aman
             if 'model_save_file' not in config['output']:
                  model_subdir = config['output'].get('model_subdir', 'saved_model') # Gunakan default 'saved_model' jika model_subdir tidak ada
                  # Gunakan base_dir dari config['output'] untuk membangun path lengkap
                  base_dir = config['output'].get('base_dir', '') # Gunakan default '' jika base_dir tidak ada
                  config['output']['model_save_file'] = os.path.join(base_dir, model_subdir, 'best_model.h5') # Sesuaikan dengan nama file .h5
                  logger.info(f"Menambahkan path default model_save_file: {config['output']['model_save_file']}")

        else:
             logger.error("Kunci 'output' ada tetapi bukan dictionary. Pastikan format file konfigurasi benar.")
             exit(1) # Keluar jika format output salah


        # Perbarui log warning untuk load_path jika mode bukan initial_train dan load_path terlihat salah format
        # Pastikan config['model'] adalah dictionary sebelum mengaksesnya
        if isinstance(config['model'], dict):
             if config['mode'] in ['incremental_learn', 'predict_only']:
                 if 'load_path' in config['model']:
                     # Check if load_path ends with .h5 or is a directory (for SavedModel)
                     # If saving is forced to .h5, the load path should ideally be .h5
                     expected_extension = '.h5'
                     if not config['model']['load_path'].endswith(expected_extension):
                          logger.warning(f"model.load_path ({config['model']['load_path']}) di config sepertinya bukan file {expected_extension}. Pastikan sesuai dengan format simpan ('{config['output']['model_save_file']}').")
                 else:
                      logger.error("Mode inkremental_learn atau predict_only dipilih, tetapi model.load_path tidak ada di file konfigurasi.")
                      exit(1) # Keluar jika load_path tidak ada di mode yang membutuhkannya
        else:
            logger.error("Kunci 'model' ada tetapi bukan dictionary. Pastikan format file konfigurasi benar.")
            exit(1)


        # --- Jalankan Pipeline Utama ---
        # Pastikan semua path relatif di config di-resolve jika perlu di run_pipeline
        # run_pipeline() saat ini menggunakan os.path.join(output_dir, ...) yang bergantung pada base_dir
        # Pastikan base_dir ada dan digunakan konsisten. base_dir juga sudah divalidasi ada di essential_keys.
        if isinstance(config['output'], dict) and 'base_dir' in config['output']:
             run_pipeline(config)
        else:
             # Ini seharusnya tidak tercapai jika essential_keys dan validasi dictionary di atas benar
             logger.error("Konfigurasi output atau base_dir tidak ditemukan setelah validasi.")
             exit(1)


    else:
        # Ini seharusnya tidak tercapai jika exit(1) dipanggil di blok try/except di atas
        logger.error("Tidak ada konfigurasi yang tersedia setelah mencoba memuat file.")
        exit(1)
